//===- sycl-post-link.cpp - SYCL post-link device code processing tool ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This source is a collection of utilities run on device code's LLVM IR before
// handing off to back-end for further compilation or emitting SPIRV. The
// utilities are:
// - module splitter to split a big input module into smaller ones
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SetVector.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <memory>

using namespace llvm;

cl::OptionCategory ExtractCat{"sycl-post-link Options"};

// InputFilename - The filename to read from.
static cl::opt<std::string> InputFilename{
    cl::Positional, cl::desc("<input bitcode file>"), cl::init("-"),
    cl::value_desc("filename")};

static cl::opt<std::string> BaseOutputFilename{
    "o",
    cl::desc("Specify base output filename. E.g. if base is 'out', then output "
             "filenames will be saved "
             "into out_0.bc, out_1.bc, ..., out_0.txt, out_1.txt, ...."),
    cl::value_desc("filename"), cl::init("-"), cl::cat(ExtractCat)};

// Module splitter produces multiple IR files. IR files list is a file list
// with prodced IR modules files names.
static cl::opt<std::string> OutputIRFilesList{
    "ir-files-list", cl::desc("Specify output filename for IR files list"),
    cl::value_desc("filename"), cl::init("-"), cl::cat(ExtractCat)};

// Module splitter produces multiple TXT files. These files contain kernel names
// list presented in a produced module. TXT files list is a file list
// with produced TXT files names.
static cl::opt<std::string> OutputTxtFilesList{
    "txt-files-list", cl::desc("Specify output filename for txt files list"),
    cl::value_desc("filename"), cl::init("-"), cl::cat(ExtractCat)};

static cl::opt<bool> Force{"f", cl::desc("Enable binary output on terminals"),
                           cl::cat(ExtractCat)};

static cl::opt<bool> OutputAssembly{"S",
                                    cl::desc("Write output as LLVM assembly"),
                                    cl::Hidden, cl::cat(ExtractCat)};

static cl::opt<bool> OneKernelPerModule{
    "one-kernel", cl::desc("Emit a separate module for each kernel"),
    cl::init(false), cl::cat(ExtractCat)};

static void error(const Twine &Msg) {
  errs() << "sycl-post-link: " << Msg << '\n';
  exit(1);
}

static void error(std::error_code EC, const Twine &Prefix) {
  if (EC)
    error(Prefix + ": " + EC.message());
}

static void writeToFile(std::string Filename, std::string Content) {
  std::error_code EC;
  raw_fd_ostream OS{Filename, EC, sys::fs::OpenFlags::OF_None};
  error(EC, "error opening the file '" + Filename + "'");
  OS.write(Content.data(), Content.size());
  OS.close();
}

static void collectKernelsSet(
    Module &M, std::map<std::string, std::vector<Function *>> &ResKernelsSet) {
  for (auto &F : M.functions()) {
    if (F.getCallingConv() == CallingConv::SPIR_KERNEL) {
      if (OneKernelPerModule) {
        ResKernelsSet[F.getName()].push_back(&F);
      } else if (F.hasFnAttribute("sycl-module-id")) {
        auto Id = F.getFnAttribute("sycl-module-id");
        auto Val = Id.getValueAsString();
        ResKernelsSet[Val].push_back(&F);
      }
    }
  }
}

// Splits input LLVM IR module M into smaller ones.
// Input parameter KernelsSet is a map containing groups of kernels with same
// values in the sycl-module-id attribute. For each group of kernels a separate
// IR module will be produced.
// ResModules and ResSymbolsLists are output parameters.
// Result modules are stored into
// ResModules vector. For each result module set of kernel names is collected.
// Sets of kernel names are stored into ResSymbolsLists.
static void
splitModule(Module &M,
            std::map<std::string, std::vector<Function *>> &KernelsSet,
            std::vector<std::unique_ptr<Module>> &ResModules,
            std::vector<std::string> &ResSymbolsLists) {
  for (auto &It : KernelsSet) {
    // For each group of kernels collect all dependencies.
    SetVector<GlobalValue *> GVs;
    std::vector<llvm::Function *> Workqueue;
    std::string SymbolsList;

    for (auto &F : It.second) {
      GVs.insert(F);
      Workqueue.push_back(F);
      SymbolsList =
          (Twine(SymbolsList) + Twine(F->getName()) + Twine("\n")).str();
    }

    while (!Workqueue.empty()) {
      Function *F = &*Workqueue.back();
      Workqueue.pop_back();
      for (auto &I : instructions(F)) {
        if (CallBase *CB = dyn_cast<CallBase>(&I))
          if (Function *CF = CB->getCalledFunction())
            if (!CF->isDeclaration() && !GVs.count(CF)) {
              GVs.insert(CF);
              Workqueue.push_back(CF);
            }
      }
    }

    // It's not easy to trace global variable's uses inside needed functions
    // because global variable can be used inside a combination of operators, so
    // mark all global variables as needed and remove dead ones after
    // extraction.
    for (auto &G : M.globals()) {
      GVs.insert(&G);
    }

    // Clone the module, understand which globals we need to extract from the
    // clone.
    ValueToValueMapTy VMap;
    std::unique_ptr<Module> MClone = CloneModule(M, VMap);
    std::vector<GlobalValue *> GVsInClone(GVs.size());
    int I = 0;
    for (GlobalValue *GV : GVs) {
      GVsInClone[I] = cast<GlobalValue>(VMap[GV]);
      ++I;
    }

    // TODO: Use the new PassManager instead?
    legacy::PassManager Extract;

    // Extract needed globals.
    Extract.add(createGVExtractionPass(GVsInClone, /* deleteS */ false));
    Extract.run(*MClone.get());

    // Extactor pass sets external linkage to all globals. Return linkage back.
    for (auto &G : MClone->globals()) {
      if (G.getVisibility() == GlobalValue::HiddenVisibility) {
        G.setLinkage(GlobalValue::InternalLinkage);
      }
    }

    legacy::PassManager Passes;
    // Do cleanup.
    Passes.add(createGlobalDCEPass());           // Delete unreachable globals.
    Passes.add(createStripDeadDebugInfoPass());  // Remove dead debug info.
    Passes.add(createStripDeadPrototypesPass()); // Remove dead func decls.
    Passes.run(*MClone.get());

    // Save results.
    ResModules.push_back(std::move(MClone));
    ResSymbolsLists.push_back(std::move(SymbolsList));
  }
}

// Saves specified collections of llvm IR modules and corresponding lists of
// kernel names to files. Saves IR files list and TXT files list if user
// specified corresponding filenames.
static void saveResults(std::vector<std::unique_ptr<Module>> &ResModules,
                        std::vector<std::string> &ResSymbolsLists) {
  int NumOfFile = 0;
  std::string IRFilesList;
  std::string TxtFilesList;
  for (size_t I = 0; I < ResModules.size(); ++I) {
    std::error_code EC;
    std::string CurOutFileName = BaseOutputFilename + "_" +
                                 std::to_string(NumOfFile) +
                                 ((OutputAssembly) ? ".ll" : ".bc");

    raw_fd_ostream Out{CurOutFileName, EC, sys::fs::OF_None};
    error(EC, "error opening the file '" + CurOutFileName + "'");

    // TODO: Use the new PassManager instead?
    legacy::PassManager PrintModule;

    if (OutputAssembly)
      PrintModule.add(createPrintModulePass(Out, ""));
    else if (Force || !CheckBitcodeOutputToConsole(Out, true))
      PrintModule.add(createBitcodeWriterPass(Out));
    PrintModule.run(*ResModules[I].get());

    IRFilesList =
        (Twine(IRFilesList) + Twine(CurOutFileName) + Twine("\n")).str();

    CurOutFileName =
        BaseOutputFilename + "_" + std::to_string(NumOfFile) + ".txt";
    writeToFile(CurOutFileName, ResSymbolsLists[I]);

    TxtFilesList =
        (Twine(TxtFilesList) + Twine(CurOutFileName) + Twine("\n")).str();

    ++NumOfFile;
  }

  if (OutputIRFilesList != "-")
    writeToFile(OutputIRFilesList, IRFilesList);
  if (OutputTxtFilesList != "-")
    writeToFile(OutputTxtFilesList, TxtFilesList);
}

int main(int argc, char **argv) {
  InitLLVM X{argc, argv};

  LLVMContext Context;
  cl::HideUnrelatedOptions(ExtractCat);
  cl::ParseCommandLineOptions(
      argc, argv,
      "SYCL post-link device code processing tool.\n"
      "This is a collection of utilities run on device code's LLVM IR before\n"
      "handing off to back-end for further compilation or emitting SPIRV.\n"
      "The utilities are:\n"
      "- Module splitter to split a big input module into smaller ones.\n"
      "  Groups kernels using function attribute 'sycl-module-id', i.e.\n"
      "  kernels with the same values of the 'sycl-module-id' attribute will\n"
      "  be put into the same module. If --one-kernel option is specified,\n"
      "  one module per kernel will be emitted.\n"
      "  For each produced module a text file\n"
      "  containing the names of all spir kernels in it is generated.\n"
      "  Optionally can generate lists produced files.\n"
      "  Usage:\n"
      "  sycl-post-link -S linked.ll -ir-files-list=ir.txt \\\n"
      "  -txt-files-list=files.txt -o out \\\n"
      "  This command will produce several llvm IR files: out_0.ll, "
      "  out_1.ll...,\n"
      "  several text files containing spir kernel names out_0.txt, "
      "  out_1.txt,...,\n"
      "  and two filelists in ir.txt and files.txt.\n");

  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseIRFile(InputFilename, Err, Context);

  if (!M.get()) {
    Err.print(argv[0], errs());
    return 1;
  }

  std::map<std::string, std::vector<Function *>> GlobalsSet;

  collectKernelsSet(*M.get(), GlobalsSet);

  std::vector<std::unique_ptr<Module>> ResultModules;
  std::vector<std::string> ResultSymbolsLists;

  splitModule(*M.get(), GlobalsSet, ResultModules, ResultSymbolsLists);

  if (BaseOutputFilename == "-")
    BaseOutputFilename = "a.out";

  saveResults(ResultModules, ResultSymbolsLists);

  return 0;
}
