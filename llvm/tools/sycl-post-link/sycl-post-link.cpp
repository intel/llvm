//===- sycl-post-link.cpp - SYCL post-link device code processing tool ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This utility splits an input module into smaller ones.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SetVector.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/IPO.h"
#include <memory>

using namespace llvm;

cl::OptionCategory ExtractCat{"sycl-post-link Options"};

// InputFilename - The filename to read from.
static cl::opt<std::string> InputFilename{
    cl::Positional, cl::desc("<input bitcode file>"), cl::init("-"),
    cl::value_desc("filename")};

static cl::opt<std::string> BaseOutputFilename{
    "o",
    cl::desc("Specify base output filename, output filenames will be saved "
             "into out_0.bc, out_1.bc, ..., out_0.txt, out_1.txt, ...."),
    cl::value_desc("filename"), cl::init("-"), cl::cat(ExtractCat)};

static cl::opt<std::string> OutputIRFilesList{
    "ir-files-list", cl::desc("Specify output filename for IR files list"),
    cl::value_desc("filename"), cl::init("-"), cl::cat(ExtractCat)};

static cl::opt<std::string> OutputTxtFilesList{
    "txt-files-list", cl::desc("Specify output filename for txt files list"),
    cl::value_desc("filename"), cl::init("-"), cl::cat(ExtractCat)};

static cl::opt<bool> Force{"f", cl::desc("Enable binary output on terminals"),
                           cl::cat(ExtractCat)};

static cl::opt<bool> OutputAssembly{"S",
                                    cl::desc("Write output as LLVM assembly"),
                                    cl::Hidden, cl::cat(ExtractCat)};

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
    Module &M,
    std::map<std::string, std::vector<Function *>> &ResKernelsSet) {
  for (auto &F : M.functions()) {
    if (F.getCallingConv() == CallingConv::SPIR_KERNEL &&
        F.hasFnAttribute("sycl-module-id")) {
      auto Id = F.getFnAttribute("sycl-module-id");
      auto Val = Id.getValueAsString();
      ResKernelsSet[Val].push_back(&F);
    }
  }
}

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
      for (auto &BB : *F) {
        for (auto &I : BB) {
          if (CallBase *CB = dyn_cast<CallBase>(&I)) {
            if (Function *CF = CB->getCalledFunction()) {
              if (!CF->isDeclaration() && !GVs.count(CF)) {
                GVs.insert(CF);
                Workqueue.push_back(CF);
              }
            }
          }
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
    ResSymbolsLists.push_back(SymbolsList);
  }
}

static void saveResults(std::vector<std::unique_ptr<Module>> &ResModules,
                        std::vector<std::string> &ResSymbolsLists) {
  int NumOfFile = 0;
  std::error_code EC;
  std::string IRFilesList;
  std::string TxtFilesList;
  for (size_t I = 0; I < ResModules.size(); ++I) {
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
      "Splits a fully linked module into smaller ones. Groups kernels\n"
      "using function attribute 'sycl-module-id', i.e. kernels with the same\n"
      "values of the 'sycl-module-id' attribute will be put into the same\n"
      "module. For each produced module a text file containing the names of\n"
      "all spir kernels in it is generated. Optionally can generate lists of\n"
      "produced files. Usage:\n"
      "sycl-post-link -S linked.ll -ir-files-list=ir.txt \\\n"
      "-txt-files-list=files.txt -o out \\\n"
      "This command will produce several llvm IR files: out_0.ll, "
      "out_1.ll...,\n"
      "several text files containing spir kernel names out_0.txt, "
      "out_1.txt,...,\n"
      "and two filelists in ir.txt and files.txt.\n");

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
