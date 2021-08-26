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
// - specialization constant intrinsic transformation
//===----------------------------------------------------------------------===//

#include "SPIRKernelParamOptInfo.h"
#include "SYCLDeviceLibReqMask.h"
#include "SpecConstants.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/GenXIntrinsics/GenXSPIRVWriterAdaptor.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/SYCLLowerIR/LowerESIMD.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PropertySetIO.h"
#include "llvm/Support/SimpleTable.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <algorithm>
#include <memory>

using namespace llvm;

using string_vector = std::vector<std::string>;

cl::OptionCategory PostLinkCat{"sycl-post-link options"};

// Column names in the output file table. Must match across tools -
// clang/lib/Driver/Driver.cpp, sycl-post-link.cpp, ClangOffloadWrapper.cpp
static constexpr char COL_CODE[] = "Code";
static constexpr char COL_SYM[] = "Symbols";
static constexpr char COL_PROPS[] = "Properties";
static constexpr char ATTR_SYCL_MODULE_ID[] = "sycl-module-id";

// InputFilename - The filename to read from.
static cl::opt<std::string> InputFilename{
    cl::Positional, cl::desc("<input bitcode file>"), cl::init("-"),
    cl::value_desc("filename")};

static cl::opt<std::string> OutputDir{
    "out-dir",
    cl::desc(
        "Directory where files listed in the result file table will be output"),
    cl::value_desc("dirname"), cl::cat(PostLinkCat)};

static cl::opt<std::string> OutputFilename{"o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"), cl::cat(PostLinkCat)};

static cl::opt<bool> Force{"f", cl::desc("Enable binary output on terminals"),
                           cl::cat(PostLinkCat)};

static cl::opt<bool> IROutputOnly{
    "ir-output-only", cl::desc("Output single IR file"), cl::cat(PostLinkCat)};

static cl::opt<bool> OutputAssembly{"S",
                                    cl::desc("Write output as LLVM assembly"),
                                    cl::Hidden, cl::cat(PostLinkCat)};

static cl::opt<bool> SplitEsimd{"split-esimd",
                                cl::desc("Split SYCL and ESIMD kernels"),
                                cl::cat(PostLinkCat)};

// TODO Design note: sycl-post-link should probably separate different kinds of
// its functionality on  logical and source level:
//  - LLVM IR module splitting
//  - Running LLVM IR passes on resulting modules
//  - Generating additional files (like spec constants, dead arg info,...)
// The tool itself could be just a "driver" creating needed pipelines from the
// above actions. This could help make the tool structure clearer and more
// maintainable.

static cl::opt<bool> LowerEsimd{
    "lower-esimd", cl::desc("Lower ESIMD constructs"), cl::cat(PostLinkCat)};

static cl::opt<bool>
    OptLevelO0("O0", cl::desc("Optimization level 0. Similar to clang -O0"),
               cl::cat(PostLinkCat));

static cl::opt<bool>
    OptLevelO1("O1", cl::desc("Optimization level 1. Similar to clang -O1"),
               cl::cat(PostLinkCat));

static cl::opt<bool>
    OptLevelO2("O2", cl::desc("Optimization level 2. Similar to clang -O2"),
               cl::cat(PostLinkCat));

static cl::opt<bool> OptLevelOs(
    "Os",
    cl::desc(
        "Like -O2 with extra optimizations for size. Similar to clang -Os"),
    cl::cat(PostLinkCat));

static cl::opt<bool> OptLevelOz(
    "Oz",
    cl::desc("Like -Os but reduces code size further. Similar to clang -Oz"),
    cl::cat(PostLinkCat));

static cl::opt<bool>
    OptLevelO3("O3", cl::desc("Optimization level 3. Similar to clang -O3"),
               cl::cat(PostLinkCat));

enum IRSplitMode {
  SPLIT_PER_TU,     // one module per translation unit
  SPLIT_PER_KERNEL, // one module per kernel
  SPLIT_AUTO        // automatically select split mode
};

static cl::opt<IRSplitMode> SplitMode(
    "split", cl::desc("split input module"), cl::Optional, cl::init(SPLIT_AUTO),
    cl::values(
        clEnumValN(SPLIT_PER_TU, "source",
                   "1 output module per source (translation unit)"),
        clEnumValN(SPLIT_PER_KERNEL, "kernel", "1 output module per kernel"),
        clEnumValN(SPLIT_AUTO, "auto", "Choose split mode automatically")),
    cl::cat(PostLinkCat));

static cl::opt<bool> DoSymGen{"symbols",
                              cl::desc("generate exported symbol files"),
                              cl::cat(PostLinkCat)};

enum SpecConstMode { SC_USE_RT_VAL, SC_USE_DEFAULT_VAL };

static cl::opt<SpecConstMode> SpecConstLower{
    "spec-const",
    cl::desc("lower and generate specialization constants information"),
    cl::Optional,
    cl::init(SC_USE_RT_VAL),
    cl::values(
        clEnumValN(SC_USE_RT_VAL, "rt", "spec constants are set at runtime"),
        clEnumValN(SC_USE_DEFAULT_VAL, "default",
                   "set spec constants to C++ defaults")),
    cl::cat(PostLinkCat)};

static cl::opt<bool> EmitKernelParamInfo{
    "emit-param-info", cl::desc("emit kernel parameter optimization info"),
    cl::cat(PostLinkCat)};

static cl::opt<bool> EmitProgramMetadata{"emit-program-metadata",
                                         cl::desc("emit SYCL program metadata"),
                                         cl::cat(PostLinkCat)};

struct ImagePropSaveInfo {
  bool NeedDeviceLibReqMask;
  bool DoSpecConst;
  bool SetSpecConstAtRT;
  bool SpecConstsMet;
  bool EmitKernelParamInfo;
  bool EmitProgramMetadata;
  bool IsEsimdKernel;
};

static void error(const Twine &Msg) {
  errs() << "sycl-post-link: " << Msg << '\n';
  exit(1);
}

static void checkError(std::error_code EC, const Twine &Prefix) {
  if (EC)
    error(Prefix + ": " + EC.message());
}

static void writeToFile(std::string Filename, std::string Content) {
  std::error_code EC;
  raw_fd_ostream OS{Filename, EC, sys::fs::OpenFlags::OF_None};
  checkError(EC, "error opening the file '" + Filename + "'");
  OS.write(Content.data(), Content.size());
  OS.close();
}

// Describes scope covered by each entry in the module-kernel map populated by
// the collectKernelModuleMap function.
enum KernelMapEntryScope {
  Scope_PerKernel, // one entry per kernel
  Scope_PerModule, // one entry per module
  Scope_Global     // single entry in the map for all kernels
};

static KernelMapEntryScope selectDeviceCodeSplitScopeAutomatically(Module &M) {
  if (IROutputOnly) {
    // We allow enabling auto split mode even in presence of -ir-output-only
    // flag, but in this case we are limited by it so we can't do any split at
    // all.
    return Scope_Global;
  }

  for (const auto &F : M.functions()) {
    // There are functions marked with [[intel::device_indirectly_callable]]
    // attribute, because it instructs us to make this function available to the
    // whole program as it was compiled as a single module.
    if (F.hasFnAttribute("referenced-indirectly"))
      return Scope_Global;
    if (F.isDeclaration())
      continue;
    // There are indirect calls in the module, which means that we don't know
    // how to group functions so both caller and callee of indirect call are in
    // the same module.
    for (const auto &BB : F) {
      for (const auto &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          if (!CI->getCalledFunction())
            return Scope_Global;
        }
      }
    }
  }

  // At the moment, we assume that per-source split is the best way of splitting
  // device code and can always be used execpt for cases handled above.
  return Scope_PerModule;
}

// Return true if the function is a SPIRV or SYCL builtin, e.g.
// _Z28__spirv_GlobalInvocationId_xv
static bool funcIsSpirvSyclBuiltin(StringRef FName) {
  if (!FName.consume_front("_Z"))
    return false;
  // now skip the digits
  FName = FName.drop_while([](char C) { return std::isdigit(C); });

  return FName.startswith("__spirv_") || FName.startswith("__sycl_");
}

// This function decides how kernels of the input module M will be distributed
// ("split") into multiple modules based on the command options and IR
// attributes. The decision is recorded in the output map parameter
// ResKernelModuleMap which maps some key to a group of kernels. Each such group
// along with IR it depends on (globals, functions from its call graph,...) will
// constitute a separate module.
static void collectKernelModuleMap(
    Module &M, std::map<StringRef, std::vector<Function *>> &ResKernelModuleMap,
    KernelMapEntryScope EntryScope) {

  // Process module entry points: kernels and SYCL_EXTERNAL functions.
  // Only they have sycl-module-id attribute, so any other unrefenced
  // functions are dropped. SPIRV and SYCL builtin functions are not
  // considered as module entry points.
  for (auto &F : M.functions()) {
    if (F.hasFnAttribute(ATTR_SYCL_MODULE_ID) &&
        !funcIsSpirvSyclBuiltin(F.getName())) {
      switch (EntryScope) {
      case Scope_PerKernel:
        ResKernelModuleMap[F.getName()].push_back(&F);
        break;
      case Scope_PerModule: {
        Attribute Id = F.getFnAttribute(ATTR_SYCL_MODULE_ID);
        StringRef Val = Id.getValueAsString();
        ResKernelModuleMap[Val].push_back(&F);
        break;
      }
      case Scope_Global:
        // the map key is not significant here
        ResKernelModuleMap["<GLOBAL>"].push_back(&F);
        break;
      }
    } else if (EntryScope == Scope_PerModule &&
               F.getCallingConv() == CallingConv::SPIR_KERNEL) {
      // TODO It may make sense to group all kernels w/o the attribute into
      // a separate module rather than issuing an error. Should probably be
      // controlled by an option.
      // Functions with spir_func calling convention are allowed to not have
      // a sycl-module-id attribute.
      error("no '" + Twine(ATTR_SYCL_MODULE_ID) + "' attribute in kernel '" +
            F.getName() + "', per-module split not possible");
    }
  }
}

enum HasAssertStatus { No_Assert, Assert, Assert_Indirect };

// Go through function call graph searching for assert call.
static HasAssertStatus hasAssertInFunctionCallGraph(llvm::Function *Func) {
  // Map holds the info about assertions in already examined functions:
  // true  - if there is an assertion in underlying functions,
  // false - if there are definetely no assertions in underlying functions.
  static std::map<llvm::Function *, bool> hasAssertionInCallGraphMap;
  std::vector<llvm::Function *> FuncCallStack;

  static std::vector<llvm::Function *> isIndirectlyCalledInGraph;

  std::vector<llvm::Function *> Workstack;
  Workstack.push_back(Func);

  while (!Workstack.empty()) {
    Function *F = Workstack.back();
    Workstack.pop_back();
    if (F != Func)
      FuncCallStack.push_back(F);

    bool HasIndirectlyCalledAttr = false;
    if (std::find(isIndirectlyCalledInGraph.begin(),
                  isIndirectlyCalledInGraph.end(),
                  F) != isIndirectlyCalledInGraph.end())
      HasIndirectlyCalledAttr = true;
    else if (F->hasFnAttribute("referenced-indirectly")) {
      HasIndirectlyCalledAttr = true;
      isIndirectlyCalledInGraph.push_back(F);
    }

    bool IsLeaf = true;
    for (auto &I : instructions(F)) {
      if (!isa<CallBase>(&I))
        continue;

      Function *CF = cast<CallBase>(&I)->getCalledFunction();
      if (!CF)
        continue;

      bool IsIndirectlyCalled =
          HasIndirectlyCalledAttr ||
          std::find(isIndirectlyCalledInGraph.begin(),
                    isIndirectlyCalledInGraph.end(),
                    CF) != isIndirectlyCalledInGraph.end();

      // Return if we've already discovered if there are asserts in the
      // function call graph.
      auto HasAssert = hasAssertionInCallGraphMap.find(CF);
      if (HasAssert != hasAssertionInCallGraphMap.end()) {
        // If we know, that this function does not contain assert, we still
        // should investigate another instructions in the function.
        if (!HasAssert->second)
          continue;

        return IsIndirectlyCalled ? Assert_Indirect : Assert;
      }

      if (CF->getName().startswith("__devicelib_assert_fail")) {
        // Mark all the functions above in call graph as ones that can call
        // assert.
        for (auto *It : FuncCallStack)
          hasAssertionInCallGraphMap[It] = true;

        hasAssertionInCallGraphMap[Func] = true;
        hasAssertionInCallGraphMap[CF] = true;

        return IsIndirectlyCalled ? Assert_Indirect : Assert;
      }

      if (!CF->isDeclaration()) {
        Workstack.push_back(CF);
        IsLeaf = false;
        if (HasIndirectlyCalledAttr)
          isIndirectlyCalledInGraph.push_back(CF);
      }
    }

    if (IsLeaf && !FuncCallStack.empty()) {
      // Mark the leaf function as one that definetely does not call assert.
      hasAssertionInCallGraphMap[FuncCallStack.back()] = false;
      FuncCallStack.clear();
    }
  }
  return No_Assert;
}

// Gets reqd_work_group_size information for function Func.
static std::vector<uint32_t>
getKernelReqdWorkGroupSizeMetadata(const Function &Func) {
  auto ReqdWorkGroupSizeMD = Func.getMetadata("reqd_work_group_size");
  if (!ReqdWorkGroupSizeMD)
    return {};
  // TODO: Remove 3-operand assumption when it is relaxed.
  assert(ReqdWorkGroupSizeMD->getNumOperands() == 3);
  uint32_t X = mdconst::extract<ConstantInt>(ReqdWorkGroupSizeMD->getOperand(0))
                   ->getZExtValue();
  uint32_t Y = mdconst::extract<ConstantInt>(ReqdWorkGroupSizeMD->getOperand(1))
                   ->getZExtValue();
  uint32_t Z = mdconst::extract<ConstantInt>(ReqdWorkGroupSizeMD->getOperand(2))
                   ->getZExtValue();
  return {X, Y, Z};
}

// Input parameter KernelModuleMap is a map containing groups of kernels with
// same values of the sycl-module-id attribute. ResSymbolsLists is a vector of
// kernel name lists. Each vector element is a string with kernel names from the
// same module separated by \n.
// The function saves names of kernels from one group to a single std::string
// and stores this string to the ResSymbolsLists vector.
static void collectSymbolsLists(
    std::map<StringRef, std::vector<Function *>> &KernelModuleMap,
    string_vector &ResSymbolsLists) {
  for (auto &It : KernelModuleMap) {
    std::string SymbolsList;
    for (auto &F : It.second) {
      SymbolsList =
          (Twine(SymbolsList) + Twine(F->getName()) + Twine("\n")).str();
    }
    ResSymbolsLists.push_back(std::move(SymbolsList));
  }
}

// Input parameter KernelModuleMap is a map containing groups of kernels with
// same values of the sycl-module-id attribute. For each group of kernels a
// separate IR module will be produced.
// ResModules is a vector of produced modules.
// The function splits input LLVM IR module M into smaller ones and stores them
// to the ResModules vector.
static void
splitModule(Module &M,
            std::map<StringRef, std::vector<Function *>> &KernelModuleMap,
            std::vector<std::unique_ptr<Module>> &ResModules) {
  for (auto &It : KernelModuleMap) {
    // For each group of kernels collect all dependencies.
    SetVector<const GlobalValue *> GVs;
    std::vector<llvm::Function *> Workqueue;

    for (auto &F : It.second) {
      GVs.insert(F);
      Workqueue.push_back(F);
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
    // cloning.
    for (auto &G : M.globals()) {
      GVs.insert(&G);
    }

    ValueToValueMapTy VMap;
    // Clone definitions only for needed globals. Others will be added as
    // declarations and removed later.
    std::unique_ptr<Module> MClone = CloneModule(
        M, VMap, [&](const GlobalValue *GV) { return GVs.count(GV); });

    // TODO: Use the new PassManager instead?
    legacy::PassManager Passes;
    // Do cleanup.
    Passes.add(createGlobalDCEPass());           // Delete unreachable globals.
    Passes.add(createStripDeadDebugInfoPass());  // Remove dead debug info.
    Passes.add(createStripDeadPrototypesPass()); // Remove dead func decls.
    Passes.run(*MClone.get());

    // Save results.
    ResModules.push_back(std::move(MClone));
  }
}

static std::string makeResultFileName(Twine Ext, int I, StringRef Suffix) {
  const StringRef Dir0 = OutputDir.getNumOccurrences() > 0
                             ? OutputDir
                             : sys::path::parent_path(OutputFilename);
  const StringRef Sep = sys::path::get_separator();
  std::string Dir = Dir0.str();
  if (!Dir0.empty() && !Dir0.endswith(Sep))
    Dir += Sep.str();
  return Dir + sys::path::stem(OutputFilename).str() + "_" + Suffix.str() +
         std::to_string(I) + Ext.str();
}

static void saveModule(Module &M, StringRef OutFilename) {
  std::error_code EC;
  raw_fd_ostream Out{OutFilename, EC, sys::fs::OF_None};
  checkError(EC, "error opening the file '" + OutFilename + "'");

  // TODO: Use the new PassManager instead?
  legacy::PassManager PrintModule;

  if (OutputAssembly)
    PrintModule.add(createPrintModulePass(Out, ""));
  else if (Force || !CheckBitcodeOutputToConsole(Out))
    PrintModule.add(createBitcodeWriterPass(Out));
  PrintModule.run(M);
}

// Saves specified collection of llvm IR modules to files.
// Saves file list if user specified corresponding filename.
static string_vector
saveResultModules(std::vector<std::unique_ptr<Module>> &ResModules,
                  StringRef Suffix) {
  string_vector Res;

  for (size_t I = 0; I < ResModules.size(); ++I) {
    std::error_code EC;
    StringRef FileExt = (OutputAssembly) ? ".ll" : ".bc";
    std::string CurOutFileName = makeResultFileName(FileExt, I, Suffix);
    saveModule(*ResModules[I].get(), CurOutFileName);
    Res.emplace_back(std::move(CurOutFileName));
  }
  return Res;
}

static string_vector saveDeviceImageProperty(
    const std::vector<std::unique_ptr<Module>> &ResultModules,
    const ImagePropSaveInfo &ImgPSInfo) {
  string_vector Res;
  legacy::PassManager GetSYCLDeviceLibReqMask;
  SYCLDeviceLibReqMaskPass *SDLReqMaskLegacyPass =
      new SYCLDeviceLibReqMaskPass();
  GetSYCLDeviceLibReqMask.add(SDLReqMaskLegacyPass);
  for (size_t I = 0; I < ResultModules.size(); ++I) {
    llvm::util::PropertySetRegistry PropSet;
    if (ImgPSInfo.NeedDeviceLibReqMask) {
      GetSYCLDeviceLibReqMask.run(*ResultModules[I]);
      uint32_t MRMask = SDLReqMaskLegacyPass->getSYCLDeviceLibReqMask();
      std::map<StringRef, uint32_t> RMEntry = {{"DeviceLibReqMask", MRMask}};
      PropSet.add(llvm::util::PropertySetRegistry::SYCL_DEVICELIB_REQ_MASK,
                  RMEntry);
    }
    if (ImgPSInfo.DoSpecConst) {
      if (ImgPSInfo.SpecConstsMet) {
        // extract spec constant maps per each module
        SpecIDMapTy TmpSpecIDMap;
        SpecConstantsPass::collectSpecConstantMetadata(*ResultModules[I].get(),
                                                       TmpSpecIDMap);
        PropSet.add(
            llvm::util::PropertySetRegistry::SYCL_SPECIALIZATION_CONSTANTS,
            TmpSpecIDMap);

        // Add property with the default values of spec constants only in native
        // (default) mode.
        if (!ImgPSInfo.SetSpecConstAtRT) {
          std::vector<char> DefaultValues;
          SpecConstantsPass::collectSpecConstantDefaultValuesMetadata(
              *ResultModules[I].get(), DefaultValues);
          PropSet.add(llvm::util::PropertySetRegistry::
                          SYCL_SPEC_CONSTANTS_DEFAULT_VALUES,
                      "all", DefaultValues);
        }
      }
    }
    if (ImgPSInfo.EmitKernelParamInfo) {
      // extract kernel parameter optimization info per module
      ModuleAnalysisManager MAM;
      // Register required analysis
      MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
      // Register the payload analysis
      MAM.registerPass([&] { return SPIRKernelParamOptInfoAnalysis(); });
      SPIRKernelParamOptInfo PInfo =
          MAM.getResult<SPIRKernelParamOptInfoAnalysis>(*ResultModules[I]);

      // convert analysis results into properties and record them
      llvm::util::PropertySet &Props =
          PropSet[llvm::util::PropertySetRegistry::SYCL_KERNEL_PARAM_OPT_INFO];

      for (const auto &NameInfoPair : PInfo) {
        const llvm::BitVector &Bits = NameInfoPair.second;
        const llvm::ArrayRef<uintptr_t> Arr = NameInfoPair.second.getData();
        const unsigned char *Data =
            reinterpret_cast<const unsigned char *>(Arr.begin());
        llvm::util::PropertyValue::SizeTy DataBitSize = Bits.size();
        Props.insert(std::make_pair(
            NameInfoPair.first, llvm::util::PropertyValue(Data, DataBitSize)));
      }
    }

    // Metadata names may be composite so we keep them alive until the
    // properties have been written.
    SmallVector<std::string, 4> MetadataNames;
    if (ImgPSInfo.EmitProgramMetadata) {
      auto &ProgramMetadata =
          PropSet[llvm::util::PropertySetRegistry::SYCL_PROGRAM_METADATA];

      // Add reqd_work_group_size information to program metadata
      for (const Function &Func : ResultModules[I]->functions()) {
        std::vector<uint32_t> KernelReqdWorkGroupSize =
            getKernelReqdWorkGroupSizeMetadata(Func);
        if (KernelReqdWorkGroupSize.empty())
          continue;
        MetadataNames.push_back(Func.getName().str() + "@reqd_work_group_size");
        ProgramMetadata.insert({MetadataNames.back(), KernelReqdWorkGroupSize});
      }
    }

    if (ImgPSInfo.IsEsimdKernel) {
      PropSet[llvm::util::PropertySetRegistry::SYCL_MISC_PROP].insert(
          {"isEsimdImage", true});
    }

    {
      Module *M = ResultModules[I].get();
      bool HasIndirectlyCalledAssert = false;
      std::vector<llvm::Function *> Kernels;
      for (auto &F : M->functions()) {
        // TODO: handle SYCL_EXTERNAL functions for dynamic linkage.
        // TODO: handle function pointers.
        if (F.getCallingConv() != CallingConv::SPIR_KERNEL)
          continue;

        Kernels.push_back(&F);
        if (HasIndirectlyCalledAssert)
          continue;

        HasAssertStatus HasAssert = hasAssertInFunctionCallGraph(&F);
        switch (HasAssert) {
        case Assert:
          PropSet[llvm::util::PropertySetRegistry::SYCL_ASSERT_USED].insert(
              {F.getName(), true});
          break;
        case Assert_Indirect:
          HasIndirectlyCalledAssert = true;
          break;
        case No_Assert:
          break;
        }
      }

      if (HasIndirectlyCalledAssert) {
        for (auto *F : Kernels)
          PropSet[llvm::util::PropertySetRegistry::SYCL_ASSERT_USED].insert(
              {F->getName(), true});
      }
    }

    std::error_code EC;
    std::string SCFile =
        makeResultFileName(".prop", I, ImgPSInfo.IsEsimdKernel ? "esimd_" : "");
    raw_fd_ostream SCOut(SCFile, EC);
    PropSet.write(SCOut);
    Res.emplace_back(std::move(SCFile));
  }

  return Res;
}

// Saves specified collection of symbols lists to files.
// Saves file list if user specified corresponding filename.
static string_vector saveResultSymbolsLists(string_vector &ResSymbolsLists,
                                            StringRef Suffix) {
  string_vector Res;

  std::string TxtFilesList;
  for (size_t I = 0; I < ResSymbolsLists.size(); ++I) {
    std::string CurOutFileName = makeResultFileName(".sym", I, Suffix);
    writeToFile(CurOutFileName, ResSymbolsLists[I]);
    Res.emplace_back(std::move(CurOutFileName));
  }
  return Res;
}

#define CHECK_AND_EXIT(E)                                                      \
  {                                                                            \
    Error LocE = std::move(E);                                                 \
    if (LocE) {                                                                \
      logAllUnhandledErrors(std::move(LocE), WithColor::error(errs()));        \
      return 1;                                                                \
    }                                                                          \
  }

// When ESIMD code was separated from the regular SYCL code,
// we can safely process ESIMD part.
// TODO: support options like -debug-pass, -print-[before|after], and others
static void LowerEsimdConstructs(Module &M) {
  legacy::PassManager MPM;
  MPM.add(createSYCLLowerESIMDPass());
  if (!OptLevelO0) {
    // Force-inline all functions marked 'alwaysinline' by the LowerESIMD pass.
    MPM.add(createAlwaysInlinerLegacyPass());
    MPM.add(createSROAPass());
  }
  MPM.add(createESIMDLowerVecArgPass());
  MPM.add(createESIMDLowerLoadStorePass());
  if (!OptLevelO0) {
    MPM.add(createSROAPass());
    MPM.add(createEarlyCSEPass(true));
    MPM.add(createInstructionCombiningPass());
    MPM.add(createDeadCodeEliminationPass());
    // TODO: maybe remove some passes below that don't affect code quality
    MPM.add(createSROAPass());
    MPM.add(createEarlyCSEPass(true));
    MPM.add(createInstructionCombiningPass());
    MPM.add(createDeadCodeEliminationPass());
  }
  MPM.add(createGenXSPIRVWriterAdaptorPass(/*RewriteTypes=*/true));
  MPM.run(M);
}

using TableFiles = std::map<StringRef, string_vector>;

static TableFiles processOneModule(std::unique_ptr<Module> M, bool IsEsimd,
                                   bool SyclAndEsimdCode) {
  TableFiles TblFiles;
  if (!M)
    return TblFiles;

  if (IsEsimd && LowerEsimd)
    LowerEsimdConstructs(*M);

  std::map<StringRef, std::vector<Function *>> GlobalsSet;

  bool DoSplit = SplitMode.getNumOccurrences() > 0;
  bool DoSpecConst = SpecConstLower.getNumOccurrences() > 0;

  if (DoSplit || DoSymGen) {
    KernelMapEntryScope Scope = Scope_Global;
    if (DoSplit) {
      if (SplitMode == SPLIT_AUTO)
        Scope = selectDeviceCodeSplitScopeAutomatically(*M);
      else
        Scope =
            SplitMode == SPLIT_PER_KERNEL ? Scope_PerKernel : Scope_PerModule;
    }
    collectKernelModuleMap(*M, GlobalsSet, Scope);
  }

  std::vector<std::unique_ptr<Module>> ResultModules;
  string_vector ResultSymbolsLists;

  bool SpecConstsMet = false;
  bool SetSpecConstAtRT = DoSpecConst && (SpecConstLower == SC_USE_RT_VAL);

  if (DoSplit)
    splitModule(*M, GlobalsSet, ResultModules);
  // post-link always produces a code result, even if it is unmodified input
  if (ResultModules.empty())
    ResultModules.push_back(std::move(M));

  if (DoSpecConst) {
    ModulePassManager RunSpecConst;
    ModuleAnalysisManager MAM;
    SpecConstantsPass SCP(SetSpecConstAtRT);
    // Register required analysis
    MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
    RunSpecConst.addPass(std::move(SCP));

    for (auto &MPtr : ResultModules) {
      // perform the spec constant intrinsics transformation on each resulting
      // module
      PreservedAnalyses Res = RunSpecConst.run(*MPtr, MAM);
      SpecConstsMet |= !Res.areAllPreserved();
    }
  }

  if (IROutputOnly) {
    // the result is the transformed input LLVMIR file rather than a file table
    saveModule(*ResultModules.front(), OutputFilename);
    return TblFiles;
  }

  {
    // Reuse input module with only regular SYCL kernels if there were
    // no spec constants and no splitting.
    // We cannot reuse input module for ESIMD code since it was transformed.
    bool CanReuseInputModule = !SpecConstsMet && (ResultModules.size() == 1) &&
                               !SyclAndEsimdCode && !IsEsimd;
    string_vector Files =
        CanReuseInputModule
            ? string_vector{InputFilename}
            : saveResultModules(ResultModules, IsEsimd ? "esimd_" : "");

    // "Code" column is always output
    std::copy(Files.begin(), Files.end(),
              std::back_inserter(TblFiles[COL_CODE]));
  }

  {
    ImagePropSaveInfo ImgPSInfo = {
        true,          DoSpecConst,         SetSpecConstAtRT,
        SpecConstsMet, EmitKernelParamInfo, EmitProgramMetadata,
        IsEsimd};
    string_vector Files = saveDeviceImageProperty(ResultModules, ImgPSInfo);
    std::copy(Files.begin(), Files.end(),
              std::back_inserter(TblFiles[COL_PROPS]));
  }
  if (DoSymGen) {
    // extract symbols per each module
    collectSymbolsLists(GlobalsSet, ResultSymbolsLists);
    if (ResultSymbolsLists.empty()) {
      // push empty symbols list for consistency
      assert(ResultModules.size() == 1);
      ResultSymbolsLists.push_back("");
    }
    string_vector Files =
        saveResultSymbolsLists(ResultSymbolsLists, IsEsimd ? "esimd_" : "");
    std::copy(Files.begin(), Files.end(),
              std::back_inserter(TblFiles[COL_SYM]));
  }
  return TblFiles;
}

using ModulePair = std::pair<std::unique_ptr<Module>, std::unique_ptr<Module>>;

// This function splits a module with a mix of SYCL and ESIMD kernels
// into two separate modules.
static ModulePair splitSyclEsimd(std::unique_ptr<Module> M) {
  std::vector<Function *> SyclFunctions;
  std::vector<Function *> EsimdFunctions;
  // Collect information about the SYCL and ESIMD functions in the module.
  // Process module entry points: kernels and SYCL_EXTERNAL functions.
  // Only they have sycl-module-id attribute, so any other unrefenced
  // functions are dropped. SPIRV and SYCL builtin functions are not
  // considered as module entry points.
  for (auto &F : M->functions()) {
    if (F.hasFnAttribute(ATTR_SYCL_MODULE_ID) &&
        !funcIsSpirvSyclBuiltin(F.getName())) {
      if (F.getMetadata("sycl_explicit_simd"))
        EsimdFunctions.push_back(&F);
      else
        SyclFunctions.push_back(&F);
    }
  }

  // If only SYCL kernels or only ESIMD kernels, no splitting needed.
  if (EsimdFunctions.empty())
    return std::make_pair(std::move(M), std::unique_ptr<Module>(nullptr));

  if (SyclFunctions.empty())
    return std::make_pair(std::unique_ptr<Module>(nullptr), std::move(M));

  // Key values in KernelModuleMap are not significant, but they define the
  // order, in which kernels are processed in the splitModule function. The
  // caller of the splitSyclEsimd function expects a pair of 1-Sycl and 2-Esimd
  // modules, hence the strings names below.
  std::map<StringRef, std::vector<Function *>> KernelModuleMap(
      {{"1-SYCL", SyclFunctions}, {"2-ESIMD", EsimdFunctions}});
  std::vector<std::unique_ptr<Module>> ResultModules;
  splitModule(*M, KernelModuleMap, ResultModules);
  assert(ResultModules.size() == 2);
  return std::make_pair(std::move(ResultModules[0]),
                        std::move(ResultModules[1]));
}

static TableFiles processInputModule(std::unique_ptr<Module> M) {
  if (!SplitEsimd)
    return processOneModule(std::move(M), false, false);

  std::unique_ptr<Module> SyclModule;
  std::unique_ptr<Module> EsimdModule;
  std::tie(SyclModule, EsimdModule) = splitSyclEsimd(std::move(M));

  // Do we have both Sycl and Esimd code?
  bool SyclAndEsimdCode = SyclModule && EsimdModule;

  TableFiles SyclTblFiles =
      processOneModule(std::move(SyclModule), false, SyclAndEsimdCode);
  TableFiles EsimdTblFiles =
      processOneModule(std::move(EsimdModule), true, SyclAndEsimdCode);

  // Merge the two resulting file maps
  TableFiles MergedTblFiles;
  for (auto &ColumnStr : {COL_CODE, COL_PROPS, COL_SYM}) {
    auto &SyclFiles = SyclTblFiles[ColumnStr];
    auto &EsimdFiles = EsimdTblFiles[ColumnStr];
    auto &MergedFiles = MergedTblFiles[ColumnStr];
    std::copy(SyclFiles.begin(), SyclFiles.end(),
              std::back_inserter(MergedFiles));
    std::copy(EsimdFiles.begin(), EsimdFiles.end(),
              std::back_inserter(MergedFiles));
  }
  return MergedTblFiles;
}

int main(int argc, char **argv) {
  InitLLVM X{argc, argv};

  LLVMContext Context;
  cl::HideUnrelatedOptions(PostLinkCat);
  cl::ParseCommandLineOptions(
      argc, argv,
      "SYCL post-link device code processing tool.\n"
      "This is a collection of utilities run on device code's LLVM IR before\n"
      "handing off to back-end for further compilation or emitting SPIRV.\n"
      "The utilities are:\n"
      "- SYCL and ESIMD kernels can be split into separate modules with\n"
      "  '-split-esimd' option. The option has no effect when there is only\n"
      "  one type of kernels in the input module. Functions unreachable from\n"
      "  any kernel or SYCL_EXTERNAL function are dropped from the resulting\n"
      "  module(s)."
      "- Module splitter to split a big input module into smaller ones.\n"
      "  Groups kernels using function attribute 'sycl-module-id', i.e.\n"
      "  kernels with the same values of the 'sycl-module-id' attribute will\n"
      "  be put into the same module. If -split=kernel option is specified,\n"
      "  one module per kernel will be emitted.\n"
      "  '-split=auto' mode automatically selects the best way of splitting\n"
      "  kernels into modules based on some heuristic.\n"
      "  The '-split' option is compatible with '-split-esimd'. In this case,\n"
      "  first input module will be split into SYCL and ESIMD modules. Then\n"
      "  both modules will be further split according to the '-split' option.\n"
      "- If -symbols options is also specified, then for each produced module\n"
      "  a text file containing names of all spir kernels in it is generated.\n"
      "- Specialization constant intrinsic transformer. Replaces symbolic\n"
      "  ID-based intrinsics to integer ID-based ones to make them friendly\n"
      "  for the SPIRV translator\n"
      "When the tool splits input module into regular SYCL and ESIMD kernels,\n"
      "it performs a set of specific lowering and transformation passes on\n"
      "ESIMD module, which is enabled by the '-lower-esimd' option. Regular\n"
      "optimization level options are supported, e.g. -O[0|1|2|3|s|z].\n"
      "Normally, the tool generates a number of files and \"file table\"\n"
      "file listing all generated files in a table manner. For example, if\n"
      "the input file 'example.bc' contains two kernels, then the command\n"
      "  $ sycl-post-link --split=kernel --symbols --spec-const=rt \\\n"
      "    -o example.table example.bc\n"
      "will produce 'example.table' file with the following content:\n"
      "  [Code|Properties|Symbols]\n"
      "  example_0.bc|example_0.prop|example_0.sym\n"
      "  example_1.bc|example_1.prop|example_1.sym\n"
      "When only specialization constant processing is needed, the tool can\n"
      "output a single transformed IR file if --ir-output-only is specified:\n"
      "  $ sycl-post-link --ir-output-only --spec-const=default \\\n"
      "    -o example_p.bc example.bc\n"
      "will produce single output file example_p.bc suitable for SPIRV\n"
      "translation.\n"
      "--ir-output-only option is not not compatible with split modes other\n"
      "than 'auto'.\n");

  bool DoSplit = SplitMode.getNumOccurrences() > 0;
  bool DoSplitEsimd = SplitEsimd.getNumOccurrences() > 0;
  bool DoSpecConst = SpecConstLower.getNumOccurrences() > 0;
  bool DoParamInfo = EmitKernelParamInfo.getNumOccurrences() > 0;
  bool DoProgMetadata = EmitProgramMetadata.getNumOccurrences() > 0;

  if (!DoSplit && !DoSpecConst && !DoSymGen && !DoParamInfo &&
      !DoProgMetadata && !DoSplitEsimd) {
    errs() << "no actions specified; try --help for usage info\n";
    return 1;
  }
  if (IROutputOnly && (DoSplit && SplitMode != SPLIT_AUTO)) {
    errs() << "error: -" << SplitMode.ArgStr << "=" << SplitMode.ValueStr
           << " can't be used with -" << IROutputOnly.ArgStr << "\n";
    return 1;
  }
  if (IROutputOnly && DoSplitEsimd) {
    errs() << "error: -" << SplitEsimd.ArgStr << " can't be used with -"
           << IROutputOnly.ArgStr << "\n";
    return 1;
  }
  if (IROutputOnly && DoSymGen) {
    errs() << "error: -" << DoSymGen.ArgStr << " can't be used with -"
           << IROutputOnly.ArgStr << "\n";
    return 1;
  }
  if (IROutputOnly && DoParamInfo) {
    errs() << "error: -" << EmitKernelParamInfo.ArgStr << " can't be used with"
           << " -" << IROutputOnly.ArgStr << "\n";
    return 1;
  }
  if (IROutputOnly && DoProgMetadata) {
    errs() << "error: -" << EmitProgramMetadata.ArgStr << " can't be used with"
           << " -" << IROutputOnly.ArgStr << "\n";
    return 1;
  }
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseIRFile(InputFilename, Err, Context);
  // It is OK to use raw pointer here as we control that it does not outlive M
  // or objects it is moved to
  Module *MPtr = M.get();

  if (!MPtr) {
    Err.print(argv[0], errs());
    return 1;
  }

  // After linking device bitcode "llvm.used" holds references to the kernels
  // that are defined in the device image. But after splitting device image into
  // separate kernels we may end up with having references to kernel declaration
  // originating from "llvm.used" in the IR that is passed to llvm-spirv tool,
  // and these declarations cause an assertion in llvm-spirv. To workaround this
  // issue remove "llvm.used" from the input module before performing any other
  // actions.
  if (GlobalVariable *GV = MPtr->getGlobalVariable("llvm.used")) {
    assert(GV->user_empty() && "unexpected llvm.used users");
    GV->eraseFromParent();
  }

  if (OutputFilename.getNumOccurrences() == 0)
    OutputFilename = (Twine(sys::path::stem(InputFilename)) + ".files").str();

  TableFiles TblFiles = processInputModule(std::move(M));

  // Input module was processed and a single output file was requested.
  if (IROutputOnly)
    return 0;

  // Populate and emit the resulting table
  util::SimpleTable Table;
  for (auto &ColumnStr : {COL_CODE, COL_PROPS, COL_SYM})
    if (!TblFiles[ColumnStr].empty())
      CHECK_AND_EXIT(Table.addColumn(ColumnStr, TblFiles[ColumnStr]));

  std::error_code EC;
  raw_fd_ostream Out{OutputFilename, EC, sys::fs::OF_None};
  checkError(EC, "error opening file '" + OutputFilename + "'");
  Table.write(Out);

  return 0;
}
