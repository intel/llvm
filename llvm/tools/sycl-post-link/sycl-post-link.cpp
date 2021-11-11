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

#include "SYCLDeviceLibReqMask.h"
#include "SYCLKernelParamOptInfo.h"
#include "SpecConstants.h"

#include "llvm/ADT/SetVector.h"
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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PropertySetIO.h"
#include "llvm/Support/SimpleTable.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

using namespace llvm;

using string_vector = std::vector<std::string>;
using FuncPtrVector = std::vector<const Function *>;
using ModuleUPtr = std::unique_ptr<Module>;
using PropSetRegTy = llvm::util::PropertySetRegistry;

namespace {

cl::OptionCategory PostLinkCat{"sycl-post-link options"};

// Column names in the output file table. Must match across tools -
// clang/lib/Driver/Driver.cpp, sycl-post-link.cpp, ClangOffloadWrapper.cpp
constexpr char COL_CODE[] = "Code";
constexpr char COL_SYM[] = "Symbols";
constexpr char COL_PROPS[] = "Properties";
constexpr char ATTR_SYCL_MODULE_ID[] = "sycl-module-id";

// Identifying name for global scope
constexpr char GLOBAL_SCOPE_NAME[] = "<GLOBAL>";

// InputFilename - The filename to read from.
cl::opt<std::string> InputFilename{cl::Positional,
                                   cl::desc("<input bitcode file>"),
                                   cl::init("-"), cl::value_desc("filename")};

cl::opt<std::string> OutputDir{
    "out-dir",
    cl::desc(
        "Directory where files listed in the result file table will be output"),
    cl::value_desc("dirname"), cl::cat(PostLinkCat)};

cl::opt<std::string> OutputFilename{"o", cl::desc("Output filename"),
                                    cl::value_desc("filename"), cl::init("-"),
                                    cl::cat(PostLinkCat)};

cl::opt<bool> Force{"f", cl::desc("Enable binary output on terminals"),
                    cl::cat(PostLinkCat)};

cl::opt<bool> IROutputOnly{"ir-output-only", cl::desc("Output single IR file"),
                           cl::cat(PostLinkCat)};

cl::opt<bool> OutputAssembly{"S", cl::desc("Write output as LLVM assembly"),
                             cl::Hidden, cl::cat(PostLinkCat)};

cl::opt<bool> SplitEsimd{"split-esimd",
                         cl::desc("Split SYCL and ESIMD entry points"),
                         cl::cat(PostLinkCat)};

// TODO Design note: sycl-post-link should probably separate different kinds of
// its functionality on  logical and source level:
//  - LLVM IR module splitting
//  - Running LLVM IR passes on resulting modules
//  - Generating additional files (like spec constants, dead arg info,...)
// The tool itself could be just a "driver" creating needed pipelines from the
// above actions. This could help make the tool structure clearer and more
// maintainable.

cl::opt<bool> LowerEsimd{"lower-esimd", cl::desc("Lower ESIMD constructs"),
                         cl::cat(PostLinkCat)};

cl::opt<bool> OptLevelO0("O0",
                         cl::desc("Optimization level 0. Similar to clang -O0"),
                         cl::cat(PostLinkCat));

cl::opt<bool> OptLevelO1("O1",
                         cl::desc("Optimization level 1. Similar to clang -O1"),
                         cl::cat(PostLinkCat));

cl::opt<bool> OptLevelO2("O2",
                         cl::desc("Optimization level 2. Similar to clang -O2"),
                         cl::cat(PostLinkCat));

cl::opt<bool> OptLevelOs(
    "Os",
    cl::desc(
        "Like -O2 with extra optimizations for size. Similar to clang -Os"),
    cl::cat(PostLinkCat));

cl::opt<bool> OptLevelOz(
    "Oz",
    cl::desc("Like -Os but reduces code size further. Similar to clang -Oz"),
    cl::cat(PostLinkCat));

cl::opt<bool> OptLevelO3("O3",
                         cl::desc("Optimization level 3. Similar to clang -O3"),
                         cl::cat(PostLinkCat));

enum IRSplitMode {
  SPLIT_PER_TU,     // one module per translation unit
  SPLIT_PER_KERNEL, // one module per kernel
  SPLIT_AUTO        // automatically select split mode
};

cl::opt<IRSplitMode> SplitMode(
    "split", cl::desc("split input module"), cl::Optional, cl::init(SPLIT_AUTO),
    cl::values(
        clEnumValN(SPLIT_PER_TU, "source",
                   "1 output module per source (translation unit)"),
        clEnumValN(SPLIT_PER_KERNEL, "kernel", "1 output module per kernel"),
        clEnumValN(SPLIT_AUTO, "auto", "Choose split mode automatically")),
    cl::cat(PostLinkCat));

cl::opt<bool> DoSymGen{"symbols", cl::desc("generate exported symbol files"),
                       cl::cat(PostLinkCat)};

enum SpecConstMode { SC_USE_RT_VAL, SC_USE_DEFAULT_VAL };

cl::opt<SpecConstMode> SpecConstLower{
    "spec-const",
    cl::desc("lower and generate specialization constants information"),
    cl::Optional,
    cl::init(SC_USE_RT_VAL),
    cl::values(
        clEnumValN(SC_USE_RT_VAL, "rt", "spec constants are set at runtime"),
        clEnumValN(SC_USE_DEFAULT_VAL, "default",
                   "set spec constants to C++ defaults")),
    cl::cat(PostLinkCat)};

cl::opt<bool> EmitKernelParamInfo{
    "emit-param-info", cl::desc("emit kernel parameter optimization info"),
    cl::cat(PostLinkCat)};

cl::opt<bool> EmitProgramMetadata{"emit-program-metadata",
                                  cl::desc("emit SYCL program metadata"),
                                  cl::cat(PostLinkCat)};

cl::opt<bool> EmitExportedSymbols{"emit-exported-symbols",
                                  cl::desc("emit exported symbols"),
                                  cl::cat(PostLinkCat)};

cl::opt<bool> EmitOnlyKernelsAsEntryPoints{
    "emit-only-kernels-as-entry-points",
    cl::desc("Consider only sycl_kernel functions as entry points for "
             "device code split"),
    cl::cat(PostLinkCat), cl::init(false)};

// The option turns on using one ValueToValueMapTy instance for all split
// modules. Sharing may reduce RAM memory utilization significantly, but in some
// cases necessary data may be lost.
// For example, metadata that collects all kernels in a split module may be lost
// for CUDA target.
// This option is off by default.
cl::opt<bool> ReduceMemoryUsage{
    "reduce-memory-usage",
    cl::desc("Share temporary auxiliary memory to reduce RAM utilization"),
    cl::cat(PostLinkCat), cl::init(false)};

struct ImagePropSaveInfo {
  bool SetSpecConstAtRT;
  bool SpecConstsMet;
  bool EmitKernelParamInfo;
  bool EmitProgramMetadata;
  bool EmitExportedSymbols;
  bool IsEsimdKernel;
};

void error(const Twine &Msg) {
  errs() << "sycl-post-link: " << Msg << '\n';
  exit(1);
}

void checkError(std::error_code EC, const Twine &Prefix) {
  if (EC)
    error(Prefix + ": " + EC.message());
}

void writeToFile(const std::string &Filename, const std::string &Content) {
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

bool hasIndirectFunctionCalls(const Module &M) {
  for (const auto &F : M.functions()) {
    // There are functions marked with [[intel::device_indirectly_callable]]
    // attribute, because it instructs us to make this function available to the
    // whole program as it was compiled as a single module.
    if (F.hasFnAttribute("referenced-indirectly"))
      return true;
    if (F.isDeclaration())
      continue;
    // There are indirect calls in the module, which means that we don't know
    // how to group functions so both caller and callee of indirect call are in
    // the same module.
    for (const auto &I : instructions(F)) {
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (!CI->getCalledFunction())
          return true;
    }

    // Function pointer is used somewhere. Follow the same rule as above.
    for (const auto *U : F.users())
      if (!isa<CallInst>(U))
        return true;
  }

  return false;
}

KernelMapEntryScope selectDeviceCodeSplitScope(const Module &M) {
  bool DoSplit = SplitMode.getNumOccurrences() > 0;
  if (DoSplit) {
    switch (SplitMode) {
    case SPLIT_PER_TU:
      return Scope_PerModule;

    case SPLIT_PER_KERNEL:
      return Scope_PerKernel;

    case SPLIT_AUTO: {
      if (IROutputOnly) {
        // We allow enabling auto split mode even in presence of -ir-output-only
        // flag, but in this case we are limited by it so we can't do any split
        // at all.
        return Scope_Global;
      }

      if (hasIndirectFunctionCalls(M))
        return Scope_Global;

      // At the moment, we assume that per-source split is the best way of
      // splitting device code and can always be used except for cases handled
      // above.
      return Scope_PerModule;
    }
    }
  }
  return Scope_Global;
}

// Return true if the function is a SPIRV or SYCL builtin, e.g.
// _Z28__spirv_GlobalInvocationId_xv
bool isSpirvSyclBuiltin(StringRef FName) {
  if (!FName.consume_front("_Z"))
    return false;
  // now skip the digits
  FName = FName.drop_while([](char C) { return std::isdigit(C); });

  return FName.startswith("__spirv_") || FName.startswith("__sycl_");
}

bool isEntryPoint(const Function &F) {
  // Kernels are always considered to be entry points
  if (CallingConv::SPIR_KERNEL == F.getCallingConv())
    return true;

  if (!EmitOnlyKernelsAsEntryPoints) {
    // If not disabled, SYCL_EXTERNAL functions with sycl-module-id attribute
    // are also considered as entry points (except __spirv_* and __sycl_*
    // functions)
    return F.hasFnAttribute(ATTR_SYCL_MODULE_ID) &&
           !isSpirvSyclBuiltin(F.getName());
  }

  return false;
}

// This function decides how entry points of the input module M will be
// distributed ("split") into multiple modules based on the command options and
// IR attributes. The decision is recorded in the output map parameter
// ResKernelModuleMap which maps some key to a group of entry points. Each such
// group along with IR it depends on (globals, functions from its call graph,
// ...) will constitute a separate module.
void collectEntryPointToModuleMap(
    const Module &M, std::map<StringRef, FuncPtrVector> &ResKernelModuleMap,
    KernelMapEntryScope EntryScope) {

  // Only process module entry points:
  for (const auto &F : M.functions()) {
    if (!isEntryPoint(F))
      continue;

    switch (EntryScope) {
    case Scope_PerKernel:
      ResKernelModuleMap[F.getName()].push_back(&F);
      break;
    case Scope_PerModule: {
      if (!F.hasFnAttribute(ATTR_SYCL_MODULE_ID))
        // TODO It may make sense to group all entry points w/o the attribute
        // into a separate module rather than issuing an error. Should probably
        // be controlled by an option.
        error("no '" + Twine(ATTR_SYCL_MODULE_ID) +
              "' attribute for entry point '" + F.getName() +
              "', per-module split not possible");

      Attribute Id = F.getFnAttribute(ATTR_SYCL_MODULE_ID);
      StringRef Val = Id.getValueAsString();
      ResKernelModuleMap[Val].push_back(&F);
      break;
    }
    case Scope_Global:
      // the map key is not significant here
      ResKernelModuleMap[GLOBAL_SCOPE_NAME].push_back(&F);
      break;
    }
  }
}

enum HasAssertStatus { No_Assert, Assert, Assert_Indirect };

// Go through function call graph searching for assert call.
HasAssertStatus hasAssertInFunctionCallGraph(const Function *Func) {
  // Map holds the info about assertions in already examined functions:
  // true  - if there is an assertion in underlying functions,
  // false - if there are definetely no assertions in underlying functions.
  static std::map<const Function *, bool> hasAssertionInCallGraphMap;
  FuncPtrVector FuncCallStack;

  static FuncPtrVector isIndirectlyCalledInGraph;

  FuncPtrVector Workstack;
  Workstack.push_back(Func);

  while (!Workstack.empty()) {
    const Function *F = Workstack.back();
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
    for (const auto &I : instructions(F)) {
      if (!isa<CallBase>(&I))
        continue;

      const Function *CF = cast<CallBase>(&I)->getCalledFunction();
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
        for (const auto *It : FuncCallStack)
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

std::vector<StringRef> getKernelNamesUsingAssert(const Module &M) {
  std::vector<StringRef> Result;

  bool HasIndirectlyCalledAssert = false;
  FuncPtrVector Kernels;
  for (const auto &F : M.functions()) {
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
      Result.push_back(F.getName());
      break;
    case Assert_Indirect:
      HasIndirectlyCalledAssert = true;
      break;
    case No_Assert:
      break;
    }
  }

  if (HasIndirectlyCalledAssert)
    for (const auto *F : Kernels)
      Result.push_back(F->getName());

  return Result;
}

// Gets reqd_work_group_size information for function Func.
std::vector<uint32_t> getKernelReqdWorkGroupSizeMetadata(const Function &Func) {
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

// Input parameter ModuleEntryPoints contains group of entry points with same
// values of the sycl-module-id attribute. ResSymbolsStr is a string with entry
// point names from the same module separated by \n.
std::string collectSymbolsList(const FuncPtrVector *ModuleEntryPoints) {
  std::string ResSymbolsStr;
  // ModuleEntryPoints is nullptr if there are no entry points in input module.
  if (ModuleEntryPoints)
    for (const auto *F : *ModuleEntryPoints) {
      ResSymbolsStr =
          (Twine(ResSymbolsStr) + Twine(F->getName()) + Twine("\n")).str();
    }
  return ResSymbolsStr;
}

// During cloning of Module ValueMap tries to reuse already created old to new
// Value mappings and asserts if a new Value pointer is nullptr. It may happen
// after one cloned module is destroyed and we try to reuse ValueToValueMapTy
// object for the next module to clone.
// Proper solution is to improve CloneModule to support multiple clonings from
// one source Module.
// Current workaround is to drop all records that contain nullptr values from
// ValueToValueMapTy after cloned module is destroyed.
void cleanupVMap(ValueToValueMapTy &VMap) {
  for (auto It = VMap.begin(), End = VMap.end(); It != End;) {
    if (It->second)
      ++It;
    else
      VMap.erase(It++);
  }
}

// The function produces a copy of input LLVM IR module M with only those entry
// points that are specified in ModuleEntryPoints vector.
ModuleUPtr splitModule(const Module &M, ValueToValueMapTy &VMap,
                       const FuncPtrVector &ModuleEntryPoints) {
  cleanupVMap(VMap);

  // For each group of entry points collect all dependencies.
  SetVector<const GlobalValue *> GVs;
  FuncPtrVector Workqueue;

  for (const auto &F : ModuleEntryPoints) {
    GVs.insert(F);
    Workqueue.push_back(F);
  }

  while (!Workqueue.empty()) {
    const Function *F = &*Workqueue.back();
    Workqueue.pop_back();
    for (const auto &I : instructions(F)) {
      if (const CallBase *CB = dyn_cast<CallBase>(&I))
        if (const Function *CF = CB->getCalledFunction())
          if (!CF->isDeclaration() && !GVs.count(CF)) {
            GVs.insert(CF);
            Workqueue.push_back(CF);
          }
    }
  }

  // It's not easy to trace global variable's uses inside needed functions
  // because global variable can be used inside a combination of operators, so
  // mark all global variables as needed and remove dead ones after cloning.
  for (const auto &G : M.globals()) {
    GVs.insert(&G);
  }

  ModuleUPtr MClone{nullptr};
  {
    // If '-reduce-memory-usage=true' is set then use shared ValueToValueMapTy.
    // Otherwise, old behaviour of using local value map for splitting each
    // module is utilized.
    ValueToValueMapTy TempVMap;
    ValueToValueMapTy &VMapRef = !ReduceMemoryUsage ? TempVMap : VMap;
    // Clone definitions only for needed globals. Others will be added as
    // declarations and removed later.
    MClone = CloneModule(M, VMapRef,
                         [&](const GlobalValue *GV) { return GVs.count(GV); });
  }

  // TODO: Use the new PassManager instead?
  legacy::PassManager Passes;
  // Do cleanup.
  Passes.add(createGlobalDCEPass());           // Delete unreachable globals.
  Passes.add(createStripDeadDebugInfoPass());  // Remove dead debug info.
  Passes.add(createStripDeadPrototypesPass()); // Remove dead func decls.
  Passes.run(*MClone.get());

  return MClone;
}

std::string makeResultFileName(Twine Ext, int I, StringRef Suffix) {
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

void saveModule(Module &M, StringRef OutFilename) {
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

void saveDeviceImageProperty(Module &M, const FuncPtrVector *ModuleEntryPoints,
                             const ImagePropSaveInfo &ImgPSInfo,
                             const std::string &PropSetFile) {
  PropSetRegTy PropSet;

  {
    legacy::PassManager GetSYCLDeviceLibReqMask;
    auto *SDLReqMaskLegacyPass = new SYCLDeviceLibReqMaskPass();
    GetSYCLDeviceLibReqMask.add(SDLReqMaskLegacyPass);
    GetSYCLDeviceLibReqMask.run(M);
    uint32_t MRMask = SDLReqMaskLegacyPass->getSYCLDeviceLibReqMask();
    std::map<StringRef, uint32_t> RMEntry = {{"DeviceLibReqMask", MRMask}};
    PropSet.add(PropSetRegTy::SYCL_DEVICELIB_REQ_MASK, RMEntry);
  }

  if (ImgPSInfo.SpecConstsMet) {
    // extract spec constant maps per each module
    SpecIDMapTy TmpSpecIDMap;
    SpecConstantsPass::collectSpecConstantMetadata(M, TmpSpecIDMap);
    PropSet.add(PropSetRegTy::SYCL_SPECIALIZATION_CONSTANTS, TmpSpecIDMap);

    // Add property with the default values of spec constants
    std::vector<char> DefaultValues;
    SpecConstantsPass::collectSpecConstantDefaultValuesMetadata(
        M, DefaultValues);
    PropSet.add(PropSetRegTy::SYCL_SPEC_CONSTANTS_DEFAULT_VALUES, "all",
                DefaultValues);
  }

  if (ImgPSInfo.EmitKernelParamInfo) {
    // extract kernel parameter optimization info per module
    ModuleAnalysisManager MAM;
    // Register required analysis
    MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
    // Register the payload analysis
    MAM.registerPass([&] { return SYCLKernelParamOptInfoAnalysis(); });
    SYCLKernelParamOptInfo PInfo =
        MAM.getResult<SYCLKernelParamOptInfoAnalysis>(M);

    // convert analysis results into properties and record them
    llvm::util::PropertySet &Props =
        PropSet[PropSetRegTy::SYCL_KERNEL_PARAM_OPT_INFO];

    for (const auto &NameInfoPair : PInfo) {
      const llvm::BitVector &Bits = NameInfoPair.second;
      if (Bits.empty())
        continue; // Nothing to add

      const llvm::ArrayRef<uintptr_t> Arr = Bits.getData();
      const unsigned char *Data =
          reinterpret_cast<const unsigned char *>(Arr.begin());
      llvm::util::PropertyValue::SizeTy DataBitSize = Bits.size();
      Props.insert(std::make_pair(
          NameInfoPair.first, llvm::util::PropertyValue(Data, DataBitSize)));
    }
  }

  if (ImgPSInfo.EmitExportedSymbols && ModuleEntryPoints) {
    // Extract the exported functions for a result module
    for (const auto *F : *ModuleEntryPoints)
      if (F->getCallingConv() == CallingConv::SPIR_FUNC)
        PropSet[PropSetRegTy::SYCL_EXPORTED_SYMBOLS].insert(
            {F->getName(), true});
  }

  // Metadata names may be composite so we keep them alive until the
  // properties have been written.
  SmallVector<std::string, 4> MetadataNames;
  if (ImgPSInfo.EmitProgramMetadata) {
    auto &ProgramMetadata = PropSet[PropSetRegTy::SYCL_PROGRAM_METADATA];

    // Add reqd_work_group_size information to program metadata
    for (const Function &Func : M.functions()) {
      std::vector<uint32_t> KernelReqdWorkGroupSize =
          getKernelReqdWorkGroupSizeMetadata(Func);
      if (KernelReqdWorkGroupSize.empty())
        continue;
      MetadataNames.push_back(Func.getName().str() + "@reqd_work_group_size");
      ProgramMetadata.insert({MetadataNames.back(), KernelReqdWorkGroupSize});
    }
  }

  if (ImgPSInfo.IsEsimdKernel)
    PropSet[PropSetRegTy::SYCL_MISC_PROP].insert({"isEsimdImage", true});

  {
    std::vector<StringRef> FuncNames = getKernelNamesUsingAssert(M);
    for (const StringRef &FName : FuncNames)
      PropSet[PropSetRegTy::SYCL_ASSERT_USED].insert({FName, true});
  }

  std::error_code EC;
  raw_fd_ostream SCOut(PropSetFile, EC);
  checkError(EC, "error opening file '" + PropSetFile + "'");
  PropSet.write(SCOut);
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
void LowerEsimdConstructs(Module &M) {
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

bool transformSpecConstants(Module &M) {
  bool DoSpecConst = SpecConstLower.getNumOccurrences() > 0;
  if (!DoSpecConst)
    return false;

  bool SetSpecConstAtRT = DoSpecConst && (SpecConstLower == SC_USE_RT_VAL);
  ModulePassManager RunSpecConst;
  ModuleAnalysisManager MAM;
  SpecConstantsPass SCP(SetSpecConstAtRT);
  // Register required analysis
  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  RunSpecConst.addPass(std::move(SCP));

  // perform the spec constant intrinsics transformation on resulting module
  PreservedAnalyses Res = RunSpecConst.run(M, MAM);
  return !Res.areAllPreserved();
}

using TableFiles = std::map<StringRef, string_vector>;

TableFiles processOneModule(ModuleUPtr M, bool IsEsimd, bool SyclAndEsimdCode) {
  TableFiles TblFiles;
  if (!M)
    return TblFiles;

  // After linking device bitcode "llvm.used" holds references to the kernels
  // that are defined in the device image. But after splitting device image into
  // separate kernels we may end up with having references to kernel declaration
  // originating from "llvm.used" in the IR that is passed to llvm-spirv tool,
  // and these declarations cause an assertion in llvm-spirv. To workaround this
  // issue remove "llvm.used" from the input module before performing any other
  // actions.
  bool IsLLVMUsedRemoved = false;
  if (GlobalVariable *GV = M->getGlobalVariable("llvm.used")) {
    assert(GV->user_empty() && "unexpected llvm.used users");
    GV->eraseFromParent();
    IsLLVMUsedRemoved = true;
  }

  if (IsEsimd && LowerEsimd)
    LowerEsimdConstructs(*M);

  std::map<StringRef, FuncPtrVector> GlobalsSet;

  bool DoSplit = SplitMode.getNumOccurrences() > 0;

  if (DoSplit || DoSymGen) {
    KernelMapEntryScope Scope = selectDeviceCodeSplitScope(*M);
    collectEntryPointToModuleMap(*M, GlobalsSet, Scope);
  }

  bool DoSpecConst = SpecConstLower.getNumOccurrences() > 0;
  bool SetSpecConstAtRT = DoSpecConst && (SpecConstLower == SC_USE_RT_VAL);

  StringRef FileSuffix = IsEsimd ? "esimd_" : "";
  size_t I = 0;
  auto GlobSetIt = GlobalsSet.cbegin();

  // ValueToValueMapTy map should be shared between split Module objects during
  // llvm::CloneModule call. Otherwise, some Value* instances allocated for
  // split module may remain in memory after both split module and
  // ValueToValueMapTy are destroyed. In case if there a lot of split modules
  // and/or they utilize a lot of memory it may lead to memory overflow.
  ValueToValueMapTy SplitVMap;

  do {
    const FuncPtrVector *ResModuleGlobals{nullptr};
    if (GlobSetIt != GlobalsSet.cend())
      ResModuleGlobals = &(GlobSetIt->second);

    ModuleUPtr ResM{nullptr};

    if (DoSplit && ResModuleGlobals) {
      ResM = splitModule(*M, SplitVMap, *ResModuleGlobals);
    } else {
      // sycl-post-link always produces a code result, even if it doesn't modify
      // input.
      assert(GlobalsSet.size() <= 1);
      // NOTE: Do not forget that M is no more usable after that.
      ResM = std::move(M);
    }

    bool SpecConstsMet = transformSpecConstants(*ResM);

    if (IROutputOnly) {
      // the result is the transformed input LLVM IR file rather than a file
      // table
      saveModule(*ResM, OutputFilename);
      return TblFiles;
    }

    {
      // Reuse input module with only regular SYCL kernels if there were
      // no spec constants and no splitting.
      // We cannot reuse input module for ESIMD code since it was transformed.
      std::string ResModuleFile{};
      bool CanReuseInputModule = !SyclAndEsimdCode && !IsEsimd &&
                                 !IsLLVMUsedRemoved && !SpecConstsMet &&
                                 (!DoSplit || GlobalsSet.size() <= 1);
      if (CanReuseInputModule)
        ResModuleFile = InputFilename;
      else {
        ResModuleFile =
            makeResultFileName((OutputAssembly) ? ".ll" : ".bc", I, FileSuffix);
        saveModule(*ResM, ResModuleFile);
      }
      // "Code" column is always output
      TblFiles[COL_CODE].push_back(ResModuleFile);
    }

    {
      ImagePropSaveInfo ImgPSInfo = {SetSpecConstAtRT,    SpecConstsMet,
                                     EmitKernelParamInfo, EmitProgramMetadata,
                                     EmitExportedSymbols, IsEsimd};
      std::string PropSetFile = makeResultFileName(".prop", I, FileSuffix);
      saveDeviceImageProperty(*ResM, ResModuleGlobals, ImgPSInfo, PropSetFile);
      TblFiles[COL_PROPS].push_back(PropSetFile);
    }

    if (DoSymGen) {
      // extract symbols from module
      std::string ResultSymbolsList = collectSymbolsList(ResModuleGlobals);
      std::string ResultSymbolsFile = makeResultFileName(".sym", I, FileSuffix);
      writeToFile(ResultSymbolsFile, ResultSymbolsList);
      TblFiles[COL_SYM].push_back(ResultSymbolsFile);
    }

    ++I;
    if (GlobSetIt != GlobalsSet.cend())
      ++GlobSetIt;
  } while (GlobSetIt != GlobalsSet.cend());

  return TblFiles;
}

TableFiles processInputModule(ModuleUPtr M) {
  if (!SplitEsimd)
    return processOneModule(std::move(M), false, false);

  FuncPtrVector SyclFunctions;
  FuncPtrVector EsimdFunctions;
  // Collect information about the SYCL and ESIMD functions in the module.
  // Only process module entry points.
  for (const auto &F : M->functions()) {
    if (isEntryPoint(F)) {
      if (F.getMetadata("sycl_explicit_simd"))
        EsimdFunctions.push_back(&F);
      else
        SyclFunctions.push_back(&F);
    }
  }

  // Do we have both Sycl and Esimd code?
  bool SyclAndEsimdCode = !SyclFunctions.empty() && !EsimdFunctions.empty();
  ValueToValueMapTy SyclEsimdVMap;

  // If only SYCL kernels or only ESIMD kernels, no splitting needed.
  // Otherwise splitting a module with a mix of SYCL and ESIMD kernels into two
  // separate modules.
  // Warning: if one global ValueToValueMapTy object is used for splitting then
  // we can keep only one split module in memory at a time. Otherwise, if
  // several split modules are in memory then some of Values in that map may be
  // shared between those modules and then destruction of any module is crashed
  // because it has Values that are still used by other modules.
  ModuleUPtr SyclModule{nullptr};
  if (EsimdFunctions.empty())
    SyclModule = std::move(M);
  else if (!SyclFunctions.empty())
    SyclModule = splitModule(*M, SyclEsimdVMap, SyclFunctions);
  TableFiles SyclTblFiles =
      processOneModule(std::move(SyclModule), false, SyclAndEsimdCode);

  ModuleUPtr EsimdModule{nullptr};
  if (SyclFunctions.empty())
    EsimdModule = std::move(M);
  else if (!EsimdFunctions.empty())
    EsimdModule = splitModule(*M, SyclEsimdVMap, EsimdFunctions);
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

} // namespace

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
      "  any entry point (kernels and SYCL_EXTERNAL functions) are\n"
      "  dropped from the resulting module(s).\n"
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
  bool DoExportedSyms = EmitExportedSymbols.getNumOccurrences() > 0;

  if (!DoSplit && !DoSpecConst && !DoSymGen && !DoParamInfo &&
      !DoProgMetadata && !DoSplitEsimd && !DoExportedSyms) {
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
  if (IROutputOnly && DoExportedSyms) {
    errs() << "error: -" << EmitExportedSymbols.ArgStr << " can't be used with"
           << " -" << IROutputOnly.ArgStr << "\n";
    return 1;
  }
  SMDiagnostic Err;
  ModuleUPtr M = parseIRFile(InputFilename, Err, Context);
  // It is OK to use raw pointer here as we control that it does not outlive M
  // or objects it is moved to
  Module *MPtr = M.get();

  if (!MPtr) {
    Err.print(argv[0], errs());
    return 1;
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
