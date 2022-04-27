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

#include "CompileTimePropertiesPass.h"
#include "DeviceGlobals.h"
#include "ModuleSplitter.h"
#include "SYCLDeviceLibReqMask.h"
#include "SYCLKernelParamOptInfo.h"
#include "SpecConstants.h"
#include "Support.h"

#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/GenXIntrinsics/GenXSPIRVWriterAdaptor.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/SYCLLowerIR/ESIMD/LowerESIMD.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PropertySetIO.h"
#include "llvm/Support/SimpleTable.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/GlobalStatus.h"

#include <algorithm>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace llvm;

using string_vector = std::vector<std::string>;

namespace {

cl::OptionCategory PostLinkCat{"sycl-post-link options"};

// Column names in the output file table. Must match across tools -
// clang/lib/Driver/Driver.cpp, sycl-post-link.cpp, ClangOffloadWrapper.cpp
constexpr char COL_CODE[] = "Code";
constexpr char COL_SYM[] = "Symbols";
constexpr char COL_PROPS[] = "Properties";

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
// its functionality on logical and source level:
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

cl::opt<module_split::IRSplitMode> SplitMode(
    "split", cl::desc("split input module"), cl::Optional,
    cl::init(module_split::SPLIT_AUTO),
    cl::values(clEnumValN(module_split::SPLIT_PER_TU, "source",
                          "1 output module per source (translation unit)"),
               clEnumValN(module_split::SPLIT_PER_KERNEL, "kernel",
                          "1 output module per kernel"),
               clEnumValN(module_split::SPLIT_AUTO, "auto",
                          "Choose split mode automatically")),
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

cl::opt<bool> DeviceGlobals{
    "device-globals",
    cl::desc("Lower and generate information about device global variables"),
    cl::cat(PostLinkCat)};

struct GlobalBinImageProps {
  bool SpecConstsMet;
  bool EmitKernelParamInfo;
  bool EmitProgramMetadata;
  bool EmitExportedSymbols;
  bool IsEsimdKernel;
  bool EmitDeviceGlobalPropSet;
};

void writeToFile(const std::string &Filename, const std::string &Content) {
  std::error_code EC;
  raw_fd_ostream OS{Filename, EC, sys::fs::OpenFlags::OF_None};
  checkError(EC, "error opening the file '" + Filename + "'");
  OS.write(Content.data(), Content.size());
  OS.close();
}

// This function traverses over reversed call graph by BFS algorithm.
// It means that an edge links some function @func with functions
// which contain call of function @func. It starts from
// @StartingFunction and lifts up until it reach all reachable functions,
// or it reaches some function containing "referenced-indirectly" attribute.
// If it reaches "referenced-indirectly" attribute than it returns an empty
// Optional.
// Otherwise, it returns an Optional containing a list of reached
// SPIR kernel function's names.
Optional<std::vector<StringRef>>
traverseCGToFindSPIRKernels(const Function *StartingFunction) {
  std::queue<const Function *> FunctionsToVisit;
  std::unordered_set<const Function *> VisitedFunctions;
  FunctionsToVisit.push(StartingFunction);
  std::vector<StringRef> KernelNames;

  while (!FunctionsToVisit.empty()) {
    const Function *F = FunctionsToVisit.front();
    FunctionsToVisit.pop();

    auto InsertionResult = VisitedFunctions.insert(F);
    // It is possible that we insert some particular function several
    // times in functionsToVisit queue.
    if (!InsertionResult.second)
      continue;

    for (const auto *U : F->users()) {
      const CallInst *CI = dyn_cast<const CallInst>(U);
      if (!CI)
        continue;

      const Function *ParentF = CI->getFunction();

      if (VisitedFunctions.count(ParentF))
        continue;

      if (ParentF->hasFnAttribute("referenced-indirectly"))
        return {};

      if (ParentF->getCallingConv() == CallingConv::SPIR_KERNEL)
        KernelNames.push_back(ParentF->getName());

      FunctionsToVisit.push(ParentF);
    }
  }

  return {std::move(KernelNames)};
}

std::vector<StringRef> getKernelNamesUsingAssert(const Module &M) {
  auto *DevicelibAssertFailFunction = M.getFunction("__devicelib_assert_fail");
  if (!DevicelibAssertFailFunction)
    return {};

  auto TraverseResult =
      traverseCGToFindSPIRKernels(DevicelibAssertFailFunction);

  if (TraverseResult.hasValue())
    return std::move(*TraverseResult);

  // Here we reached "referenced-indirectly", so we need to find all kernels and
  // return them.
  std::vector<StringRef> SPIRKernelNames;
  for (const Function &F : M) {
    if (F.getCallingConv() == CallingConv::SPIR_KERNEL)
      SPIRKernelNames.push_back(F.getName());
  }

  return SPIRKernelNames;
}

// Gets reqd_work_group_size information for function Func.
std::vector<uint32_t> getKernelReqdWorkGroupSizeMetadata(const Function &Func) {
  auto *ReqdWorkGroupSizeMD = Func.getMetadata("reqd_work_group_size");
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

// The function joins names of entry points from one split module to a single
// std::string with '\n' as delimiter.
std::string
collectSymbolsList(const module_split::EntryPointVec &ModuleEntryPoints) {
  std::string SymbolsStr;
  for (const auto *F : ModuleEntryPoints)
    SymbolsStr = (Twine(SymbolsStr) + Twine(F->getName()) + Twine('\n')).str();
  return SymbolsStr;
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

void saveModuleIR(Module &M, StringRef OutFilename) {
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

void saveModuleProperties(module_split::ModuleDesc &ResM,
                          const GlobalBinImageProps &ImgPSInfo,
                          const std::string &PropSetFile) {
  Module &M = ResM.getModule();
  using PropSetRegTy = llvm::util::PropertySetRegistry;
  PropSetRegTy PropSet;

  {
    uint32_t MRMask = getSYCLDeviceLibReqMask(M);
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
    SpecConstantsPass::collectSpecConstantDefaultValuesMetadata(M,
                                                                DefaultValues);
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

  if (ImgPSInfo.EmitExportedSymbols) {
    // Extract the exported functions for a result module
    for (const auto *F : ResM.EntryPoints.Functions)
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

  if (ImgPSInfo.EmitDeviceGlobalPropSet) {
    // Extract device global maps per module
    auto DevGlobalPropertyMap = collectDeviceGlobalProperties(M);
    if (!DevGlobalPropertyMap.empty())
      PropSet.add(PropSetRegTy::SYCL_DEVICE_GLOBALS, DevGlobalPropertyMap);
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
void lowerEsimdConstructs(Module &M) {
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

bool processSpecConstants(Module &M) {
  if (SpecConstLower.getNumOccurrences() == 0)
    return false;

  ModulePassManager RunSpecConst;
  ModuleAnalysisManager MAM;
  bool SetSpecConstAtRT = (SpecConstLower == SC_USE_RT_VAL);
  SpecConstantsPass SCP(SetSpecConstAtRT);
  // Register required analysis
  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  RunSpecConst.addPass(std::move(SCP));

  // Perform the spec constant intrinsics transformation on resulting module
  PreservedAnalyses Res = RunSpecConst.run(M, MAM);
  return !Res.areAllPreserved();
}

bool processCompileTimeProperties(Module &M) {
  // TODO: the early exit can be removed as soon as we have compile-time
  // properties not attached to device globals.
  if (DeviceGlobals.getNumOccurrences() == 0)
    return false;

  ModulePassManager RunCompileTimeProperties;
  ModuleAnalysisManager MAM;
  // Register required analysis
  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  RunCompileTimeProperties.addPass(CompileTimePropertiesPass());

  // Enrich the module with compile-time properties metadata
  PreservedAnalyses Res = RunCompileTimeProperties.run(M, MAM);
  return !Res.areAllPreserved();
}

// Removes the global variable "llvm.used" and returns true on success.
// "llvm.used" is a global constant array containing references to kernels
// available in the module and callable from host code. The elements of
// the array are ConstantExpr bitcast to i8*.
// The variable must be removed as it is a) has done the job to the moment
// of this function call and b) the references to the kernels callable from
// host must not have users.
static bool removeSYCLKernelsConstRefArray(GlobalVariable *GV) {
  assert(GV->getGlobalIdentifier() == "llvm.used" &&
         "Unexpected global value is passed for removal, should be llvm.used");
  assert(GV->user_empty() && "Unexpected llvm.used users");
  Constant *Initializer = GV->getInitializer();
  GV->setInitializer(nullptr);
  GV->eraseFromParent();

  // Destroy the initializer and all operands of it.
  SmallVector<Constant *, 8> IOperands;
  for (auto It = Initializer->op_begin(); It != Initializer->op_end(); It++)
    IOperands.push_back(cast<Constant>(*It));
  assert(llvm::isSafeToDestroyConstant(Initializer) &&
         "Cannot remove initializer of llvm.used global");
  Initializer->destroyConstant();
  for (auto It = IOperands.begin(); It != IOperands.end(); It++) {
    auto Op = (*It)->getOperand(0);
    auto *F = dyn_cast<Function>(Op);
    if (llvm::isSafeToDestroyConstant(*It)) {
      (*It)->destroyConstant();
    } else if (F) {
      // The element in "llvm.used" array has other users. That is Ok for
      // specialization constants, but is wrong for kernels.
      llvm::report_fatal_error("Unexpected usage of SYCL kernel");
    }

    // Remove unused kernel declarations to avoid LLVM IR check fails.
    if (F && F->isDeclaration())
      F->eraseFromParent();
  }
  return true;
}

using TableFiles = std::map<StringRef, string_vector>;

TableFiles processOneModule(std::unique_ptr<Module> M, bool IsEsimd,
                            bool SyclAndEsimdCode) {
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
  if (GlobalVariable *GV = M->getGlobalVariable("llvm.used"))
    IsLLVMUsedRemoved = removeSYCLKernelsConstRefArray(GV);

  if (IsEsimd && LowerEsimd)
    lowerEsimdConstructs(*M);

  module_split::IRSplitMode Mode = (SplitMode.getNumOccurrences() > 0)
                                       ? SplitMode
                                       : module_split::SPLIT_NONE;
  std::unique_ptr<module_split::ModuleSplitterBase> MSplit =
      module_split::getSplitterByMode(std::move(M), Mode, IROutputOnly,
                                      EmitOnlyKernelsAsEntryPoints,
                                      DeviceGlobals);

  StringRef FileSuffix = IsEsimd ? "esimd_" : "";

  for (size_t I = 0; I < MSplit->totalSplits(); ++I) {
    module_split::ModuleDesc ResMDesc = MSplit->nextSplit();
    Module &ResM = ResMDesc.getModule();

    bool SpecConstsMet = processSpecConstants(ResM);
    bool CompileTimePropertiesMet = processCompileTimeProperties(ResM);

    if (IROutputOnly) {
      // the result is the transformed input LLVM IR file rather than a file
      // table
      saveModuleIR(ResM, OutputFilename);
      return TblFiles;
    }

    {
      // Reuse input module with only regular SYCL kernels if there were
      // no spec constants and no splitting.
      // We cannot reuse input module for ESIMD code since it was transformed.
      std::string ResModuleFile{};
      bool CanReuseInputModule = !SyclAndEsimdCode && !IsEsimd &&
                                 !IsLLVMUsedRemoved && !SpecConstsMet &&
                                 !CompileTimePropertiesMet &&
                                 (MSplit->totalSplits() == 1);
      if (CanReuseInputModule)
        ResModuleFile = InputFilename;
      else {
        ResModuleFile =
            makeResultFileName((OutputAssembly) ? ".ll" : ".bc", I, FileSuffix);
        saveModuleIR(ResM, ResModuleFile);
      }
      // "Code" column is always output
      TblFiles[COL_CODE].push_back(ResModuleFile);
    }

    {
      GlobalBinImageProps ImgPSInfo = {SpecConstsMet,
                                       EmitKernelParamInfo,
                                       EmitProgramMetadata,
                                       EmitExportedSymbols,
                                       IsEsimd,
                                       DeviceGlobals};
      std::string PropSetFile = makeResultFileName(".prop", I, FileSuffix);
      saveModuleProperties(ResMDesc, ImgPSInfo, PropSetFile);
      TblFiles[COL_PROPS].push_back(PropSetFile);
    }

    if (DoSymGen) {
      // extract symbols from module
      std::string ResultSymbolsList =
          collectSymbolsList(ResMDesc.EntryPoints.Functions);
      std::string ResultSymbolsFile = makeResultFileName(".sym", I, FileSuffix);
      writeToFile(ResultSymbolsFile, ResultSymbolsList);
      TblFiles[COL_SYM].push_back(ResultSymbolsFile);
    }
  }

  return TblFiles;
}

TableFiles processInputModule(std::unique_ptr<Module> M) {
  std::unique_ptr<module_split::ModuleSplitterBase> SrcSplit =
      module_split::getSplitterByKernelType(std::move(M), SplitEsimd,
                                            EmitOnlyKernelsAsEntryPoints);
  // Do we have both Sycl and Esimd code?
  bool SyclAndEsimdCode = (SrcSplit->totalSplits() == 2);
  TableFiles MergedTblFiles;
  while (SrcSplit->hasMoreSplits()) {
    module_split::ModuleDesc MDesc = SrcSplit->nextSplit();
    TableFiles ResTblFiles = processOneModule(
        std::move(MDesc.M), SplitEsimd && MDesc.isEsimd(), SyclAndEsimdCode);
    // Merge the two resulting file maps
    for (auto &ColumnStr : {COL_CODE, COL_PROPS, COL_SYM}) {
      auto &ResFiles = ResTblFiles[ColumnStr];
      auto &MergedFiles = MergedTblFiles[ColumnStr];
      std::copy(ResFiles.begin(), ResFiles.end(),
                std::back_inserter(MergedFiles));
    }
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
  bool DoDeviceGlobals = DeviceGlobals.getNumOccurrences() > 0;

  if (!DoSplit && !DoSpecConst && !DoSymGen && !DoParamInfo &&
      !DoProgMetadata && !DoSplitEsimd && !DoExportedSyms && !DoDeviceGlobals) {
    errs() << "no actions specified; try --help for usage info\n";
    return 1;
  }
  if (IROutputOnly && (DoSplit && SplitMode != module_split::SPLIT_AUTO)) {
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

  if (OutputFilename.getNumOccurrences() == 0)
    OutputFilename = (Twine(sys::path::stem(InputFilename)) + ".files").str();

  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseIRFile(InputFilename, Err, Context);
  if (!M) {
    Err.print(argv[0], errs());
    return 1;
  }

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
