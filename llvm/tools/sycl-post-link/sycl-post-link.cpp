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

#include "ModuleSplitter.h"
#include "SYCLDeviceLibReqMask.h"
#include "SYCLDeviceRequirements.h"
#include "SYCLKernelParamOptInfo.h"
#include "SpecConstants.h"
#include "Support.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/GenXIntrinsics/GenXSPIRVWriterAdaptor.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRPrinter/IRPrintingPasses.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/SYCLLowerIR/CompileTimePropertiesPass.h"
#include "llvm/SYCLLowerIR/DeviceGlobals.h"
#include "llvm/SYCLLowerIR/ESIMD/ESIMDUtils.h"
#include "llvm/SYCLLowerIR/ESIMD/LowerESIMD.h"
#include "llvm/SYCLLowerIR/HostPipes.h"
#include "llvm/SYCLLowerIR/LowerInvokeSimd.h"
#include "llvm/SYCLLowerIR/SYCLUtils.h"
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
#include "llvm/Transforms/IPO/StripDeadPrototypes.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/SROA.h"
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

#ifdef NDEBUG
#define DUMP_ENTRY_POINTS(...)
#else
constexpr int DebugPostLink = 0;

#define DUMP_ENTRY_POINTS(...)                                                 \
  if (DebugPostLink > 0) {                                                     \
    llvm::module_split::dumpEntryPoints(__VA_ARGS__);                          \
  }
#endif // NDEBUG

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

cl::opt<module_split::IRSplitMode> SplitMode(
    "split", cl::desc("split input module"), cl::Optional,
    cl::init(module_split::SPLIT_NONE),
    cl::values(clEnumValN(module_split::SPLIT_PER_TU, "source",
                          "1 output module per source (translation unit)"),
               clEnumValN(module_split::SPLIT_PER_KERNEL, "kernel",
                          "1 output module per kernel"),
               clEnumValN(module_split::SPLIT_AUTO, "auto",
                          "Choose split mode automatically")),
    cl::cat(PostLinkCat));

cl::opt<bool> DoSymGen{"symbols", cl::desc("generate exported symbol files"),
                       cl::cat(PostLinkCat)};

enum SpecConstLowerMode { SC_NATIVE_MODE, SC_EMULATION_MODE };

cl::opt<SpecConstLowerMode> SpecConstLower{
    "spec-const",
    cl::desc("lower and generate specialization constants information"),
    cl::Optional,
    cl::init(SC_NATIVE_MODE),
    cl::values(
        clEnumValN(SC_NATIVE_MODE, "native",
                   "lower spec constants to native spirv instructions so that "
                   "these values could be set at runtime"),
        clEnumValN(
            SC_EMULATION_MODE, "emulation",
            "remove specialization constants and replace it with emulation")),
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

cl::opt<bool> GenerateDeviceImageWithDefaultSpecConsts{
    "generate-device-image-default-spec-consts",
    cl::desc("Generate new device image(s) which is a copy of output images "
             "but contain specialization constants "
             "replaced with default values from specialization id(s)."),
    cl::cat(PostLinkCat)};

struct GlobalBinImageProps {
  bool EmitKernelParamInfo;
  bool EmitProgramMetadata;
  bool EmitExportedSymbols;
  bool EmitDeviceGlobalPropSet;
};

struct IrPropSymFilenameTriple {
  std::string Ir;
  std::string Prop;
  std::string Sym;
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
std::optional<std::vector<StringRef>>
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

  if (TraverseResult.has_value())
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

bool isModuleUsingAsan(const Module &M) {
  auto *AsanInitFunction = M.getFunction("__asan_init");
  return AsanInitFunction;
}

// Gets reqd_work_group_size information for function Func.
std::vector<uint32_t> getKernelReqdWorkGroupSizeMetadata(const Function &Func) {
  MDNode *ReqdWorkGroupSizeMD = Func.getMetadata("reqd_work_group_size");
  if (!ReqdWorkGroupSizeMD)
    return {};
  size_t NumOperands = ReqdWorkGroupSizeMD->getNumOperands();
  assert(NumOperands >= 1 && NumOperands <= 3 &&
         "reqd_work_group_size does not have between 1 and 3 operands.");
  std::vector<uint32_t> OutVals;
  OutVals.reserve(NumOperands);
  for (const MDOperand &MDOp : ReqdWorkGroupSizeMD->operands())
    OutVals.push_back(mdconst::extract<ConstantInt>(MDOp)->getZExtValue());
  return OutVals;
}

// Creates a filename based on current output filename, given extension,
// sequential ID and suffix.
std::string makeResultFileName(Twine Ext, int I, StringRef Suffix) {
  const StringRef Dir0 = OutputDir.getNumOccurrences() > 0
                             ? OutputDir
                             : sys::path::parent_path(OutputFilename);
  const StringRef Sep = sys::path::get_separator();
  std::string Dir = Dir0.str();
  if (!Dir0.empty() && !Dir0.endswith(Sep))
    Dir += Sep.str();
  return Dir + sys::path::stem(OutputFilename).str() + Suffix.str() + "_" +
         std::to_string(I) + Ext.str();
}

void saveModuleIR(Module &M, StringRef OutFilename) {
  std::error_code EC;
  raw_fd_ostream Out{OutFilename, EC, sys::fs::OF_None};
  checkError(EC, "error opening the file '" + OutFilename + "'");

  ModulePassManager MPM;
  ModuleAnalysisManager MAM;
  PassBuilder PB;
  PB.registerModuleAnalyses(MAM);
  if (OutputAssembly)
    MPM.addPass(PrintModulePass(Out));
  else if (Force || !CheckBitcodeOutputToConsole(Out))
    MPM.addPass(BitcodeWriterPass(Out));
  MPM.run(M, MAM);
}

std::string saveModuleIR(Module &M, int I, StringRef Suff) {
  DUMP_ENTRY_POINTS(M, EmitOnlyKernelsAsEntryPoints, "saving IR");
  StringRef FileExt = (OutputAssembly) ? ".ll" : ".bc";
  std::string OutFilename = makeResultFileName(FileExt, I, Suff);
  saveModuleIR(M, OutFilename);
  return OutFilename;
}

std::string saveModuleProperties(module_split::ModuleDesc &MD,
                                 const GlobalBinImageProps &GlobProps, int I,
                                 StringRef Suff) {
  using PropSetRegTy = llvm::util::PropertySetRegistry;
  PropSetRegTy PropSet;
  Module &M = MD.getModule();
  {
    uint32_t MRMask = getSYCLDeviceLibReqMask(M);
    std::map<StringRef, uint32_t> RMEntry = {{"DeviceLibReqMask", MRMask}};
    PropSet.add(PropSetRegTy::SYCL_DEVICELIB_REQ_MASK, RMEntry);
  }
  {
    std::map<StringRef, llvm::util::PropertyValue> Requirements;
    getSYCLDeviceRequirements(MD, Requirements);
    PropSet.add(PropSetRegTy::SYCL_DEVICE_REQUIREMENTS, Requirements);
  }
  if (MD.Props.SpecConstsMet) {
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
  if (GlobProps.EmitKernelParamInfo) {
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
  if (GlobProps.EmitExportedSymbols) {
    // extract exported functions if any and save them into property set
    for (const auto *F : MD.entries()) {
      // TODO FIXME some of SYCL/ESIMD functions maybe marked with __regcall CC,
      // so they won't make it into the export list. Should the check be
      // F->getCallingConv() != CallingConv::SPIR_KERNEL?
      if (F->getCallingConv() == CallingConv::SPIR_FUNC) {
        PropSet[PropSetRegTy::SYCL_EXPORTED_SYMBOLS].insert(
            {F->getName(), true});
      }
    }
  }
  // Metadata names may be composite so we keep them alive until the
  // properties have been written.
  SmallVector<std::string, 4> MetadataNames;

  if (GlobProps.EmitProgramMetadata) {
    auto &ProgramMetadata = PropSet[PropSetRegTy::SYCL_PROGRAM_METADATA];

    // Add reqd_work_group_size information to program metadata
    for (const Function &Func : M.functions()) {
      std::vector<uint32_t> KernelReqdWorkGroupSize =
          getKernelReqdWorkGroupSizeMetadata(Func);
      if (KernelReqdWorkGroupSize.empty())
        continue;
      MetadataNames.push_back(Func.getName().str() + "@reqd_work_group_size");
      ProgramMetadata.insert_or_assign(MetadataNames.back(),
                                       KernelReqdWorkGroupSize);
    }

    // Add global_id_mapping information with mapping between device-global
    // unique identifiers and the variable's name in the IR.
    for (auto &GV : M.globals()) {
      if (!isDeviceGlobalVariable(GV))
        continue;

      StringRef GlobalID = getGlobalVariableUniqueId(GV);
      MetadataNames.push_back(GlobalID.str() + "@global_id_mapping");
      ProgramMetadata.insert_or_assign(MetadataNames.back(), GV.getName());
    }
  }
  if (MD.isESIMD()) {
    PropSet[PropSetRegTy::SYCL_MISC_PROP].insert_or_assign("isEsimdImage",
                                                           true);
  }
  {
    StringRef RegAllocModeAttr = "sycl-register-alloc-mode";
    uint32_t RegAllocModeVal;

    bool HasRegAllocMode = llvm::any_of(MD.entries(), [&](const Function *F) {
      if (!F->hasFnAttribute(RegAllocModeAttr))
        return false;
      const auto &Attr = F->getFnAttribute(RegAllocModeAttr);
      RegAllocModeVal = getAttributeAsInteger<uint32_t>(Attr);
      return true;
    });
    if (HasRegAllocMode) {
      PropSet[PropSetRegTy::SYCL_MISC_PROP].insert(
          {RegAllocModeAttr, RegAllocModeVal});
    }
  }

  {
    StringRef GRFSizeAttr = "sycl-grf-size";
    uint32_t GRFSizeVal;

    bool HasGRFSize = llvm::any_of(MD.entries(), [&](const Function *F) {
      if (!F->hasFnAttribute(GRFSizeAttr))
        return false;
      const auto &Attr = F->getFnAttribute(GRFSizeAttr);
      GRFSizeVal = getAttributeAsInteger<uint32_t>(Attr);
      return true;
    });
    if (HasGRFSize) {
      PropSet[PropSetRegTy::SYCL_MISC_PROP].insert({GRFSizeAttr, GRFSizeVal});
    }
  }

  // FIXME: Remove 'if' below when possible
  // GPU backend has a problem with accepting optimization level options in form
  // described by Level Zero specification (-ze-opt-level=1) when 'invoke_simd'
  // functionality is involved. JIT compilation results in the following error:
  //     error: VLD: Failed to compile SPIR-V with following error:
  //     invalid api option: -ze-opt-level=O1
  //     -11 (PI_ERROR_BUILD_PROGRAM_FAILURE)
  // 'if' below essentially preserves the behavior (presumably mistakenly)
  // implemented in intel/llvm#8763: ignore 'optLevel' property for images which
  // were produced my merge after ESIMD split
  if (MD.getEntryPointGroup().Props.HasESIMD !=
      module_split::SyclEsimdSplitStatus::SYCL_AND_ESIMD) {
    // Handle sycl-optlevel property
    int OptLevel = -1;
    for (const Function *F : MD.entries()) {
      if (!F->hasFnAttribute(llvm::sycl::utils::ATTR_SYCL_OPTLEVEL))
        continue;

      // getAsInteger returns true on error
      if (!F->getFnAttribute(llvm::sycl::utils::ATTR_SYCL_OPTLEVEL)
               .getValueAsString()
               .getAsInteger(10, OptLevel)) {
        // It is expected that device-code split has separated kernels with
        // different values of sycl-optlevel attribute. Therefore, it is enough
        // to only look at the first function with such attribute to compute
        // the property for the whole device image.
        break;
      }
    }

    if (OptLevel != -1)
      PropSet[PropSetRegTy::SYCL_MISC_PROP].insert_or_assign("optLevel",
                                                             OptLevel);
  }
  {
    std::vector<StringRef> FuncNames = getKernelNamesUsingAssert(M);
    for (const StringRef &FName : FuncNames)
      PropSet[PropSetRegTy::SYCL_ASSERT_USED].insert_or_assign(FName, true);
  }

  {
    if (isModuleUsingAsan(M))
      PropSet[PropSetRegTy::SYCL_MISC_PROP].insert_or_assign("asanUsed", true);
  }

  if (GlobProps.EmitDeviceGlobalPropSet) {
    // Extract device global maps per module
    auto DevGlobalPropertyMap = collectDeviceGlobalProperties(M);
    if (!DevGlobalPropertyMap.empty())
      PropSet.add(PropSetRegTy::SYCL_DEVICE_GLOBALS, DevGlobalPropertyMap);
  }

  auto HostPipePropertyMap = collectHostPipeProperties(M);
  if (!HostPipePropertyMap.empty()) {
    PropSet.add(PropSetRegTy::SYCL_HOST_PIPES, HostPipePropertyMap);
  }

  if (MD.isSpecConstantDefault())
    PropSet[PropSetRegTy::SYCL_MISC_PROP].insert_or_assign(
        "specConstsReplacedWithDefault", 1);

  std::error_code EC;
  std::string SCFile = makeResultFileName(".prop", I, Suff);
  raw_fd_ostream SCOut(SCFile, EC);
  checkError(EC, "error opening file '" + SCFile + "'");
  PropSet.write(SCOut);

  return SCFile;
}

// Saves specified collection of symbols to a file.
std::string saveModuleSymbolTable(const module_split::EntryPointSet &Es, int I,
                                  StringRef Suffix) {
#ifndef NDEBUG
  if (DebugPostLink > 0) {
    llvm::errs() << "ENTRY POINTS saving Sym table {\n";
    for (const auto *F : Es) {
      llvm::errs() << "  " << F->getName() << "\n";
    }
    llvm::errs() << "}\n";
  }
#endif // NDEBUG
  // Concatenate names of the input entry points with "\n".
  std::string SymT;

  for (const auto *F : Es) {
    SymT = (Twine(SymT) + Twine(F->getName()) + Twine("\n")).str();
  }
  // Save to file.
  std::string OutFileName = makeResultFileName(".sym", I, Suffix);
  writeToFile(OutFileName, SymT);
  return OutFileName;
}

template <class PassClass> bool runModulePass(Module &M) {
  ModulePassManager MPM;
  ModuleAnalysisManager MAM;
  // Register required analysis
  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  MPM.addPass(PassClass{});
  PreservedAnalyses Res = MPM.run(M, MAM);
  return !Res.areAllPreserved();
}

// When ESIMD code was separated from the regular SYCL code,
// we can safely process ESIMD part.
// TODO: support options like -debug-pass, -print-[before|after], and others
bool lowerEsimdConstructs(module_split::ModuleDesc &MD) {
  LoopAnalysisManager LAM;
  CGSCCAnalysisManager CGAM;
  FunctionAnalysisManager FAM;
  ModuleAnalysisManager MAM;

  PassBuilder PB;
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  ModulePassManager MPM;
  MPM.addPass(SYCLLowerESIMDPass{});

  if (!OptLevelO0) {
    FunctionPassManager FPM;
    FPM.addPass(SROAPass(SROAOptions::ModifyCFG));
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  }
  MPM.addPass(ESIMDOptimizeVecArgCallConvPass{});
  FunctionPassManager MainFPM;
  MainFPM.addPass(ESIMDLowerLoadStorePass{});

  if (!OptLevelO0) {
    MainFPM.addPass(SROAPass(SROAOptions::ModifyCFG));
    MainFPM.addPass(EarlyCSEPass(true));
    MainFPM.addPass(InstCombinePass{});
    MainFPM.addPass(DCEPass{});
    // TODO: maybe remove some passes below that don't affect code quality
    MainFPM.addPass(SROAPass(SROAOptions::ModifyCFG));
    MainFPM.addPass(EarlyCSEPass(true));
    MainFPM.addPass(InstCombinePass{});
    MainFPM.addPass(DCEPass{});
  }
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(MainFPM)));
  MPM.addPass(GenXSPIRVWriterAdaptor(/*RewriteTypes=*/true,
                                     /*RewriteSingleElementVectorsIn*/ false));
  // GenXSPIRVWriterAdaptor pass replaced some functions with "rewritten"
  // versions so the entry point table must be rebuilt.
  // TODO Change entry point search to analysis?
  std::vector<std::string> Names;
  MD.saveEntryPointNames(Names);
  PreservedAnalyses Res = MPM.run(MD.getModule(), MAM);
  MD.rebuildEntryPoints(Names);
  return !Res.areAllPreserved();
}

// Compute the filename suffix for the module
StringRef getModuleSuffix(const module_split::ModuleDesc &MD) {
  return MD.isESIMD() ? "_esimd" : "";
}

// @param MD Module descriptor to save
// @param IRFilename filename of already available IR component. If not empty,
//   IR component saving is skipped, and this file name is recorded as such in
//   the result.
// @return a triple of files where IR, Property and Symbols components of the
//   Module descriptor are written respectively.
IrPropSymFilenameTriple saveModule(module_split::ModuleDesc &MD, int I,
                                   StringRef IRFilename = "") {
  IrPropSymFilenameTriple Res;
  StringRef Suffix = getModuleSuffix(MD);

  if (!IRFilename.empty()) {
    // don't save IR, just record the filename
    Res.Ir = IRFilename.str();
  } else {
    MD.cleanup();
    Res.Ir = saveModuleIR(MD.getModule(), I, Suffix);
  }
  GlobalBinImageProps Props = {EmitKernelParamInfo, EmitProgramMetadata,
                               EmitExportedSymbols, DeviceGlobals};
  Res.Prop = saveModuleProperties(MD, Props, I, Suffix);

  if (DoSymGen) {
    // save the names of the entry points - the symbol table
    Res.Sym = saveModuleSymbolTable(MD.entries(), I, Suffix);
  }
  return Res;
}

module_split::ModuleDesc link(module_split::ModuleDesc &&MD1,
                              module_split::ModuleDesc &&MD2) {
  std::vector<std::string> Names;
  MD1.saveEntryPointNames(Names);
  MD2.saveEntryPointNames(Names);
  bool link_error = llvm::Linker::linkModules(
      MD1.getModule(), std::move(MD2.releaseModulePtr()));

  if (link_error) {
    error(" error when linking SYCL and ESIMD modules");
  }
  module_split::ModuleDesc Res(MD1.releaseModulePtr(), std::move(Names));
  Res.assignMergedProperties(MD1, MD2);
  Res.Name = "linked[" + MD1.Name + "," + MD2.Name + "]";
  return Res;
}

bool processSpecConstants(module_split::ModuleDesc &MD) {
  MD.Props.SpecConstsMet = false;

  if (SpecConstLower.getNumOccurrences() == 0)
    return false;

  ModulePassManager RunSpecConst;
  ModuleAnalysisManager MAM;
  SpecConstantsPass SCP(SpecConstLower == SC_NATIVE_MODE
                            ? SpecConstantsPass::HandlingMode::native
                            : SpecConstantsPass::HandlingMode::emulation);
  // Register required analysis
  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  RunSpecConst.addPass(std::move(SCP));

  // Perform the spec constant intrinsics transformation on resulting module
  PreservedAnalyses Res = RunSpecConst.run(MD.getModule(), MAM);
  MD.Props.SpecConstsMet = !Res.areAllPreserved();
  return MD.Props.SpecConstsMet;
}

/// Function generates the copy of the given ModuleDesc where all uses of
/// Specialization Constants are replaced by corresponding default values.
/// If the Module in MD doesn't contain specialization constants then
/// std::nullopt is returned.
std::optional<module_split::ModuleDesc>
processSpecConstantsWithDefaultValues(const module_split::ModuleDesc &MD) {
  std::optional<module_split::ModuleDesc> NewModuleDesc;
  if (!checkModuleContainsSpecConsts(MD.getModule()))
    return NewModuleDesc;

  NewModuleDesc = MD.clone();
  NewModuleDesc->setSpecConstantDefault(true);

  ModulePassManager MPM;
  ModuleAnalysisManager MAM;
  SpecConstantsPass SCP(SpecConstantsPass::HandlingMode::default_values);
  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  MPM.addPass(std::move(SCP));
  MPM.addPass(StripDeadPrototypesPass());

  PreservedAnalyses Res = MPM.run(NewModuleDesc->getModule(), MAM);
  NewModuleDesc->Props.SpecConstsMet = !Res.areAllPreserved();
  assert(NewModuleDesc->Props.SpecConstsMet &&
         "This property should be true since the presence of SpecConsts "
         "has been checked before the run of the pass");
  NewModuleDesc->rebuildEntryPoints();
  return std::move(NewModuleDesc);
}

constexpr int MAX_COLUMNS_IN_FILE_TABLE = 3;

void addTableRow(util::SimpleTable &Table,
                 const IrPropSymFilenameTriple &RowData) {
  SmallVector<StringRef, MAX_COLUMNS_IN_FILE_TABLE> Row;

  for (const std::string *S : {&RowData.Ir, &RowData.Prop, &RowData.Sym}) {
    if (!S->empty()) {
      Row.push_back(StringRef(*S));
    }
  }
  assert(static_cast<size_t>(Table.getNumColumns()) == Row.size());
  Table.addRow(Row);
}

// Removes the global variable "llvm.used" and returns true on success.
// "llvm.used" is a global constant array containing references to kernels
// available in the module and callable from host code. The elements of
// the array are ConstantExpr bitcast to i8*.
// The variable must be removed as it is a) has done the job to the moment
// of this function call and b) the references to the kernels callable from
// host must not have users.
static bool removeSYCLKernelsConstRefArray(Module &M) {
  GlobalVariable *GV = M.getGlobalVariable("llvm.used");

  if (!GV) {
    return false;
  }
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
    auto Op = (*It)->stripPointerCasts();
    auto *F = dyn_cast<Function>(Op);
    if (llvm::isSafeToDestroyConstant(*It)) {
      (*It)->destroyConstant();
    } else if (F && F->getCallingConv() == CallingConv::SPIR_KERNEL &&
               !F->use_empty()) {
      // The element in "llvm.used" array has other users. That is Ok for
      // specialization constants, but is wrong for kernels.
      llvm::report_fatal_error("Unexpected usage of SYCL kernel");
    }

    // Remove unused kernel declarations to avoid LLVM IR check fails.
    if (F && F->isDeclaration() && F->use_empty())
      F->eraseFromParent();
  }
  return true;
}

// Removes all device_global variables from the llvm.compiler.used global
// variable. A device_global with internal linkage will be in llvm.compiler.used
// to avoid the compiler wrongfully removing it during optimizations. However,
// as an effect the device_global variables will also be distributed across
// binaries, even if llvm.compiler.used has served its purpose. To avoid
// polluting other binaries with unused device_global variables, we remove them
// from llvm.compiler.used and erase them if they have no further uses.
static bool removeDeviceGlobalFromCompilerUsed(Module &M) {
  GlobalVariable *GV = M.getGlobalVariable("llvm.compiler.used");
  if (!GV)
    return false;

  // Erase the old llvm.compiler.used. A new one will be created at the end if
  // there are other values in it (other than device_global).
  assert(GV->user_empty() && "Unexpected llvm.compiler.used users");
  Constant *Initializer = GV->getInitializer();
  const auto *VAT = cast<ArrayType>(GV->getValueType());
  GV->setInitializer(nullptr);
  GV->eraseFromParent();

  // Destroy the initializer. Keep the operands so we keep the ones we need.
  SmallVector<Constant *, 8> IOperands;
  for (auto It = Initializer->op_begin(); It != Initializer->op_end(); It++)
    IOperands.push_back(cast<Constant>(*It));
  assert(llvm::isSafeToDestroyConstant(Initializer) &&
         "Cannot remove initializer of llvm.compiler.used global");
  Initializer->destroyConstant();

  // Iterate through all operands. If they are device_global then we drop them
  // and erase them if they have no uses afterwards. All other values are kept.
  SmallVector<Constant *, 8> NewOperands;
  for (auto It = IOperands.begin(); It != IOperands.end(); It++) {
    Constant *Op = *It;
    auto *DG = dyn_cast<GlobalVariable>(Op->stripPointerCasts());

    // If it is not a device_global we keep it.
    if (!DG || !isDeviceGlobalVariable(*DG)) {
      NewOperands.push_back(Op);
      continue;
    }

    // Destroy the device_global operand.
    if (llvm::isSafeToDestroyConstant(Op))
      Op->destroyConstant();

    // Remove device_global if it no longer has any uses.
    if (!DG->isConstantUsed())
      DG->eraseFromParent();
  }

  // If we have any operands left from the original llvm.compiler.used we create
  // a new one with the new size.
  if (!NewOperands.empty()) {
    ArrayType *ATy = ArrayType::get(VAT->getElementType(), NewOperands.size());
    GlobalVariable *NGV =
        new GlobalVariable(M, ATy, false, GlobalValue::AppendingLinkage,
                           ConstantArray::get(ATy, NewOperands), "");
    NGV->setName("llvm.compiler.used");
    NGV->setSection("llvm.metadata");
  }

  return true;
}

SmallVector<module_split::ModuleDesc, 2>
handleESIMD(module_split::ModuleDesc &&MDesc, bool &Modified,
            bool &SplitOccurred) {
  // Do SYCL/ESIMD splitting. It happens always, as ESIMD and SYCL must
  // undergo different set of LLVMIR passes. After this they are linked back
  // together to form single module with disjoint SYCL and ESIMD call graphs
  // unless -split-esimd option is specified. The graphs become disjoint
  // when linked back because functions shared between graphs are cloned and
  // renamed.
  SmallVector<module_split::ModuleDesc, 2> Result = module_split::splitByESIMD(
      std::move(MDesc), EmitOnlyKernelsAsEntryPoints);

  if (Result.size() > 1 && SplitOccurred &&
      (SplitMode == module_split::SPLIT_PER_KERNEL) && !SplitEsimd) {
    // Controversial state reached - SYCL and ESIMD entry points resulting
    // from SYCL/ESIMD split (which is done always) are linked back, since
    // -split-esimd is not specified, but per-kernel split is requested.
    warning("SYCL and ESIMD entry points detected and split mode is "
            "per-kernel, so " +
            SplitEsimd.ValueStr + " must also be specified");
  }
  SplitOccurred |= Result.size() > 1;

  for (auto &MD : Result) {
    DUMP_ENTRY_POINTS(MD.entries(), MD.Name.c_str(), 3);
    if (LowerEsimd && MD.isESIMD())
      Modified |= lowerEsimdConstructs(MD);
  }

  if (!SplitEsimd && Result.size() > 1) {
    // SYCL/ESIMD splitting is not requested, link back into single module.
    assert(Result.size() == 2 &&
           "Unexpected number of modules as results of ESIMD split");
    int ESIMDInd = Result[0].isESIMD() ? 0 : 1;
    int SYCLInd = 1 - ESIMDInd;
    assert(Result[SYCLInd].isSYCL() &&
           "no non-ESIMD module as a result ESIMD split?");

    // ... but before that, make sure no link conflicts will occur.
    Result[ESIMDInd].renameDuplicatesOf(Result[SYCLInd].getModule(), ".esimd");
    module_split::ModuleDesc Linked =
        link(std::move(Result[0]), std::move(Result[1]));
    Linked.restoreLinkageOfDirectInvokeSimdTargets();
    string_vector Names;
    Linked.saveEntryPointNames(Names);
    Linked.cleanup(); // may remove some entry points, need to save/rebuild
    Linked.rebuildEntryPoints(Names);
    Result.clear();
    Result.emplace_back(std::move(Linked));
    DUMP_ENTRY_POINTS(Result.back().entries(), Result.back().Name.c_str(), 3);
    Modified = true;
  }

  return Result;
}

std::unique_ptr<util::SimpleTable>
processInputModule(std::unique_ptr<Module> M) {
  // Construct the resulting table which will accumulate all the outputs.
  SmallVector<StringRef, MAX_COLUMNS_IN_FILE_TABLE> ColumnTitles{
      StringRef(COL_CODE), StringRef(COL_PROPS)};

  if (DoSymGen) {
    ColumnTitles.push_back(COL_SYM);
  }
  Expected<std::unique_ptr<util::SimpleTable>> TableE =
      util::SimpleTable::create(ColumnTitles);
  CHECK_AND_EXIT(TableE.takeError());
  std::unique_ptr<util::SimpleTable> Table = std::move(TableE.get());

  // Used in output filenames generation.
  int ID = 0;

  // Keeps track of any changes made to the input module and report to the user
  // if none were made.
  bool Modified = false;

  // Propagate ESIMD attribute to wrapper functions to prevent
  // spurious splits and kernel link errors.
  Modified |= runModulePass<SYCLFixupESIMDKernelWrapperMDPass>(*M);

  // After linking device bitcode "llvm.used" holds references to the kernels
  // that are defined in the device image. But after splitting device image into
  // separate kernels we may end up with having references to kernel declaration
  // originating from "llvm.used" in the IR that is passed to llvm-spirv tool,
  // and these declarations cause an assertion in llvm-spirv. To workaround this
  // issue remove "llvm.used" from the input module before performing any other
  // actions.
  Modified |= removeSYCLKernelsConstRefArray(*M.get());

  // There may be device_global variables kept alive in "llvm.compiler.used"
  // to keep the optimizer from wrongfully removing them. Since it has served
  // its purpose, these device_global variables can be removed. If they are not
  // used inside the device code after they have been removed from
  // "llvm.compiler.used" they can be erased safely.
  Modified |= removeDeviceGlobalFromCompilerUsed(*M.get());

  // Do invoke_simd processing before splitting because this:
  // - saves processing time (the pass is run once, even though on larger IR)
  // - doing it before SYCL/ESIMD splitting is required for correctness
  const bool InvokeSimdMet = runModulePass<SYCLLowerInvokeSimdPass>(*M);

  if (InvokeSimdMet && SplitEsimd) {
    error("'invoke_simd' calls detected, '-" + SplitEsimd.ArgStr +
          "' must not be specified");
  }
  Modified |= InvokeSimdMet;

  DUMP_ENTRY_POINTS(*M, EmitOnlyKernelsAsEntryPoints, "Input");

  // -ir-output-only assumes single module output thus no code splitting.
  // Violation of this invariant is user error and must've been reported.
  // However, if split mode is "auto", then entry point filtering is still
  // performed.
  assert((!IROutputOnly || (SplitMode == module_split::SPLIT_NONE) ||
          (SplitMode == module_split::SPLIT_AUTO)) &&
         "invalid split mode for IR-only output");

  std::unique_ptr<module_split::ModuleSplitterBase> Splitter =
      module_split::getDeviceCodeSplitter(
          module_split::ModuleDesc{std::move(M)}, SplitMode, IROutputOnly,
          EmitOnlyKernelsAsEntryPoints);
  bool SplitOccurred = Splitter->remainingSplits() > 1;
  Modified |= SplitOccurred;

  // FIXME: this check is not performed for ESIMD splits
  if (DeviceGlobals)
    Splitter->verifyNoCrossModuleDeviceGlobalUsage();

  // It is important that we *DO NOT* preserve all the splits in memory at the
  // same time, because it leads to a huge RAM consumption by the tool on bigger
  // inputs.
  while (Splitter->hasMoreSplits()) {
    module_split::ModuleDesc MDesc = Splitter->nextSplit();
    DUMP_ENTRY_POINTS(MDesc.entries(), MDesc.Name.c_str(), 1);

    MDesc.fixupLinkageOfDirectInvokeSimdTargets();

    SmallVector<module_split::ModuleDesc, 2> MMs =
        handleESIMD(std::move(MDesc), Modified, SplitOccurred);
    assert(MMs.size() && "at least one module is expected after ESIMD split");

    SmallVector<module_split::ModuleDesc, 2> MMsWithDefaultSpecConsts;
    for (size_t I = 0; I != MMs.size(); ++I) {
      if (GenerateDeviceImageWithDefaultSpecConsts) {
        std::optional<module_split::ModuleDesc> NewMD =
            processSpecConstantsWithDefaultValues(MMs[I]);
        if (NewMD)
          MMsWithDefaultSpecConsts.push_back(std::move(*NewMD));
      }

      Modified |= processSpecConstants(MMs[I]);
    }

    if (IROutputOnly) {
      if (SplitOccurred) {
        error("some modules had to be split, '-" + IROutputOnly.ArgStr +
              "' can't be used");
      }
      MMs.front().cleanup();
      saveModuleIR(MMs.front().getModule(), OutputFilename);
      return Table;
    }
    // Empty IR file name directs saveModule to generate one and save IR to
    // it:
    std::string OutIRFileName = "";

    if (!Modified && (OutputFilename.getNumOccurrences() == 0)) {
      assert(!SplitOccurred);
      OutIRFileName = InputFilename; // ... non-empty means "skip IR writing"
      errs() << "sycl-post-link NOTE: no modifications to the input LLVM IR "
                "have been made\n";
    }
    for (module_split::ModuleDesc &IrMD : MMs) {
      IrPropSymFilenameTriple T = saveModule(IrMD, ID, OutIRFileName);
      addTableRow(*Table, T);
    }

    ++ID;

    if (!MMsWithDefaultSpecConsts.empty()) {
      for (size_t i = 0; i != MMsWithDefaultSpecConsts.size(); ++i) {
        module_split::ModuleDesc &IrMD = MMsWithDefaultSpecConsts[i];
        IrPropSymFilenameTriple T = saveModule(IrMD, ID, OutIRFileName);
        addTableRow(*Table, T);
      }

      ++ID;
    }
  }
  return Table;
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
      "  first input module will be split according to the '-split' option\n"
      "  processing algorithm, not distinguishing between SYCL and ESIMD\n"
      "  kernels. Then each resulting module is further split into SYCL and\n"
      "  ESIMD parts if the module has both kinds of entry points.\n"
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
      "  $ sycl-post-link --split=kernel --symbols --spec-const=native \\\n"
      "    -o example.table example.bc\n"
      "will produce 'example.table' file with the following content:\n"
      "  [Code|Properties|Symbols]\n"
      "  example_0.bc|example_0.prop|example_0.sym\n"
      "  example_1.bc|example_1.prop|example_1.sym\n"
      "When only specialization constant processing is needed, the tool can\n"
      "output a single transformed IR file if --ir-output-only is specified:\n"
      "  $ sycl-post-link --ir-output-only --spec-const=emulation \\\n"
      "    -o example_p.bc example.bc\n"
      "will produce single output file example_p.bc suitable for SPIRV\n"
      "translation.\n"
      "--ir-output-only option is not not compatible with split modes other\n"
      "than 'auto'.\n");

  bool DoSplit = SplitMode.getNumOccurrences() > 0;
  bool DoSplitEsimd = SplitEsimd.getNumOccurrences() > 0;
  bool DoLowerEsimd = LowerEsimd.getNumOccurrences() > 0;
  bool DoSpecConst = SpecConstLower.getNumOccurrences() > 0;
  bool DoParamInfo = EmitKernelParamInfo.getNumOccurrences() > 0;
  bool DoProgMetadata = EmitProgramMetadata.getNumOccurrences() > 0;
  bool DoExportedSyms = EmitExportedSymbols.getNumOccurrences() > 0;
  bool DoDeviceGlobals = DeviceGlobals.getNumOccurrences() > 0;
  bool DoGenerateDeviceImageWithDefaulValues =
      GenerateDeviceImageWithDefaultSpecConsts.getNumOccurrences() > 0;

  if (!DoSplit && !DoSpecConst && !DoSymGen && !DoParamInfo &&
      !DoProgMetadata && !DoSplitEsimd && !DoExportedSyms && !DoDeviceGlobals &&
      !DoLowerEsimd) {
    errs() << "no actions specified; try --help for usage info\n";
    return 1;
  }
  if (IROutputOnly && DoSplit && (SplitMode != module_split::SPLIT_AUTO)) {
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
  if (IROutputOnly && DoGenerateDeviceImageWithDefaulValues) {
    errs() << "error: -" << GenerateDeviceImageWithDefaultSpecConsts.ArgStr
           << " can't be used with -" << IROutputOnly.ArgStr << "\n";
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

  if (OutputFilename.getNumOccurrences() == 0) {
    std::string S =
        IROutputOnly ? (OutputAssembly ? ".out.ll" : "out.bc") : ".files";
    OutputFilename = (Twine(sys::path::stem(InputFilename)) + S).str();
  }

  std::unique_ptr<util::SimpleTable> Table = processInputModule(std::move(M));

  // Input module was processed and a single output file was requested.
  if (IROutputOnly)
    return 0;

  // Emit the resulting table
  std::error_code EC;
  raw_fd_ostream Out{OutputFilename, EC, sys::fs::OF_None};
  checkError(EC, "error opening file '" + OutputFilename + "'");
  Table->write(Out);

  return 0;
}
