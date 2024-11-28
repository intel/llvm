//===--- ComputeModuleRuntimeInfo.cpp - compute runtime info for module ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// See comments in the header.
//===----------------------------------------------------------------------===//
#include "llvm/SYCLLowerIR/ComputeModuleRuntimeInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/SYCLLowerIR/CompileTimePropertiesPass.h"
#include "llvm/SYCLLowerIR/DeviceGlobals.h"
#include "llvm/SYCLLowerIR/HostPipes.h"
#include "llvm/SYCLLowerIR/ModuleSplitter.h"
#include "llvm/SYCLLowerIR/SYCLDeviceLibReqMask.h"
#include "llvm/SYCLLowerIR/SYCLKernelParamOptInfo.h"
#include "llvm/SYCLLowerIR/SYCLUtils.h"
#include "llvm/SYCLLowerIR/SpecConstants.h"
#include <queue>
#include <unordered_set>
#ifndef NDEBUG
constexpr int DebugModuleProps = 0;
#endif

namespace llvm::sycl {
namespace {
module_split::SyclEsimdSplitStatus
getSYCLESIMDSplitStatusFromMetadata(const Module &M) {
  auto *SplitMD = M.getNamedMetadata(module_split::SYCL_ESIMD_SPLIT_MD_NAME);
  assert(SplitMD && "Unexpected metadata");
  auto *MDOp = SplitMD->getOperand(0);
  assert(MDOp && "Unexpected metadata operand");
  const auto &MDConst = MDOp->getOperand(0);
  auto *MDVal = mdconst::dyn_extract_or_null<ConstantInt>(MDConst);
  assert(MDVal && "Unexpected metadata operand type");
  uint8_t Val = MDVal->getZExtValue();
  assert(Val < 3 && "Unexpected value for split metadata");
  auto AsEnum = static_cast<module_split::SyclEsimdSplitStatus>(Val);
  return AsEnum;
}
} // namespace

bool isModuleUsingAsan(const Module &M) {
  return M.getNamedGlobal("__AsanKernelMetadata");
}

bool isModuleUsingMsan(const Module &M) {
  return M.getGlobalVariable("__MsanKernelMetadata") != nullptr;
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

// Gets 1- to 3-dimension work-group related information for function Func.
// Returns an empty vector if not present.
template <typename T>
std::vector<T> getKernelWorkGroupMetadata(const Function &Func,
                                          const char *MDName) {
  MDNode *WorkGroupMD = Func.getMetadata(MDName);
  if (!WorkGroupMD)
    return {};
  size_t NumOperands = WorkGroupMD->getNumOperands();
  assert(NumOperands >= 1 && NumOperands <= 3 &&
         "work-group metadata does not have between 1 and 3 operands.");
  std::vector<T> OutVals;
  OutVals.reserve(NumOperands);
  for (const MDOperand &MDOp : WorkGroupMD->operands())
    OutVals.push_back(mdconst::extract<ConstantInt>(MDOp)->getZExtValue());
  return OutVals;
}

// Gets a single-dimensional piece of information for function Func.
// Returns std::nullopt if metadata is not present.
template <typename T>
std::optional<T> getKernelSingleEltMetadata(const Function &Func,
                                            const char *MDName) {
  if (MDNode *MaxDimMD = Func.getMetadata(MDName)) {
    assert(MaxDimMD->getNumOperands() == 1 && "Malformed node.");
    return mdconst::extract<ConstantInt>(MaxDimMD->getOperand(0))
        ->getZExtValue();
  }
  return std::nullopt;
}

PropSetRegTy computeModuleProperties(const Module &M,
                                     const EntryPointSet &EntryPoints,
                                     const GlobalBinImageProps &GlobProps) {

  PropSetRegTy PropSet;
  {
    uint32_t MRMask = getSYCLDeviceLibReqMask(M);
    std::map<StringRef, uint32_t> RMEntry = {{"DeviceLibReqMask", MRMask}};
    PropSet.add(PropSetRegTy::SYCL_DEVICELIB_REQ_MASK, RMEntry);
  }
  {
    PropSet.add(PropSetRegTy::SYCL_DEVICE_REQUIREMENTS,
                computeDeviceRequirements(M, EntryPoints).asMap());
  }

  // extract spec constant maps per each module
  SpecIDMapTy TmpSpecIDMap;
  SpecConstantsPass::collectSpecConstantMetadata(M, TmpSpecIDMap);
  if (!TmpSpecIDMap.empty()) {
    PropSet.add(PropSetRegTy::SYCL_SPECIALIZATION_CONSTANTS, TmpSpecIDMap);

    // Add property with the default values of spec constants
    std::vector<char> DefaultValues;
    SpecConstantsPass::collectSpecConstantDefaultValuesMetadata(M,
                                                                DefaultValues);
    assert(!DefaultValues.empty() &&
           "Expected metadata for spec constant defaults.");
    PropSet.add(PropSetRegTy::SYCL_SPEC_CONSTANTS_DEFAULT_VALUES, "all",
                DefaultValues);
  } else {
#ifndef NDEBUG
    std::vector<char> DefaultValues;
    SpecConstantsPass::collectSpecConstantDefaultValuesMetadata(M,
                                                                DefaultValues);
    assert(DefaultValues.empty() &&
           "Unexpected metadata for spec constant defaults.");
#endif
  }
  if (GlobProps.EmitKernelParamInfo) {
    // extract kernel parameter optimization info per module
    ModuleAnalysisManager MAM;
    // Register required analysis
    MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
    // Register the payload analysis

    MAM.registerPass([&] { return SYCLKernelParamOptInfoAnalysis(); });
    SYCLKernelParamOptInfo PInfo =
        MAM.getResult<SYCLKernelParamOptInfoAnalysis>(const_cast<Module &>(M));

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
    for (const auto *F : EntryPoints) {
      // Virtual functions use a different mechanism of dynamic linking, they
      // should not be registered here.
      if (F->hasFnAttribute("indirectly-callable"))
        continue;
      // TODO FIXME some of SYCL/ESIMD functions maybe marked with __regcall CC,
      // so they won't make it into the export list. Should the check be
      // F->getCallingConv() != CallingConv::SPIR_KERNEL?
      if (F->getCallingConv() == CallingConv::SPIR_FUNC) {
        PropSet.add(PropSetRegTy::SYCL_EXPORTED_SYMBOLS, F->getName(),
                    /*PropVal=*/true);
      }
    }
  }

  if (GlobProps.EmitImportedSymbols) {
    // record imported functions in the property set
    for (const auto &F : M) {
      // A function that can be imported may still be defined in one split
      // image. Only add import property if this is not the image where the
      // function is defined.
      if (!F.isDeclaration())
        continue;

      // Even though virtual functions are considered to be imported by the
      // function below, we shouldn't list them in the property because they
      // use different mechanism for dynamic linking.
      if (F.hasFnAttribute("indirectly-callable"))
        continue;

      if (module_split::canBeImportedFunction(F)) {
        // StripDeadPrototypes is called during module splitting
        // cleanup.  At this point all function decls should have uses.
        assert(!F.use_empty() && "Function F has no uses");
        PropSet.add(PropSetRegTy::SYCL_IMPORTED_SYMBOLS, F.getName(),
                    /*PropVal=*/true);
      }
    }
  }

  // Metadata names may be composite so we keep them alive until the
  // properties have been written.
  SmallVector<std::string, 4> MetadataNames;

  if (GlobProps.EmitProgramMetadata) {
    // Add various pieces of function metadata to program metadata.
    for (const Function &Func : M.functions()) {
      // Note - we're implicitly truncating 64-bit work-group data to 32 bits in
      // all work-group related metadata. All current consumers of this program
      // metadata format only support SYCL ID queries that fit within MAX_INT.
      if (auto KernelReqdWorkGroupSize = getKernelWorkGroupMetadata<uint32_t>(
              Func, "reqd_work_group_size");
          !KernelReqdWorkGroupSize.empty()) {
        MetadataNames.push_back(Func.getName().str() + "@reqd_work_group_size");
        PropSet.add(PropSetRegTy::SYCL_PROGRAM_METADATA, MetadataNames.back(),
                    KernelReqdWorkGroupSize);
      }

      if (auto WorkGroupNumDim = getKernelSingleEltMetadata<uint32_t>(
              Func, "work_group_num_dim")) {
        MetadataNames.push_back(Func.getName().str() + "@work_group_num_dim");
        PropSet.add(PropSetRegTy::SYCL_PROGRAM_METADATA, MetadataNames.back(),
                    *WorkGroupNumDim);
      }

      if (auto KernelMaxWorkGroupSize =
              getKernelWorkGroupMetadata<uint32_t>(Func, "max_work_group_size");
          !KernelMaxWorkGroupSize.empty()) {
        MetadataNames.push_back(Func.getName().str() + "@max_work_group_size");
        PropSet.add(PropSetRegTy::SYCL_PROGRAM_METADATA, MetadataNames.back(),
                    KernelMaxWorkGroupSize);
      }

      if (auto MaxLinearWGSize = getKernelSingleEltMetadata<uint64_t>(
              Func, "max_linear_work_group_size")) {
        MetadataNames.push_back(Func.getName().str() +
                                "@max_linear_work_group_size");
        PropSet.add(PropSetRegTy::SYCL_PROGRAM_METADATA, MetadataNames.back(),
                    *MaxLinearWGSize);
      }
    }

    // Add global_id_mapping information with mapping between device-global
    // unique identifiers and the variable's name in the IR.
    for (auto &GV : M.globals()) {
      if (!isDeviceGlobalVariable(GV))
        continue;

      StringRef GlobalID = getGlobalVariableUniqueId(GV);
      MetadataNames.push_back(GlobalID.str() + "@global_id_mapping");
      PropSet.add(PropSetRegTy::SYCL_PROGRAM_METADATA, MetadataNames.back(),
                  GV.getName());
    }
  }

  module_split::SyclEsimdSplitStatus SplitType =
      getSYCLESIMDSplitStatusFromMetadata(M);

  if (SplitType == module_split::SyclEsimdSplitStatus::ESIMD_ONLY)
    PropSet.add(PropSetRegTy::SYCL_MISC_PROP, "isEsimdImage", true);
  {
    StringRef RegAllocModeAttr = "sycl-register-alloc-mode";
    uint32_t RegAllocModeVal;

    bool HasRegAllocMode = llvm::any_of(EntryPoints, [&](const Function *F) {
      if (!F->hasFnAttribute(RegAllocModeAttr))
        return false;
      const auto &Attr = F->getFnAttribute(RegAllocModeAttr);
      RegAllocModeVal = getAttributeAsInteger<uint32_t>(Attr);
      return true;
    });
    if (HasRegAllocMode) {
      PropSet.add(PropSetRegTy::SYCL_MISC_PROP, RegAllocModeAttr,
                  RegAllocModeVal);
    }
  }

  {
    StringRef GRFSizeAttr = "sycl-grf-size";
    uint32_t GRFSizeVal;

    bool HasGRFSize = llvm::any_of(EntryPoints, [&](const Function *F) {
      if (!F->hasFnAttribute(GRFSizeAttr))
        return false;
      const auto &Attr = F->getFnAttribute(GRFSizeAttr);
      GRFSizeVal = getAttributeAsInteger<uint32_t>(Attr);
      return true;
    });
    if (HasGRFSize) {
      PropSet.add(PropSetRegTy::SYCL_MISC_PROP, GRFSizeAttr, GRFSizeVal);
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
  if (SplitType != module_split::SyclEsimdSplitStatus::SYCL_AND_ESIMD) {
    // Handle sycl-optlevel property
    int OptLevel = -1;
    for (const Function *F : EntryPoints) {
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
      PropSet.add(PropSetRegTy::SYCL_MISC_PROP, "optLevel", OptLevel);
  }
  {
    std::vector<StringRef> FuncNames = getKernelNamesUsingAssert(M);
    for (const StringRef &FName : FuncNames)
      PropSet.add(PropSetRegTy::SYCL_ASSERT_USED, FName, true);
  }

  {
    if (isModuleUsingAsan(M))
      PropSet.add(PropSetRegTy::SYCL_MISC_PROP, "sanUsed", "asan");
    else if (isModuleUsingMsan(M))
      PropSet.add(PropSetRegTy::SYCL_MISC_PROP, "sanUsed", "msan");
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
  bool IsSpecConstantDefault =
      M.getNamedMetadata(
          SpecConstantsPass::SPEC_CONST_DEFAULT_VAL_MODULE_MD_STRING) !=
      nullptr;
  if (IsSpecConstantDefault)
    PropSet.add(PropSetRegTy::SYCL_MISC_PROP, "specConstsReplacedWithDefault",
                1);

  { // Properties related to virtual functions
    StringSet<> UsedVFSets;
    bool AddedVFSetProperty = false;
    for (const Function &F : M) {
      if (F.isDeclaration())
        continue;

      if (F.hasFnAttribute("indirectly-callable")) {
        PropSet.add(PropSetRegTy::SYCL_VIRTUAL_FUNCTIONS,
                    "virtual-functions-set",
                    F.getFnAttribute("indirectly-callable").getValueAsString());
        AddedVFSetProperty = true;
        // Device code split should ensure that virtual functions that belong
        // to different sets are split into separate device images and hence
        // there is no need to scan other functions.
        break;
      }

      if (F.hasFnAttribute("calls-indirectly")) {
        SmallVector<StringRef, 4> Sets;
        F.getFnAttribute("calls-indirectly")
            .getValueAsString()
            .split(Sets, ',', /* MaxSplits */ -1, /* KeepEmpty */ false);
        for (auto Set : Sets)
          UsedVFSets.insert(Set);
      }
    }

    if (!UsedVFSets.empty()) {
      assert(!AddedVFSetProperty &&
             "device image cannot have both virtual-functions-set and "
             "uses-virtual-functions-set property");
      SmallString<128> AllSets;
      for (auto &It : UsedVFSets) {
        if (!AllSets.empty())
          AllSets += ',';
        AllSets += It.getKey();
      }

      PropSet.add(PropSetRegTy::SYCL_VIRTUAL_FUNCTIONS,
                   "uses-virtual-functions-set", AllSets);
    }
  }

  return PropSet;
}
std::string computeModuleSymbolTable(const Module &M,
                                     const EntryPointSet &EntryPoints) {

#ifndef NDEBUG
  if (DebugModuleProps > 0) {
    llvm::errs() << "ENTRY POINTS saving Sym table {\n";
    for (const auto *F : EntryPoints) {
      llvm::errs() << "  " << F->getName() << "\n";
    }
    llvm::errs() << "}\n";
  }
#endif // NDEBUG
  // Concatenate names of the input entry points with "\n".
  std::string SymT;

  for (const auto *F : EntryPoints) {
    SymT = (Twine(SymT) + Twine(F->getName()) + Twine("\n")).str();
  }
  return SymT;
}

} // namespace llvm::sycl
