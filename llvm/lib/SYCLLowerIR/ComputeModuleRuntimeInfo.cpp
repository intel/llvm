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
#include "llvm/Demangle/Demangle.h"
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
bool isModuleUsingAsan(const Module &M) {
  NamedMDNode *MD = M.getNamedMetadata("device.sanitizer");
  if (MD == nullptr)
    return false;
  assert(MD->getNumOperands() != 0);
  auto *MDVal = cast<MDString>(MD->getOperand(0)->getOperand(0));
  return MDVal->getString() == "asan";
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
// Gets work_group_num_dim information for function Func, conviniently 0 if
// metadata is not present.
uint32_t getKernelWorkGroupNumDim(const Function &Func) {
  MDNode *MaxDimMD = Func.getMetadata("work_group_num_dim");
  if (!MaxDimMD)
    return 0;
  assert(MaxDimMD->getNumOperands() == 1 && "Malformed node.");
  return mdconst::extract<ConstantInt>(MaxDimMD->getOperand(0))->getZExtValue();
}

PropSetRegTy computeModuleProperties(const Module &M,
                                     const EntryPointSet &EntryPoints,
                                     const GlobalBinImageProps &GlobProps,
                                     bool SpecConstsMet,
                                     bool IsSpecConstantDefault) {

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
  if (SpecConstsMet) {
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
      if ( // A function that can be imported may still be defined in one split
           // image. Only add import property if this is not the image where the
           // function is defined.
          F.isDeclaration() && module_split::canBeImportedFunction(F)) {

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
    // Add reqd_work_group_size and work_group_num_dim information to
    // program metadata.
    for (const Function &Func : M.functions()) {
      std::vector<uint32_t> KernelReqdWorkGroupSize =
          getKernelReqdWorkGroupSizeMetadata(Func);
      if (!KernelReqdWorkGroupSize.empty()) {
        MetadataNames.push_back(Func.getName().str() + "@reqd_work_group_size");
        PropSet.add(PropSetRegTy::SYCL_PROGRAM_METADATA, MetadataNames.back(),
                    KernelReqdWorkGroupSize);
      }

      uint32_t WorkGroupNumDim = getKernelWorkGroupNumDim(Func);
      if (WorkGroupNumDim) {
        MetadataNames.push_back(Func.getName().str() + "@work_group_num_dim");
        PropSet.add(PropSetRegTy::SYCL_PROGRAM_METADATA, MetadataNames.back(),
                    WorkGroupNumDim);
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
  bool SeenESIMDFunction = false;
  bool SeenSYCLFunction = false;
  for (const auto &F : M) {
    if (llvm::module_split::isESIMDFunction(F))
      SeenESIMDFunction = true;
    else if (utils::isSYCLExternalFunction(&F) &&
             !F.getName().starts_with("__itt"))
      SeenSYCLFunction = true;
  }
  if (SeenESIMDFunction && !SeenSYCLFunction)
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
  if (!SeenESIMDFunction || !SeenSYCLFunction) {
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
      PropSet.add(PropSetRegTy::SYCL_MISC_PROP, "asanUsed", true);
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

  if (IsSpecConstantDefault)
    PropSet.add(PropSetRegTy::SYCL_MISC_PROP, "specConstsReplacedWithDefault",
                1);

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
