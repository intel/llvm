//==-------------------------- KernelFusion.cpp ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "KernelFusion.h"
#include "Kernel.h"
#include "NDRangesHelper.h"
#include "Options.h"
#include "fusion/FusionHelper.h"
#include "fusion/FusionPipeline.h"
#include "helper/ConfigHelper.h"
#include "helper/ErrorHandling.h"
#include "translation/KernelTranslation.h"
#include "translation/SPIRVLLVMTranslation.h"
#include <llvm/Support/Error.h>
#include <sstream>

using namespace jit_compiler;

using FusedFunction = helper::FusionHelper::FusedFunction;
using FusedFunctionList = std::vector<FusedFunction>;

static FusionResult errorToFusionResult(llvm::Error &&Err,
                                        const std::string &Msg) {
  std::stringstream ErrMsg;
  ErrMsg << Msg << "\nDetailed information:\n";
  llvm::handleAllErrors(std::move(Err),
                        [&ErrMsg](const llvm::StringError &StrErr) {
                          // Cannot throw an exception here if LLVM itself is
                          // compiled without exception support.
                          ErrMsg << "\t" << StrErr.getMessage() << "\n";
                        });
  return FusionResult{ErrMsg.str().c_str()};
}

static std::vector<jit_compiler::NDRange>
gatherNDRanges(llvm::ArrayRef<SYCLKernelInfo> KernelInformation) {
  std::vector<jit_compiler::NDRange> NDRanges;
  NDRanges.reserve(KernelInformation.size());
  std::transform(KernelInformation.begin(), KernelInformation.end(),
                 std::back_inserter(NDRanges),
                 [](const auto &I) { return I.NDR; });
  return NDRanges;
}

static bool isTargetFormatSupported(BinaryFormat TargetFormat) {
  switch (TargetFormat) {
  case BinaryFormat::SPIRV:
    return true;
  case BinaryFormat::PTX: {
#ifdef FUSION_JIT_SUPPORT_PTX
    return true;
#else  // FUSION_JIT_SUPPORT_PTX
    return false;
#endif // FUSION_JIT_SUPPORT_PTX
  }
  case BinaryFormat::AMDGCN: {
#ifdef FUSION_JIT_SUPPORT_AMDGCN
    return true;
#else  // FUSION_JIT_SUPPORT_AMDGCN
    return false;
#endif // FUSION_JIT_SUPPORT_AMDGCN
  }
  default:
    return false;
  }
}

FusionResult KernelFusion::fuseKernels(
    View<SYCLKernelInfo> KernelInformation, const char *FusedKernelName,
    View<ParameterIdentity> Identities, BarrierFlags BarriersFlags,
    View<ParameterInternalization> Internalization,
    View<jit_compiler::JITConstant> Constants) {

  std::vector<std::string> KernelsToFuse;
  llvm::transform(KernelInformation, std::back_inserter(KernelsToFuse),
                  [](const auto &KI) { return std::string{KI.Name.c_str()}; });

  const auto NDRanges = gatherNDRanges(KernelInformation.to<llvm::ArrayRef>());

  if (!isValidCombination(NDRanges)) {
    return FusionResult{
        "Cannot fuse kernels with different offsets or local sizes, or "
        "different global sizes in dimensions [2, N) and non-zero offsets, "
        "or those whose fusion would yield non-uniform work-groups sizes"};
  }

  bool IsHeterogeneousList = jit_compiler::isHeterogeneousList(NDRanges);

  TargetInfo TargetInfo = ConfigHelper::get<option::JITTargetInfo>();
  BinaryFormat TargetFormat = TargetInfo.getFormat();
  DeviceArchitecture TargetArch = TargetInfo.getArch();

  if (!isTargetFormatSupported(TargetFormat)) {
    return FusionResult(
        "Fusion output target format not supported by this build");
  }

  auto &JITCtx = JITContext::getInstance();
  bool CachingEnabled = ConfigHelper::get<option::JITEnableCaching>();
  CacheKeyT CacheKey{TargetArch,
                     KernelsToFuse,
                     Identities.to<std::vector>(),
                     BarriersFlags,
                     Internalization.to<std::vector>(),
                     Constants.to<std::vector>(),
                     IsHeterogeneousList
                         ? std::optional<std::vector<NDRange>>{NDRanges}
                         : std::optional<std::vector<NDRange>>{std::nullopt}};
  if (CachingEnabled) {
    std::optional<SYCLKernelInfo> CachedKernel = JITCtx.getCacheEntry(CacheKey);
    if (CachedKernel) {
      helper::printDebugMessage("Re-using cached JIT kernel");
      if (!IsHeterogeneousList) {
        // If the cache query didn't include the ranges, update the fused range
        // before returning the kernel info to the runtime.
        CachedKernel->NDR = combineNDRanges(NDRanges);
      }
      return FusionResult{*CachedKernel, /*Cached*/ true};
    }
    helper::printDebugMessage(
        "Compiling new kernel, no suitable cached kernel found");
  }

  SYCLModuleInfo ModuleInfo;
  // Copy the kernel information for the input kernels to the module
  // information. We could remove the copy, if we removed the const from the
  // input interface, so it depends on the guarantees we want to give to
  // callers.
  ModuleInfo.kernels().insert(ModuleInfo.kernels().end(),
                              KernelInformation.begin(),
                              KernelInformation.end());
  // Load all input kernels from their respective SPIR-V modules into a single
  // LLVM IR module.
  llvm::Expected<std::unique_ptr<llvm::Module>> ModOrError =
      translation::KernelTranslator::loadKernels(*JITCtx.getLLVMContext(),
                                                 ModuleInfo.kernels());
  if (auto Error = ModOrError.takeError()) {
    return errorToFusionResult(std::move(Error), "SPIR-V translation failed");
  }
  std::unique_ptr<llvm::Module> LLVMMod = std::move(*ModOrError);

  // Add information about the kernel that should be fused as metadata into the
  // LLVM module.
  FusedFunction FusedKernel{FusedKernelName,
                            KernelsToFuse,
                            Identities.to<llvm::ArrayRef>(),
                            Internalization.to<llvm::ArrayRef>(),
                            Constants.to<llvm::ArrayRef>(),
                            NDRanges};
  FusedFunctionList FusedKernelList;
  FusedKernelList.push_back(FusedKernel);
  llvm::Expected<std::unique_ptr<llvm::Module>> NewModOrError =
      helper::FusionHelper::addFusedKernel(LLVMMod.get(), FusedKernelList);
  if (auto Error = NewModOrError.takeError()) {
    return errorToFusionResult(std::move(Error),
                               "Insertion of fused kernel stub failed");
  }
  std::unique_ptr<llvm::Module> NewMod = std::move(*NewModOrError);

  // Invoke the actual fusion via LLVM pass manager.
  std::unique_ptr<SYCLModuleInfo> NewModInfo =
      fusion::FusionPipeline::runFusionPasses(*NewMod, ModuleInfo,
                                              BarriersFlags);

  if (!NewMod->getFunction(FusedKernelName)) {
    return FusionResult{"Kernel fusion failed"};
  }

  // Get the updated kernel info for the fused kernel and add the information to
  // the existing KernelInfo.
  if (!NewModInfo->hasKernelFor(FusedKernelName)) {
    return FusionResult{"No KernelInfo for fused kernel"};
  }

  SYCLKernelInfo &FusedKernelInfo = *NewModInfo->getKernelFor(FusedKernelName);

  if (auto Error = translation::KernelTranslator::translateKernel(
          FusedKernelInfo, *NewMod, JITCtx, TargetFormat)) {
    return errorToFusionResult(std::move(Error),
                               "Translation to output format failed");
  }

  FusedKernelInfo.NDR = FusedKernel.FusedNDRange;

  if (CachingEnabled) {
    JITCtx.addCacheEntry(CacheKey, FusedKernelInfo);
  }

  return FusionResult{FusedKernelInfo};
}

void KernelFusion::resetConfiguration() { ConfigHelper::reset(); }

void KernelFusion::set(OptionPtrBase *Option) {
  ConfigHelper::getConfig().set(Option);
}
