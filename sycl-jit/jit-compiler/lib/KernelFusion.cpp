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

static JITResult errorToFusionResult(llvm::Error &&Err,
                                     const std::string &Msg) {
  std::stringstream ErrMsg;
  ErrMsg << Msg << "\nDetailed information:\n";
  llvm::handleAllErrors(std::move(Err),
                        [&ErrMsg](const llvm::StringError &StrErr) {
                          // Cannot throw an exception here if LLVM itself is
                          // compiled without exception support.
                          ErrMsg << "\t" << StrErr.getMessage() << "\n";
                        });
  return JITResult{ErrMsg.str().c_str()};
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
#ifdef JIT_SUPPORT_PTX
    return true;
#else  // JIT_SUPPORT_PTX
    return false;
#endif // JIT_SUPPORT_PTX
  }
  case BinaryFormat::AMDGCN: {
#ifdef JIT_SUPPORT_AMDGCN
    return true;
#else  // JIT_SUPPORT_AMDGCN
    return false;
#endif // JIT_SUPPORT_AMDGCN
  }
  default:
    return false;
  }
}

extern "C" JITResult
materializeSpecConstants(const char *KernelName,
                         jit_compiler::SYCLKernelBinaryInfo &BinInfo,
                         View<unsigned char> SpecConstBlob,
                         const char *TargetCPU, const char *TargetFeatures) {
  auto &JITCtx = JITContext::getInstance();

  TargetInfo TargetInfo = ConfigHelper::get<option::JITTargetInfo>();
  BinaryFormat TargetFormat = TargetInfo.getFormat();
  if (TargetFormat != BinaryFormat::PTX &&
      TargetFormat != BinaryFormat::AMDGCN) {
    return JITResult("Output target format not supported by this build. "
                     "Available targets are: PTX or AMDGCN.");
  }

  ::jit_compiler::SYCLKernelInfo KernelInfo{
      KernelName, ::jit_compiler::SYCLArgumentDescriptor{},
      ::jit_compiler::NDRange{}, BinInfo};
  SYCLModuleInfo ModuleInfo;
  ModuleInfo.kernels().insert(ModuleInfo.kernels().end(), KernelInfo);
  // Load all input kernels from their respective modules into a single
  // LLVM IR module.
  llvm::Expected<std::unique_ptr<llvm::Module>> ModOrError =
      translation::KernelTranslator::loadKernels(*JITCtx.getLLVMContext(),
                                                 ModuleInfo.kernels());
  if (auto Error = ModOrError.takeError()) {
    return errorToFusionResult(std::move(Error), "Failed to load kernels");
  }
  std::unique_ptr<llvm::Module> NewMod = std::move(*ModOrError);
  if (!fusion::FusionPipeline::runMaterializerPasses(
          *NewMod, SpecConstBlob.to<llvm::ArrayRef>()) ||
      !NewMod->getFunction(KernelName)) {
    return JITResult{"Materializer passes should not fail"};
  }

  SYCLKernelInfo &MaterializerKernelInfo = *ModuleInfo.getKernelFor(KernelName);
  if (auto Error = translation::KernelTranslator::translateKernel(
          MaterializerKernelInfo, *NewMod, JITCtx, TargetFormat, TargetCPU,
          TargetFeatures)) {
    return errorToFusionResult(std::move(Error),
                               "Translation to output format failed");
  }

  return JITResult{MaterializerKernelInfo};
}

extern "C" JITResult fuseKernels(View<SYCLKernelInfo> KernelInformation,
                                 const char *FusedKernelName,
                                 View<ParameterIdentity> Identities,
                                 BarrierFlags BarriersFlags,
                                 View<ParameterInternalization> Internalization,
                                 View<jit_compiler::JITConstant> Constants) {

  std::vector<std::string> KernelsToFuse;
  llvm::transform(KernelInformation, std::back_inserter(KernelsToFuse),
                  [](const auto &KI) { return std::string{KI.Name.c_str()}; });

  const auto NDRanges = gatherNDRanges(KernelInformation.to<llvm::ArrayRef>());

  TargetInfo TargetInfo = ConfigHelper::get<option::JITTargetInfo>();
  BinaryFormat TargetFormat = TargetInfo.getFormat();
  DeviceArchitecture TargetArch = TargetInfo.getArch();

  llvm::Expected<jit_compiler::FusedNDRange> FusedNDR =
      jit_compiler::FusedNDRange::get(NDRanges);
  if (llvm::Error Err = FusedNDR.takeError()) {
    return errorToFusionResult(std::move(Err), "Illegal ND-range combination");
  }

  if (!isTargetFormatSupported(TargetFormat)) {
    return JITResult("Fusion output target format not supported by this build");
  }

  auto &JITCtx = JITContext::getInstance();
  bool CachingEnabled = ConfigHelper::get<option::JITEnableCaching>();
  CacheKeyT CacheKey{TargetArch,
                     KernelsToFuse,
                     Identities.to<std::vector>(),
                     BarriersFlags,
                     Internalization.to<std::vector>(),
                     Constants.to<std::vector>(),
                     FusedNDR->isHeterogeneousList()
                         ? std::optional<std::vector<NDRange>>{NDRanges}
                         : std::optional<std::vector<NDRange>>{std::nullopt}};
  if (CachingEnabled) {
    std::optional<SYCLKernelInfo> CachedKernel = JITCtx.getCacheEntry(CacheKey);
    if (CachedKernel) {
      helper::printDebugMessage("Re-using cached JIT kernel");
      if (!FusedNDR->isHeterogeneousList()) {
        // If the cache query didn't include the ranges, update the fused range
        // before returning the kernel info to the runtime.
        CachedKernel->NDR = FusedNDR->getNDR();
      }
      return JITResult{*CachedKernel, /*Cached*/ true};
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
                            *FusedNDR};
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
    return JITResult{"Kernel fusion failed"};
  }

  // Get the updated kernel info for the fused kernel and add the information to
  // the existing KernelInfo.
  if (!NewModInfo->hasKernelFor(FusedKernelName)) {
    return JITResult{"No KernelInfo for fused kernel"};
  }

  SYCLKernelInfo &FusedKernelInfo = *NewModInfo->getKernelFor(FusedKernelName);

  if (auto Error = translation::KernelTranslator::translateKernel(
          FusedKernelInfo, *NewMod, JITCtx, TargetFormat)) {
    return errorToFusionResult(std::move(Error),
                               "Translation to output format failed");
  }

  FusedKernelInfo.NDR = FusedNDR->getNDR();

  if (CachingEnabled) {
    JITCtx.addCacheEntry(CacheKey, FusedKernelInfo);
  }

  return JITResult{FusedKernelInfo};
}

extern "C" void resetJITConfiguration() { ConfigHelper::reset(); }

extern "C" void addToJITConfiguration(OptionStorage &&Opt) {
  ConfigHelper::getConfig().set(std::move(Opt));
}
