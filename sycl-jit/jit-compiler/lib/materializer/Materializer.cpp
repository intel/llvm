//==-------------------------- Materializer.cpp ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Materializer.h"
#include "Kernel.h"
#include "Options.h"
#include "helper/ConfigHelper.h"
#include "helper/ErrorHelper.h"
#include "materializer/MaterializerPipeline.h"
#include "translation/KernelTranslation.h"

using namespace jit_compiler;

extern "C" SCM_EXPORT_SYMBOL JITResult materializeSpecConstants(
    const char *KernelName, const SYCLKernelBinaryInfo &BinaryInfo,
    View<unsigned char> SpecConstBlob) {
  auto &JITCtx = JITContext::getInstance();

  TargetInfo TargetInfo = ConfigHelper::get<option::JITTargetInfo>();
  BinaryFormat TargetFormat = TargetInfo.getFormat();
  if (TargetFormat != BinaryFormat::PTX &&
      TargetFormat != BinaryFormat::AMDGCN) {
    return JITResult("Output target format not supported by this build. "
                     "Available targets are: PTX or AMDGCN.");
  }

  std::vector<SYCLKernelBinaryInfo> BinaryInfos{BinaryInfo};
  // Load all input kernels from their respective modules into a single
  // LLVM IR module.
  llvm::LLVMContext Ctx;
  llvm::Expected<std::unique_ptr<llvm::Module>> ModOrError =
      translation::KernelTranslator::loadKernels(Ctx, BinaryInfos);
  if (auto Error = ModOrError.takeError()) {
    return errorTo<JITResult>(std::move(Error), "Failed to load kernels");
  }
  std::unique_ptr<llvm::Module> NewMod = std::move(*ModOrError);
  if (!MaterializerPipeline::runMaterializerPasses(
          *NewMod, SpecConstBlob.to<llvm::ArrayRef>()) ||
      !NewMod->getFunction(KernelName)) {
    return JITResult{"Materializer passes should not fail"};
  }

  auto BinInfoOrErr = translation::KernelTranslator::translateKernel(
      KernelName, *NewMod, JITCtx, TargetFormat);
  if (!BinInfoOrErr) {
    return errorTo<JITResult>(BinInfoOrErr.takeError(),
                              "Translation to output format failed");
  }

  return JITResult{*BinInfoOrErr};
}

extern "C" SCM_EXPORT_SYMBOL void resetJITConfiguration() {
  ConfigHelper::reset();
}

extern "C" SCM_EXPORT_SYMBOL void addToJITConfiguration(OptionStorage &&Opt) {
  ConfigHelper::getConfig().set(std::move(Opt));
}
