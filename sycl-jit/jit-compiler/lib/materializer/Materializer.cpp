//==-------------------------- Materializer.cpp ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Materializer.h"
#include "helper/ConfigHelper.h"
#include "helper/ErrorHelper.h"
#include "materializer/MaterializerPipeline.h"
#include "translation/Translation.h"
#include <llvm/ADT/StringRef.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Support/MemoryBuffer.h>

using namespace jit_compiler;

extern "C" JIT_EXPORT_SYMBOL SCMResult materializeSpecConstants(
    const char *KernelName, const JITBinaryInfo &BinaryInfo,
    View<unsigned char> SpecConstBlob) {
  auto &JITCtx = JITContext::getInstance();

  // TODO(jopperm): Why is the target format an option instead of an argument?
  BinaryFormat TargetFormat = ConfigHelper::get<option::JITTargetFormat>();
  if (TargetFormat != BinaryFormat::PTX &&
      TargetFormat != BinaryFormat::AMDGCN) {
    return SCMResult("Output target format not supported by this build. "
                     "Available targets are: PTX or AMDGCN.");
  }

  llvm::LLVMContext Ctx;
  llvm::StringRef RawData(
      reinterpret_cast<const char *>(BinaryInfo.BinaryStart),
      BinaryInfo.BinarySize);
  llvm::Expected<std::unique_ptr<llvm::Module>> ModOrError =
      llvm::parseBitcodeFile(
          llvm::MemoryBuffer::getMemBuffer(RawData, KernelName,
                                           /* RequiresNullTermnator*/ false)
              ->getMemBufferRef(),
          Ctx);
  if (auto Error = ModOrError.takeError()) {
    return errorTo<SCMResult>(std::move(Error), "Failed to load kernels");
  }
  std::unique_ptr<llvm::Module> NewMod = std::move(*ModOrError);
  if (!MaterializerPipeline::runMaterializerPasses(
          *NewMod, SpecConstBlob.to<llvm::ArrayRef>()) ||
      !NewMod->getFunction(KernelName)) {
    return SCMResult{"Materializer passes should not fail"};
  }

  auto BinInfoOrErr =
      Translator::translate(*NewMod, JITCtx, TargetFormat, KernelName);
  if (!BinInfoOrErr) {
    return errorTo<SCMResult>(BinInfoOrErr.takeError(),
                              "Translation to output format failed");
  }

  return SCMResult{*BinInfoOrErr};
}
