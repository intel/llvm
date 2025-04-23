//===- Materializer.cpp ---------------------------------------------------===//
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

JIT_EXPORT_SYMBOL SCMResult materializeSpecConstants(
    const char *KernelName, const JITBinaryInfo &BinaryInfo,
    BinaryFormat TargetFormat, View<unsigned char> SpecConstBlob) {
  if (BinaryInfo.Format != BinaryFormat::LLVM) {
    return SCMResult("Unsupported input format.");
  }
  if (TargetFormat != BinaryFormat::PTX &&
      TargetFormat != BinaryFormat::AMDGCN) {
    return SCMResult("Output target format not supported by this build. "
                     "Available targets are: PTX or AMDGCN.");
  }

  llvm::LLVMContext Ctx;
  llvm::StringRef RawData(
      reinterpret_cast<const char *>(BinaryInfo.BinaryStart),
      BinaryInfo.BinarySize);
  std::unique_ptr<llvm::Module> Module;
  if (auto Error = llvm::parseBitcodeFile(llvm::MemoryBuffer::getMemBuffer(
                                              RawData, KernelName,
                                              /* RequiresNullTermnator*/ false)
                                              ->getMemBufferRef(),
                                          Ctx)
                       .moveInto(Module)) {
    return errorTo<SCMResult>(std::move(Error), "Failed to load module");
  }
  if (!MaterializerPipeline::runMaterializerPasses(
          *Module, SpecConstBlob.to<llvm::ArrayRef>()) ||
      !Module->getFunction(KernelName)) {
    return SCMResult{"Materializer passes should not fail"};
  }

  JITBinaryInfo TranslatedBinaryInfo;
  if (auto Error = Translator::translate(*Module, JITContext::getInstance(),
                                         TargetFormat, KernelName)
                       .moveInto(TranslatedBinaryInfo)) {
    return errorTo<SCMResult>(std::move(Error),
                              "Translation to output format failed");
  }

  return SCMResult{TranslatedBinaryInfo};
}
