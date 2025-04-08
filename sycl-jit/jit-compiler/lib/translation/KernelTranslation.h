//==- KernelTranslation - Translate SYCL kernels between different formats -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "Kernel.h"
#include "helper/JITContext.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include <vector>

namespace jit_compiler {
namespace translation {

class KernelTranslator {

public:
  static llvm::Expected<std::unique_ptr<llvm::Module>>
  loadKernels(llvm::LLVMContext &LLVMCtx,
              const std::vector<JITBinaryInfo> &BinaryInfos);

  static llvm::Expected<JITBinaryInfo> translateKernel(const char *KernelName,
                                                       llvm::Module &Mod,
                                                       JITContext &JITCtx,
                                                       BinaryFormat Format);

  static llvm::Expected<RTCDevImgBinaryInfo>
  translateDevImgToSPIRV(llvm::Module &Mod, JITContext &JITCtx);

private:
  ///
  /// Pair of address and size to represent a binary blob.
  using BinaryBlob = std::pair<BinaryAddress, size_t>;

  static llvm::Expected<std::unique_ptr<llvm::Module>>
  loadLLVMKernel(llvm::LLVMContext &LLVMCtx, const JITBinaryInfo &BinaryInfo);

  static llvm::Expected<std::unique_ptr<llvm::Module>>
  loadSPIRVKernel(llvm::LLVMContext &LLVMCtx, const JITBinaryInfo &BinaryInfo);

  static llvm::Expected<JITBinary *> translateToSPIRV(llvm::Module &Mod,
                                                      JITContext &JITCtx);

  static llvm::Expected<JITBinary *>
  translateToPTX(const char *KernelName, llvm::Module &Mod, JITContext &JITCtx);

  static llvm::Expected<JITBinary *> translateToAMDGCN(const char *KernelName,
                                                       llvm::Module &Mod,
                                                       JITContext &JITCtx);
};
} // namespace translation
} // namespace jit_compiler
