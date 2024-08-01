//==- KernelTranslation - Translate SYCL kernels between different formats -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef SYCL_FUSION_JIT_COMPILER_TRANSLATION_KERNELTRANSLATION_H
#define SYCL_FUSION_JIT_COMPILER_TRANSLATION_KERNELTRANSLATION_H

#include "Kernel.h"
#include "fusion/JITContext.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include <vector>

namespace jit_compiler {
namespace translation {

class KernelTranslator {

public:
  static llvm::Expected<std::unique_ptr<llvm::Module>>
  loadKernels(llvm::LLVMContext &LLVMCtx, std::vector<SYCLKernelInfo> &Kernels);

  static llvm::Error translateKernel(SYCLKernelInfo &Kernel, llvm::Module &Mod,
                                     JITContext &JITCtx, BinaryFormat Format);

private:
  ///
  /// Pair of address and size to represent a binary blob.
  using BinaryBlob = std::pair<BinaryAddress, size_t>;

  static llvm::Expected<std::unique_ptr<llvm::Module>>
  loadLLVMKernel(llvm::LLVMContext &LLVMCtx, SYCLKernelInfo &Kernel);

  static llvm::Expected<std::unique_ptr<llvm::Module>>
  loadSPIRVKernel(llvm::LLVMContext &LLVMCtx, SYCLKernelInfo &Kernel);

  static llvm::Expected<KernelBinary *> translateToSPIRV(llvm::Module &Mod,
                                                         JITContext &JITCtx);

  static llvm::Expected<KernelBinary *>
  translateToPTX(SYCLKernelInfo &Kernel, llvm::Module &Mod, JITContext &JITCtx);

  static llvm::Expected<KernelBinary *>
  translateToAMDGCN(SYCLKernelInfo &KernelInfo, llvm::Module &Mod,
                    JITContext &JITCtx);
};
} // namespace translation
} // namespace jit_compiler

#endif // SYCL_FUSION_JIT_COMPILER_TRANSLATION_KERNELTRANSLATION_H
