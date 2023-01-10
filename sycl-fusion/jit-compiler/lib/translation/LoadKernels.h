//==-- LoadKernels.h - Load LLVM IR for SYCL kernels in different formats  -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Kernel.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include <llvm/Support/Error.h>
#include <vector>

namespace jit_compiler {
namespace translation {

class KernelLoader {

public:
  static llvm::Expected<std::unique_ptr<llvm::Module>>
  loadKernels(llvm::LLVMContext &LLVMCtx, std::vector<SYCLKernelInfo> &Kernels);

private:
  ///
  /// Pair of address and size to represent a binary blob.
  using BinaryBlob = std::pair<BinaryAddress, size_t>;

  static llvm::Expected<std::unique_ptr<llvm::Module>>
  loadLLVMKernel(llvm::LLVMContext &LLVMCtx, SYCLKernelInfo &Kernel);

  static llvm::Expected<std::unique_ptr<llvm::Module>>
  loadSPIRVKernel(llvm::LLVMContext &LLVMCtx, SYCLKernelInfo &Kernel);
};
} // namespace translation
} // namespace jit_compiler
