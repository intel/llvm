//===- SYCLRemangleLibspirv.h - Remangle libspirv builtins for SYCL -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Remangles __spirv_* builtin functions in libspirv to provide mangled
// variants for both OpenCL C and SYCL type representations, allowing
// SYCL device code to link libspirv.

#ifndef LLVM_SYCLLOWERIR_SYCLREMANGLELIBSPIRV_H
#define LLVM_SYCLLOWERIR_SYCLREMANGLELIBSPIRV_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class SYCLRemangleLibspirvPass
    : public PassInfoMixin<SYCLRemangleLibspirvPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

} // namespace llvm

#endif // LLVM_SYCLLOWERIR_SYCLREMANGLELIBSPIRV_H
