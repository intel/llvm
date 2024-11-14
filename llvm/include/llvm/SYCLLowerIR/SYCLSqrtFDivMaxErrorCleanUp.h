//===-- SYCLSqrtFDivMaxErrorCleanUp.h - SYCLSqrtFDivMaxErrorCleanUp Pass --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Remove llvm.fpbuiltin.[sqrt/fdiv] intrinsics to ensure backward compatibility
// with the old drivers (that don't support SPV_INTEL_fp_max_error extension)
// in case if they are used with standart for OpenCL max-error (e.g [3.0/2.5]
// ULP and there are no other llvm.fpbuiltin.* intrinsic functions, fdiv
// instructions or @sqrt builtins/intrinsics in the module.
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_SYCL_SQRT_FDIV_MAX_ERROR_CLEAN_UP_H
#define LLVM_SYCL_SQRT_FDIV_MAX_ERROR_CLEAN_UP_H

#include "llvm/IR/PassManager.h"

namespace llvm {

// FIXME: remove this pass, it's not really needed.
class SYCLSqrtFDivMaxErrorCleanUpPass
    : public PassInfoMixin<SYCLSqrtFDivMaxErrorCleanUpPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);

  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_SYCL_SQRT_FDIV_MAX_ERROR_CLEAN_UP_H
