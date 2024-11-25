//===--- SYCLJointMatrixTransform.h - SYCLJointMatrixTransformPass --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A transformation pass which mutates Joint Matrix builtin calls to make them
// conformant with SPIR-V friendly LLVM IR specification.
//
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_SYCL_JOINT_MATRIX_TRANSFORM_H
#define LLVM_SYCL_JOINT_MATRIX_TRANSFORM_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class SYCLJointMatrixTransformPass
    : public PassInfoMixin<SYCLJointMatrixTransformPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);

  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_SYCL_JOINT_MATRIX_TRANSFORM_H
