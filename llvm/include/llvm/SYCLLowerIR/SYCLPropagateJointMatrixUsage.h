//===- SYCLPropagateJointMatrixUsage.cpp - SYCLPropagateJointMatrixUsage Pass
//-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass propagates optional kernel features metadata through a module call graph
// for sycl_ext_oneapi_matrix extension
//
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_SYCL_PROPAGATE_JOINT_MATRIX_USAGE_H
#define LLVM_SYCL_PROPAGATE_JOINT_MATRIX_USAGE_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class SYCLPropagateJointMatrixUsagePass
    : public PassInfoMixin<SYCLPropagateJointMatrixUsagePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

} // namespace llvm

#endif // LLVM_SYCL_PROPAGATE_JOINT_MATRIX_USAGE_H
