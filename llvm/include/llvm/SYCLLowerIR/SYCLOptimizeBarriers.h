//==- SYCLOptimizeBarriers.h - SYCLOptimizeBarriers Pass -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass cleans up ControlBarrier and MemoryBarrier calls.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_SYCL_OPTIMIZE_BARRIERS_H
#define LLVM_SYCL_OPTIMIZE_BARRIERS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class SYCLOptimizeBarriersPass
    : public PassInfoMixin<SYCLOptimizeBarriersPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &);

  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_SYCL_OPTIMIZE_BARRIERS_H
