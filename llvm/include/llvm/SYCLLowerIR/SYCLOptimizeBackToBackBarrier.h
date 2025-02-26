//==- SYCLOptimizeBackToBackBarrier.h - SYCLOptimizeBackToBackBarrier Pass -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass cleans up back-to-back ControlBarrier calls.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_SYCL_OPTIMIZE_BACK_TO_BACK_BARRIER_H
#define LLVM_SYCL_OPTIMIZE_BACK_TO_BACK_BARRIER_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class SYCLOptimizeBackToBackBarrierPass
    : public PassInfoMixin<SYCLOptimizeBackToBackBarrierPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);

  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_SYCL_OPTIMIZE_BACK_TO_BACK_BARRIER_H
