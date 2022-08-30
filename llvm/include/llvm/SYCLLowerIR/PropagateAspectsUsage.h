//===---- PropagateAspectsUsage.h - PropagateAspectsUsage Pass ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass propagates metadata corresponding to usage of optional device
// features.
//
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_SYCLPROPAGATE_ASPECTS_USAGE_H
#define LLVM_SYCLPROPAGATE_ASPECTS_USAGE_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class PropagateAspectsUsagePass
    : public PassInfoMixin<PropagateAspectsUsagePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

} // namespace llvm

#endif // LLVM_SYCLPROPAGATE_ASPECTS_USAGE_H
