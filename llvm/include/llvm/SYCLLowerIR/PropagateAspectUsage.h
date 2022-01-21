//===---- PropagateAspectUsage.h - AspectUsagePropagation Pass ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_SYCLPROPAGATE_ASPECT_USAGE_H
#define LLVM_SYCLPROPAGATE_ASPECT_USAGE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

#include <vector>

namespace llvm {

// Desc
class PropagateAspectUsagePass
    : public PassInfoMixin<PropagateAspectUsagePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

ModulePass *createPropagateAspectUsagePass();
void initializeSYCLPropagateAspectUsageLegacyPassPass(PassRegistry &);

} // namespace llvm

#endif // LLVM_SYCLPROPAGATE_ASPECT_USAGE_H
