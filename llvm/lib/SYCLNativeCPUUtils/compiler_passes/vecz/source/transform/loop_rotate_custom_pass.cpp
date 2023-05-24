// Copyright (C) Codeplay Software Limited
//
// Licensed under the Apache License, Version 2.0 (the "License") with LLVM
// Exceptions; you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/codeplaysoftware/oneapi-construction-kit/blob/main/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Scalar/LoopRotation.h"
#include "transform/passes.h"

using namespace llvm;

llvm::PreservedAnalyses vecz::VeczLoopRotatePass::run(
    llvm::Loop &L, llvm::LoopAnalysisManager &LAM,
    llvm::LoopStandardAnalysisResults &AR, llvm::LPMUpdater &LU) {
  // Only process loops whose latch cannot exit the loop and its predecessors
  // cannot either.
  if (L.isLoopExiting(L.getLoopLatch())) {
    return PreservedAnalyses::all();
  }

  for (BasicBlock *pred : predecessors(L.getLoopLatch())) {
    if (L.contains(pred) && L.isLoopExiting(pred)) {
      return PreservedAnalyses::all();
    }
  }

  return LoopRotatePass().run(L, LAM, AR, LU);
}
