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

#include <compiler/utils/attributes.h>
#include <compiler/utils/builtin_info.h>
#include <compiler/utils/prepare_barriers_pass.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <multi_llvm/multi_llvm.h>

#include <functional>

using namespace llvm;

#define DEBUG_TYPE "ca-barriers"

PreservedAnalyses compiler::utils::PrepareBarriersPass::run(
    Module &M, ModuleAnalysisManager &AM) {
  SmallPtrSet<Function *, 4> Kernels;
  auto &BI = AM.getResult<BuiltinInfoAnalysis>(M);
  for (auto &F : M.functions()) {
    if (isKernelEntryPt(F)) {
      Kernels.insert(&F);
    }
  }

  SmallPtrSet<Function *, 4> FuncsWithBarriers;

  for (Function &F : M) {
    const auto B = BI.analyzeBuiltin(F);
    // If the function does not have a barrier id.
    if (!BI.isMuxBuiltinWithBarrierID(B.ID)) {
      continue;
    }

    for (User *U : F.users()) {
      if (auto *const CI = dyn_cast<CallInst>(U)) {
        auto *const Callee = CI->getFunction();

        // If it's one of our kernels don't inline it, and definitely don't
        // delete it either. No need to inline already dead functions, either!
        if (!Callee->isDefTriviallyDead() && Kernels.count(Callee) == 0) {
          FuncsWithBarriers.insert(Callee);
        }
      }
    }
  }

  bool Changed = false;

  // Walk the users of the barrier.
  while (!FuncsWithBarriers.empty()) {
    auto *F = *FuncsWithBarriers.begin();
    FuncsWithBarriers.erase(F);

    // Make a copy of the users of the function to be inlined since
    // InlineFunction modifies the state of ci/F which affects
    // the range being iterated over, resulting in use-after-free.
    const SmallVector<User *, 8> Users{F->user_begin(), F->user_end()};

    // Check the users of the function the call instruction inhabits.
    for (User *U : Users) {
      // If the call instruction's function does not any users.
      if (!isa<CallInst>(U)) {
        continue;
      }

      auto *const InfoF = cast<CallInst>(U)->getFunction();
      InlineFunctionInfo IFI;
      auto InlineResult =
          InlineFunction(*cast<CallInst>(U), IFI, /*MergeAttributes*/ false,
                         /*CalleeAAR*/ nullptr, /*InsertLifetime*/ true,
                         /*ForwardVarArgsTo*/ nullptr);
      if (InlineResult.isSuccess()) {
        Changed = true;

        // The function we inlined into now contains a barrier, so add it
        // to the set.
        if (!InfoF->isDefTriviallyDead() && Kernels.count(InfoF) == 0) {
          FuncsWithBarriers.insert(InfoF);
        }
      } else {
        LLVM_DEBUG(dbgs() << "Could not inline: " << *U << '\n';);
      }
    }

    // Delete the now-dead inlined function
    if (F->isDefTriviallyDead()) {
      F->eraseFromParent();
    }
  }

  // Assign all barriers a unique ID
  unsigned ID = 0U;
  auto &Ctx = M.getContext();
  auto *const I32Ty = IntegerType::get(Ctx, 32);
  for (auto *F : Kernels) {
    for (BasicBlock &BB : *F) {
      for (Instruction &I : BB) {
        // Check call instructions for barrier.
        if (auto *const CI = dyn_cast<CallInst>(&I)) {
          Function *Callee = CI->getCalledFunction();
          if (Callee &&
              BI.isMuxBuiltinWithBarrierID(BI.analyzeBuiltin(*Callee).ID)) {
            CI->setOperand(0, ConstantInt::get(I32Ty, ID++));
          }
        }
      }
    }
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
