//=== SYCLOptimizeBackToBackBarrier.cpp - SYCL barrier optimization pass ===//
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

#include "llvm/SYCLLowerIR/SYCLOptimizeBackToBackBarrier.h"

#include "llvm/IR/IRBuilder.h"

using namespace llvm;

namespace {

static constexpr char CONTROL_BARRIER[] = "_Z22__spirv_ControlBarrieriii";
static constexpr char ITT_BARRIER[] = "__itt_offload_wg_barrier_wrapper";
static constexpr char ITT_RESUME[] = "__itt_offload_wi_resume_wrapper";

// The function removes back-to-back ControlBarrier calls in case if they
// have the same arguments. It also cleans up ITT annotations surrounding
// the barrier call.
bool processControlBarrier(Function *F) {
  BasicBlock *PrevBB = nullptr;
  llvm::SmallPtrSet<Instruction *, 8> ToErase;
  for (auto I = F->user_begin(), E = F->user_end(); I != E;) {
    User *U = *I++;
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI)
      continue;

    // New basic block - new processing.
    BasicBlock *CurrentBB = CI->getParent();
    if (CurrentBB != PrevBB) {
      PrevBB = CurrentBB;
      continue;
    }

    llvm::SmallPtrSet<Instruction *, 2> ToEraseLocalITT;
    BasicBlock::iterator It(CI);
    // Iterate over the basic block storing back-to-back barriers and their ITT
    // annotations into ToErase container.
    while (It != CurrentBB->begin()) {
      --It;
      auto *Cand = dyn_cast<CallInst>(&*It);
      if (!Cand)
        break;
      StringRef CandName = Cand->getCalledFunction()->getName();
      if (CandName == ITT_RESUME || CandName == ITT_BARRIER) {
        ToEraseLocalITT.insert(Cand);
        continue;
      } else if (CandName == CONTROL_BARRIER) {
        bool EqualOps = true;
        for (unsigned I = 0; I != CI->getNumOperands(); ++I) {
          if (CI->getOperand(I) != Cand->getOperand(I)) {
            EqualOps = false;
            break;
          }
        }
        if (EqualOps) {
          ToErase.insert(Cand);
          for (auto *ITT : ToEraseLocalITT)
            ToErase.insert(ITT);
          ToEraseLocalITT.clear();
        }
      }
    }
  }

  if (ToErase.empty())
    return false;

  for (auto *I : ToErase) {
    I->dropAllReferences();
    I->eraseFromParent();
  }

  return true;
}

} // namespace

PreservedAnalyses
SYCLOptimizeBackToBackBarrierPass::run(Module &M, ModuleAnalysisManager &MAM) {
  bool ModuleChanged = false;
  for (Function &F : M)
    if (F.isDeclaration())
      if (F.getName() == CONTROL_BARRIER)
        ModuleChanged |= processControlBarrier(&F);

  return ModuleChanged ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
