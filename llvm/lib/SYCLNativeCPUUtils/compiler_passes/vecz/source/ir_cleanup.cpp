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

#include "ir_cleanup.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/Support/Debug.h>
#include <llvm/Transforms/Utils/Local.h>

#include "memory_operations.h"

#define DEBUG_TYPE "vecz"

using namespace llvm;
using namespace vecz;

namespace {

/// @brief Determine whether all users of the instructions are dead. An user is
/// dead if it has no use, if it is present in the 'to delete' list or if it is
/// a phi node whose only use keeps it alive is the 'backedge'.
///
/// @param[in] I Instruction to check for deletion.
/// @param[in] DeadList Instructions marked for deletion.
/// @param[in,out] WorkList Newly detected Instructions marked for deletion.
/// @param[in,out] Visited Instructions visited for deletion.
///
/// @return true if all users of the instructions are dead, false otherwise.
bool AreUsersDead(Instruction *I,
                  const SmallPtrSetImpl<Instruction *> &DeadList,
                  SmallPtrSetImpl<Instruction *> &WorkList,
                  SmallPtrSetImpl<Instruction *> &Visited) {
  for (User *U : I->users()) {
    // Ignore non-instructions.
    Instruction *UserI = dyn_cast<Instruction>(U);
    if (!UserI) {
      continue;
    }

    // Trivially dead users can be removed, even if we haven't explicitly marked
    // them for deletion. The DCE pass would have removed these later on anyway,
    // and by marking them for deletion here we can be more aggressive about
    // what we delete.
    if (isInstructionTriviallyDead(UserI)) {
      WorkList.insert(UserI);
    }

    // I is held by a non-dead user.
    if (!DeadList.count(UserI) && !WorkList.count(UserI)) {
      return false;
    }

    // Recurse over the user's users.
    if (!UserI->user_empty() && Visited.insert(UserI).second &&
        !AreUsersDead(UserI, DeadList, WorkList, Visited)) {
      return false;
    }
  }
  return true;
}

/// @brief Mark any invalid debug intrinsics in the DbgUsers list for
/// deletion. When an Instruction is deleted, its debug uses change to undef
/// or an empty MDNode. In this case we add it in the 'to delete' list.
///
/// @param[in] DbgUsers Debug Intrinsic Instructions.
/// @param[in,out] WorkList Newly detected Instructions marked for deletion.
///
/// @return void
void DeleteDebugInfoInstructions(
    const SmallVectorImpl<DbgVariableIntrinsic *> &DbgUsers,
    SmallPtrSetImpl<Instruction *> &WorkList) {
  for (llvm::DbgVariableIntrinsic *DII : DbgUsers) {
    Value *Op = DII->getOperand(0);
    // The first operand must be a non-null variable location argument.
    if (Op) {
      auto *MD = cast<MetadataAsValue>(Op)->getMetadata();

      // Check the variable location is not an undef.
      if (auto *V = dyn_cast<ValueAsMetadata>(MD)) {
        Value *Var = V->getValue();
        if (Var && !isa<UndefValue>(Var)) {
          continue;
        }
      }

      // Check the variable doesn't point to an empty MDNode.
      if (auto *mdNode = dyn_cast<MDNode>(MD)) {
        if (mdNode->getNumOperands() > 0) {
          continue;
        }
      }
    }

    // Mark the Debug Info Intrinsic for deletion.
    WorkList.insert(DII);
  }
}

}  // namespace

void IRCleanup::deleteInstructionLater(llvm::Instruction *I) {
  if (InstructionsToDelete.insert(I).second) {
    LLVM_DEBUG(dbgs() << "Marking for deletion: " << *I << "\n");
  }
}

void IRCleanup::deleteInstructions() {
  SmallPtrSet<Instruction *, 16> WorkList;
  SmallPtrSet<Instruction *, 16> VisitedForCycles;
  SmallVector<DbgVariableIntrinsic *, 1> DbgUsers;
  bool progress = true;
  while (progress && !InstructionsToDelete.empty()) {
    progress = false;
    for (Instruction *I : InstructionsToDelete) {
      WorkList.erase(I);
      if (I->use_empty()) {
        // Before we delete the current instruction we save its debug users, to
        // check for potential loss of debug information after the removal of I.
        findDbgUsers(DbgUsers, I);
        I->eraseFromParent();
        // After we delete the instruction, its debug uses (if any) may become
        // useless as a result of a loss of debug info. where the value of one
        // or more source variables becomes unavailable, so at this point we
        // will identify and delete those debug info instructions.
        DeleteDebugInfoInstructions(DbgUsers, WorkList);
        DbgUsers.clear();
        progress = true;
      } else if (PHINode *Phi = dyn_cast<PHINode>(I)) {
        if (AreUsersDead(Phi, InstructionsToDelete, WorkList,
                         VisitedForCycles)) {
          Phi->replaceAllUsesWith(UndefValue::get(Phi->getType()));
          Phi->eraseFromParent();
          progress = true;
        } else {
          WorkList.insert(Phi);
        }
        VisitedForCycles.clear();
      } else if (CallInst *CI = dyn_cast<CallInst>(I)) {
        // MemOps make deleting unnecessary instructions harder, because they
        // cannot be trivially dead instructions, thus breaking our recursive
        // deletion. However, if we have packetized a load or a store, we
        // definitely want to remove the scalar one, as it will be
        // reading/writing to invalid pointers. To make things simpler, here we
        // detect internal builtins that perform memory operations and erase
        // them. Since stores have no users, they will be removed earlier on and
        // we do not need to check here.
        auto Op = MemOp::get(CI);
        if (Op && Op->isLoad()) {
          // We need to replace loads with nops, as we need to have a value for
          // their users, which will be removed later on.
          I->replaceAllUsesWith(UndefValue::get(Op->getDataType()));
          I->eraseFromParent();
        } else {
          WorkList.insert(I);
        }
      } else {
        WorkList.insert(I);
      }
    }
    InstructionsToDelete = std::move(WorkList);
    WorkList.clear();
  }

  // Remove remaining instructions from the list.
  LLVM_DEBUG(for (Instruction *I
                  : InstructionsToDelete) {
    dbgs() << "vecz: could not delete " << *I << "\n";
  });
  InstructionsToDelete.clear();
}

void IRCleanup::deleteInstructionNow(Instruction *I) {
  I->replaceAllUsesWith(UndefValue::get(I->getType()));
  I->eraseFromParent();
}
