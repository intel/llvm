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

#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/Local.h>
#include <multi_llvm/llvm_version.h>

#include "debugging.h"
#include "transform/passes.h"

using namespace llvm;
using namespace vecz;

#define DEBUG_TYPE "vecz-mem2reg"

PreservedAnalyses BasicMem2RegPass::run(Function &F,
                                        FunctionAnalysisManager &) {
  LLVM_DEBUG(dbgs() << "\n\nVECZ MEM2REG on " << F.getName() << "\n");
  bool modified = false;
  if (F.empty()) {
    return PreservedAnalyses::all();
  }

  // Find allocas that can be promoted.
  SmallVector<AllocaInst *, 4> PromotableAllocas;
  BasicBlock &EntryBB = F.getEntryBlock();
  for (Instruction &I : EntryBB) {
    if (AllocaInst *Alloca = dyn_cast<AllocaInst>(&I)) {
      if (canPromoteAlloca(Alloca)) {
        PromotableAllocas.push_back(Alloca);
      }
    }
  }

  // Promote them.
  for (AllocaInst *Alloca : PromotableAllocas) {
    if (promoteAlloca(Alloca)) {
      LLVM_DEBUG(dbgs() << "VM2R: Promoted :" << *Alloca << "\n");
      Alloca->eraseFromParent();
      modified = true;
    }
  }

  if (!modified) {
    return PreservedAnalyses::all();
  }

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

bool BasicMem2RegPass::canPromoteAlloca(AllocaInst *Alloca) const {
  BasicBlock *ParentBB = Alloca->getParent();
  Function *F = ParentBB->getParent();
  BasicBlock &EntryBB = F->getEntryBlock();
  if (&EntryBB != ParentBB) {
    return false;
  }

  const unsigned SrcPointeeBits =
      Alloca->getAllocatedType()->getPrimitiveSizeInBits();

  if (SrcPointeeBits == 0) {
    return false;
  }

  // Validate the alloca's users.
  StoreInst *TheStore = nullptr;
  SmallPtrSet<Value *, 4> NonStoreUsers;
  for (User *U : Alloca->users()) {
    if (StoreInst *Store = dyn_cast<StoreInst>(U)) {
      // There can be at most one store.
      if (TheStore) {
        return false;
      }
      // Stores must be in the entry block.
      if (Store->getParent() != &EntryBB) {
        return false;
      }
      // Check if the store is actually storing a value *in* the alloca and not
      // using the alloca itself as the value to be stored. For example, in the
      // following IR code, the store can be used to promote p_639 but not
      // c_640:
      //
      // %c_640 = alloca %struct.S2, align 16
      // %p_639 = alloca %struct.S2*, align 8
      // store %struct.S2* %c_640, %struct.S2** %p_639, align 8
      //
      // Also, if the alloca pointer is stored in some other variable, we can
      // not promote the alloca as we need the pointer.
      if (Store->getPointerOperand() != Alloca) {
        return false;
      }
      // Everything is fine, use this store
      TheStore = Store;
    } else if (isa<LoadInst>(U)) {
      // The loaded type doesn't necessarily equal the alloca type when opaque
      // pointers are involved:
      //   %a = alloca i32
      //   %v = load i16, ptr %a
      // We can only promote the alloca if we can bitcast between the two
      // underlying types as well.
      // We could probably zero-extend or trunc if we had to? See CA-4382.
      const unsigned DstPointeeBits = U->getType()->getPrimitiveSizeInBits();
      if (!DstPointeeBits || SrcPointeeBits != DstPointeeBits) {
        return false;
      }
      NonStoreUsers.insert(U);
    } else if (BitCastInst *Cast = dyn_cast<BitCastInst>(U)) {
      // The bitcast must be from one pointer type to another.
      PointerType *SrcPtrTy = dyn_cast<PointerType>(Cast->getSrcTy());
      PointerType *DstPtrTy = dyn_cast<PointerType>(Cast->getType());
      if (!SrcPtrTy || !DstPtrTy) {
        return false;
      }
      // The cast must have one load user.
      if (!Cast->hasOneUse()) {
        return false;
      }
      User *CastUser = *Cast->user_begin();
      if (!isa<LoadInst>(CastUser)) {
        return false;
      }
      // Since this is a bitcast, we can only promote the alloca if we can
      // bitcast between the two underlying types as well.
      const unsigned DstPointeeBits =
          CastUser->getType()->getPrimitiveSizeInBits();
      if (!DstPointeeBits || SrcPointeeBits != DstPointeeBits) {
        return false;
      }
      NonStoreUsers.insert(U);
    } else {
      // Do not allow other kinds of users.
      return false;
    }
  }

  // If the alloca has no value stored into it, then there is no value to get
  // and we can't promote it.
  if (!TheStore) {
    return false;
  }

  // Stores must precede other users.
  for (Instruction &I : EntryBB) {
    if (NonStoreUsers.count(&I)) {
      return false;
    } else if (&I == TheStore) {
      break;
    }
  }

  return true;
}

bool BasicMem2RegPass::promoteAlloca(AllocaInst *Alloca) const {
  LLVM_DEBUG(dbgs() << "VM2R: NOW AT :" << *Alloca << "\n");
  // Find the value stored in the alloca.
  Value *StoredValue = nullptr;
  SmallVector<Instruction *, 8> ToDelete;
  for (User *U : Alloca->users()) {
    if (StoreInst *Store = dyn_cast<StoreInst>(U)) {
      StoredValue = Store->getValueOperand();
      ToDelete.push_back(Store);
      DIBuilder DIB(*Alloca->getModule(), /*AllowUnresolved*/ false);
#if LLVM_VERSION_GREATER_EQUAL(18, 0)
      SmallVector<DbgDeclareInst *, 1> DbgIntrinsics;
      findDbgDeclares(DbgIntrinsics, Alloca);
#elif LLVM_VERSION_GREATER_EQUAL(17, 0)
      auto DbgIntrinsics = FindDbgDeclareUses(Alloca);
#else
      auto DbgIntrinsics = FindDbgAddrUses(Alloca);
#endif
      for (auto oldDII : DbgIntrinsics) {
        ConvertDebugDeclareToDebugValue(oldDII, Store, DIB);
      }
      break;
    }
  }
  assert(StoredValue != nullptr && "Could not find value stored in alloca");

  // Replace non-store users with the stored value.
  for (User *U : Alloca->users()) {
    if (isa<StoreInst>(U)) {
      continue;
    }
    LoadInst *Load = dyn_cast<LoadInst>(U);
    Value *NewValue = StoredValue;
    BitCastInst *Cast = dyn_cast<BitCastInst>(U);
    if (Cast) {
      // We've already verified that a bitcast must have a load attached.
      Load = cast<LoadInst>(*Cast->user_begin());
      LLVM_DEBUG(dbgs() << "VM2R: Cast     :" << *Cast << "\n");
    }
    if (!Load) {
      return false;
    }
    LLVM_DEBUG(dbgs() << "VM2R: Load     :" << *Load << "\n");
    // Handle any type changes - not necessarily from the BitCastInst we've
    // checked above! We've already verified that the loaded type type and the
    // alloca size must be identical...
    assert(Load->getType()->getPrimitiveSizeInBits() ==
           Alloca->getAllocatedType()->getPrimitiveSizeInBits());
    if (Load->getType() != NewValue->getType()) {
      // ... but we haven't checked that the stored value is the right size:
      //   %a = alloca i32
      //   store i16, ptr %a
      //   %v = load i32, ptr %a
      // Note: we could do other things if the type sizes didn't match. See
      // CA-4382.
      if (Load->getType()->getPrimitiveSizeInBits() !=
          NewValue->getType()->getPrimitiveSizeInBits()) {
        return false;
      }
      NewValue = CastInst::CreateBitOrPointerCast(StoredValue, Load->getType(),
                                                  "", Load);
    }
    LLVM_DEBUG(dbgs() << "VM2R: Replaced :" << *Load << "\n");
    LLVM_DEBUG(dbgs() << "      |-> with :" << *NewValue << "\n");
    Load->replaceAllUsesWith(NewValue);
    if (Cast) {
      ToDelete.push_back(Cast);
    }
    ToDelete.push_back(Load);
  }

  // Clean up instructions bottom-up (users first).
  while (!ToDelete.empty()) {
    Instruction *I = ToDelete.pop_back_val();
    if (I->use_empty()) {
      LLVM_DEBUG(dbgs() << "VM2R: Deleted  :" << *I << "\n");
      I->eraseFromParent();
    }
  }
  return true;
}
