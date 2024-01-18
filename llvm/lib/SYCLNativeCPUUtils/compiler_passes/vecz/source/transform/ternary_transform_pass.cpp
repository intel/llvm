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

#include "transform/ternary_transform_pass.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>

#include "analysis/stride_analysis.h"
#include "analysis/uniform_value_analysis.h"
#include "analysis/vectorization_unit_analysis.h"
#include "debugging.h"
#include "ir_cleanup.h"
#include "memory_operations.h"

using namespace llvm;
using namespace vecz;

namespace {
/// @brief Determine whether the select can and should be transformed. This is
/// the case when there is at most one GEP to it and followed by Load/Store
/// memory op and there are no other users to GEP.
/// Additionally, we reject various cases where the tranform would not result
/// in better code.
bool shouldTransform(SelectInst *Select, const StrideAnalysisResult &SAR) {
  // The transform only applies to pointer selects.
  if (!Select->getType()->isPointerTy()) {
    return false;
  }

  // There is absolutely no need to transform a uniform select.
  if (!SAR.UVR.isVarying(Select)) {
    return false;
  }

  {
    // If the select itself is a strided pointer, we don't gain anything by
    // transforming it into a pair of masked memops.
    const auto *info = SAR.getInfo(Select);
    if (info && info->hasStride()) {
      return false;
    }
  }

  // Validate Select operands
  Value *VecTrue = Select->getOperand(1);
  Value *VecFalse = Select->getOperand(2);

  assert(VecTrue && VecFalse);

  // If both pointers are uniform, it's worth doing the transform, since we get
  // only scalar Mask Varying memops, instead of vector memops.
  if (SAR.UVR.isVarying(VecTrue) || SAR.UVR.isVarying(VecFalse)) {
    // Both pointers must be either strided or uniform (i.e. not divergent).
    const auto *infoT = SAR.getInfo(VecTrue);
    const auto *infoF = SAR.getInfo(VecFalse);
    if (!infoT || !infoF || infoT->mayDiverge() || infoF->mayDiverge()) {
      return false;
    }
  }

  // Validate Select users
  GetElementPtrInst *TheGEP = nullptr;
  SmallVector<Instruction *, 8> SelectsUsers;
  for (User *U : Select->users()) {
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(U)) {
      // There can be at most one GEP
      if (TheGEP) {
        return false;
      }
      TheGEP = GEP;
      SelectsUsers.push_back(GEP);
    } else {
      return false;
    }
  }

  // Validate GEP users
  while (!SelectsUsers.empty()) {
    VECZ_FAIL_IF(!isa<GetElementPtrInst>(SelectsUsers.back()));
    GetElementPtrInst *GEP =
        cast<GetElementPtrInst>(SelectsUsers.pop_back_val());

    // Validate the GEP indices
    for (Value *idx : GEP->indices()) {
      const auto *info = SAR.getInfo(idx);
      if (!info || info->mayDiverge()) {
        return false;
      }
    }
    // We only transform selects used by GEPs who are exclusively used by
    // scalar loads and stores. Performing this transform on vectors was
    // historically banned due to internal limitations, but these days we
    // *should* be able to. It's just that we don't know whether it's
    // beneficial: see CA-4337.
    for (User *U : GEP->users()) {
      if (auto *const LI = dyn_cast<LoadInst>(U)) {
        if (LI->getType()->isVectorTy()) {
          return false;
        }
      } else if (auto *const SI = dyn_cast<StoreInst>(U)) {
        if (SI->getValueOperand()->getType()->isVectorTy()) {
          return false;
        }
      } else {
        return false;
      }
    }
  }
  return true;
}

/// @brief Try to transform the select, remove GEP & memory op and
/// replace with transformed GEP and masked memory op.
void Transform(SelectInst *Select, VectorizationContext &Ctx) {
  SmallVector<Instruction *, 8> ToDelete;

  auto transformSelect = [&](GetElementPtrInst *GEP, Instruction *Memop,
                             Value *StoredValue, ArrayRef<Value *> Indices) {
    // Non-obviously, we need to insert the new instructions at the GEP. The GEP
    // is a user of the select, so we can guarantee that the GEP dominates the
    // select. To ensure that the new instructions added also dominate the
    // indices of the GEP, we need to insert at the GEP.
    IRBuilder<> B(GEP);

    Value *Condition = Select->getCondition();
    Value *InvCondition = B.CreateXor(Condition, 1);
    Value *True = Select->getTrueValue();
    Value *False = Select->getFalseValue();
    Value *GepTrue = B.CreateGEP(GEP->getSourceElementType(), True, Indices);
    Value *GepFalse = B.CreateGEP(GEP->getSourceElementType(), False, Indices);
    auto MaskedOp = MemOp::get(Memop);
    assert(MaskedOp);
    const MemOpDesc Mem = MaskedOp->getDesc();

    // We should have filtered out all vector memory operations earlier.
    assert(!Mem.getDataType()->isVectorTy());

    auto Alignment = Mem.getAlignment();
    if (isa<LoadInst>(Memop)) {
      // Transform load
      Value *LoadTrue =
          createMaskedLoad(Ctx, Mem.getDataType(), GepTrue, Condition,
                           /*VL*/ nullptr, Alignment, "", Memop);
      Value *LoadFalse =
          createMaskedLoad(Ctx, Mem.getDataType(), GepFalse, InvCondition,
                           /*VL*/ nullptr, Alignment, "", Memop);
      B.SetInsertPoint(Memop);
      Value *LoadResult = B.CreateSelect(Condition, LoadTrue, LoadFalse);

      // Replace all uses with new value
      Memop->replaceAllUsesWith(LoadResult);
    } else if (isa<StoreInst>(Memop)) {
      // Transform store
      createMaskedStore(Ctx, StoredValue, GepTrue, Condition, /*VL*/ nullptr,
                        Alignment, "", Memop);
      createMaskedStore(Ctx, StoredValue, GepFalse, InvCondition,
                        /*VL*/ nullptr, Alignment, "", Memop);
    }
  };

  for (User *U : Select->users()) {
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(U)) {
      ToDelete.push_back(GEP);

      const SmallVector<Value *, 2> Indices(GEP->idx_begin(), GEP->idx_end());

      for (User *G : GEP->users()) {
        if (LoadInst *Load = dyn_cast<LoadInst>(G)) {
          ToDelete.push_back(Load);
          transformSelect(GEP, Load, nullptr, Indices);
        } else if (StoreInst *Store = dyn_cast<StoreInst>(G)) {
          ToDelete.push_back(Store);
          transformSelect(GEP, Store, Store->getValueOperand(), Indices);
        }
      }
    }
  }

  // Clean up instructions bottom-up (users first).
  while (!ToDelete.empty()) {
    Instruction *I = ToDelete.pop_back_val();
    if (I->use_empty()) {
      IRCleanup::deleteInstructionNow(I);
    }
  }

  IRCleanup::deleteInstructionNow(Select);
}
}  // namespace

PreservedAnalyses TernaryTransformPass::run(llvm::Function &F,
                                            llvm::FunctionAnalysisManager &AM) {
  const auto &SAR = AM.getResult<StrideAnalysis>(F);

  // Find selects that can be transformed
  SmallVector<SelectInst *, 4> Selects;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (SelectInst *Select = dyn_cast<SelectInst>(&I)) {
        if (shouldTransform(Select, SAR)) {
          Selects.push_back(Select);
        }
      }
    }
  }

  if (Selects.empty()) {
    return PreservedAnalyses::all();
  }

  auto &Ctx = AM.getResult<VectorizationContextAnalysis>(F).getContext();

  // Transform them.
  for (SelectInst *Select : Selects) {
    Transform(Select, Ctx);
  }

  return PreservedAnalyses::none();
}
