// Copyright (C) Codeplay Software Limited
//
// Licensed under the Apache License, Version 2.0 (the "License") with LLVM
// Exceptions; you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/uxlfoundation/oneapi-construction-kit/blob/main/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm_helpers.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Module.h>
#include <multi_llvm/multi_llvm.h>

#include "debugging.h"
#include "memory_operations.h"

using namespace llvm;

/// @brief Determine if the value has vector type, and return it.
///
/// @param[in] V Value to analyze.
///
/// @return Vector type of V or null.
FixedVectorType *vecz::getVectorType(Value *V) {
  if (StoreInst *Store = dyn_cast<StoreInst>(V)) {
    auto *VO = Store->getValueOperand();
    assert(VO && "Could not get value operand");
    return dyn_cast<FixedVectorType>(VO->getType());
  } else if (CallInst *Call = dyn_cast<CallInst>(V)) {
    if (auto MaskedOp = MemOp::get(Call, MemOpAccessKind::Masked)) {
      if (MaskedOp->isMaskedMemOp() && MaskedOp->isStore()) {
        return dyn_cast<FixedVectorType>(MaskedOp->getDataType());
      }
    }
  }
  return dyn_cast<FixedVectorType>(V->getType());
}

/// @brief Get the default value for a type.
///
/// @param[in] T Type to get default value of.
/// @param[in] V Default value to use for numeric type
///
/// @return Default value, which will be poison for non-numeric types
Value *vecz::getDefaultValue(Type *T, uint64_t V) {
  if (T->isIntegerTy()) {
    return ConstantInt::get(T, V);
  }

  if (T->isFloatTy() || T->isDoubleTy()) {
    return ConstantFP::get(T, V);
  }

  return PoisonValue::get(T);
}

/// @brief Get the shuffle mask as sequence of integers.
///
/// @param[in] Shuffle Instruction
///
/// @return Array of integers representing the Shuffle mask
ArrayRef<int> vecz::getShuffleVecMask(ShuffleVectorInst *Shuffle) {
  return Shuffle->getShuffleMask();
}
