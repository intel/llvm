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

#include "analysis/instantiation_analysis.h"

#include <compiler/utils/builtin_info.h>
#include <multi_llvm/vector_type_helper.h>

#include "analysis/uniform_value_analysis.h"
#include "debugging.h"
#include "memory_operations.h"
#include "vectorization_context.h"

#define DEBUG_TYPE "vecz-instantiation"

using namespace vecz;
using namespace llvm;

namespace {
bool analyzeType(Type *Ty) {
  return !Ty->isVoidTy() && !Ty->isVectorTy() &&
         !FixedVectorType::isValidElementType(Ty);
}

bool analyzeMemOp(MemOp &Op) {
  assert(Op.getPointerType()->isPointerTy() && "MemOp inconsistency");
  return analyzeType(Op.getDataType());
}

bool analyzeCall(const VectorizationContext &Ctx, CallInst *CI) {
  Function *Callee = CI->getCalledFunction();
  VECZ_FAIL_IF(!Callee);

  // Handle internal builtins.
  if (Ctx.isInternalBuiltin(Callee)) {
    if (auto Op = MemOp::get(CI)) {
      return analyzeMemOp(*Op);
    }
    return false;
  }

  // Handle function containing pointers as parameter.
  if (any_of(Callee->args(),
             [](const Argument &A) { return A.getType()->isPointerTy(); })) {
    return true;
  }

  // Handle masked function calls
  if (Ctx.isMaskedFunction(Callee)) {
    return true;
  }

  const auto Props = Ctx.builtins().analyzeBuiltin(*Callee).properties;

  // Intrinsics without side-effects can be safely instantiated.
  if (Callee->isIntrinsic() &&
      (Props & compiler::utils::eBuiltinPropertyNoSideEffects)) {
    // If the intrinsic has a vector equivalent, then we can use it directly
    // instead.
    if (Props & compiler::utils::eBuiltinPropertyVectorEquivalent) {
      return analyzeType(CI->getType());
    }
    return true;
  }

  // Functions returning void must have side-effects.
  // We cannot vectorize them and instead we need to instantiate them.
  const bool HasSideEffects =
      Callee->getReturnType()->isVoidTy() ||
      (Props & compiler::utils::eBuiltinPropertySideEffects);
  if (HasSideEffects &&
      (Props & compiler::utils::eBuiltinPropertySupportsInstantiation)) {
    return true;
  }

  return analyzeType(CI->getType());
}

bool analyzeAlloca(const VectorizationContext &Ctx, AllocaInst *alloca) {
  // Possibly, we could packetize by creating a wider array, but for now let's
  // just let instantiation deal with it.
  if (alloca->isArrayAllocation()) {
    return true;
  }

  // We can create an array of anything, however, we need to be careful of
  // alignment. In the case the alloca has a specific alignment requirement, we
  // have to be sure it divides the type allocation size, otherwise only the
  // first vector element would necessarily be correctly aligned.
  auto *const dataTy = alloca->getAllocatedType();
  const uint64_t memSize = Ctx.dataLayout()->getTypeAllocSize(dataTy);
  const uint64_t align = alloca->getAlign().value();
  return (align != 0 && (memSize % align) != 0);
}
}  // namespace

namespace vecz {
bool needsInstantiation(const VectorizationContext &Ctx, Instruction &I) {
  if (CallInst *CI = dyn_cast<CallInst>(&I)) {
    return analyzeCall(Ctx, CI);
  } else if (LoadInst *Load = dyn_cast<LoadInst>(&I)) {
    if (auto Op = MemOp::get(Load)) {
      return analyzeMemOp(*Op);
    }
    // If it's not a MemOp, assume we don't need to instantiate.
    return false;
  } else if (StoreInst *Store = dyn_cast<StoreInst>(&I)) {
    if (auto Op = MemOp::get(Store)) {
      return analyzeMemOp(*Op);
    }
    // If it's not a MemOp, assume we don't need to instantiate.
    return false;
  } else if (AllocaInst *Alloca = dyn_cast<AllocaInst>(&I)) {
    return analyzeAlloca(Ctx, Alloca);
  } else if (isa<AtomicRMWInst>(&I) || isa<AtomicCmpXchgInst>(&I)) {
    return true;
  } else {
    return analyzeType(I.getType());
  }
}
}  // namespace vecz
