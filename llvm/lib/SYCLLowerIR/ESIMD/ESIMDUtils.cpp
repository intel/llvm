//===------------ ESIMDUtils.cpp - ESIMD utility functions ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Utility functions for processing ESIMD code.
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/ESIMD/ESIMDUtils.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Casting.h"

namespace llvm {
namespace esimd {

bool isESIMD(const Function &F) {
  return F.getMetadata(ESIMD_MARKER_MD) != nullptr;
}

bool isKernel(const Function &F) {
  return (F.getCallingConv() == CallingConv::SPIR_KERNEL);
}

bool isESIMDKernel(const Function &F) { return isKernel(F) && isESIMD(F); }

bool isCast(const Value *V) {
  int Opc = Operator::getOpcode(V);
  return (Opc == Instruction::BitCast) || (Opc == Instruction::AddrSpaceCast);
}

bool isZeroGEP(const Value *V) {
  const auto *GEPI = dyn_cast<GetElementPtrInst>(V);
  return GEPI && GEPI->hasAllZeroIndices();
}

const Value *stripCasts(const Value *V) {
  if (!V->getType()->isPtrOrPtrVectorTy())
    return V;
  // Even though we don't look through PHI nodes, we could be called on an
  // instruction in an unreachable block, which may be on a cycle.
  SmallPtrSet<const Value *, 4> Visited;
  Visited.insert(V);

  do {
    if (isCast(V)) {
      V = cast<Operator>(V)->getOperand(0);
    }
    assert(V->getType()->isPtrOrPtrVectorTy() && "Unexpected operand type!");
  } while (Visited.insert(V).second);
  return V;
}

Value *stripCasts(Value *V) {
  return const_cast<Value *>(stripCasts(const_cast<const Value *>(V)));
}

const Value *stripCastsAndZeroGEPs(const Value *V) {
  if (!V->getType()->isPtrOrPtrVectorTy())
    return V;
  // Even though we don't look through PHI nodes, we could be called on an
  // instruction in an unreachable block, which may be on a cycle.
  SmallPtrSet<const Value *, 4> Visited;
  Visited.insert(V);

  do {
    if (isCast(V)) {
      V = cast<Operator>(V)->getOperand(0);
    } else if (isZeroGEP(V)) {
      V = cast<GetElementPtrInst>(V)->getOperand(0);
    }
    assert(V->getType()->isPtrOrPtrVectorTy() && "Unexpected operand type!");
  } while (Visited.insert(V).second);
  return V;
}

Value *stripCastsAndZeroGEPs(Value *V) {
  return const_cast<Value *>(
      stripCastsAndZeroGEPs(const_cast<const Value *>(V)));
}

void collectUsesLookThroughCasts(const Value *V,
                                 SmallPtrSetImpl<const Use *> &Uses) {
  for (const Use &U : V->uses()) {
    Value *VV = U.getUser();

    if (isCast(VV)) {
      collectUsesLookThroughCasts(VV, Uses);
    } else {
      Uses.insert(&U);
    }
  }
}

void collectUsesLookThroughCastsAndZeroGEPs(
    const Value *V, SmallPtrSetImpl<const Use *> &Uses) {
  assert(V->getType()->isPtrOrPtrVectorTy() && "pointer type expected");

  for (const Use &U : V->uses()) {
    Value *VV = U.getUser();

    if (isCast(VV) || isZeroGEP(VV)) {
      collectUsesLookThroughCastsAndZeroGEPs(VV, Uses);
    } else {
      Uses.insert(&U);
    }
  }
}

Type *getVectorTyOrNull(StructType *STy) {
  Type *Res = nullptr;
  while (STy && (STy->getStructNumElements() == 1)) {
    Res = STy->getStructElementType(0);
    STy = dyn_cast<StructType>(Res);
  }
  if (!Res || !Res->isVectorTy())
    return nullptr;
  return Res;
}

} // namespace esimd
} // namespace llvm
