//===- type utils.cc ---------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypeUtils.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Casting.h"

namespace mlirclang {

using namespace llvm;

bool isRecursiveStruct(Type *T, Type *Meta, SmallPtrSetImpl<Type *> &seen) {
  if (seen.count(T))
    return false;
  seen.insert(T);
  if (T->isVoidTy() || T->isFPOrFPVectorTy() || T->isIntOrIntVectorTy())
    return false;
  if (T == Meta) {
    return true;
  }
  for (Type *ST : T->subtypes()) {
    if (isRecursiveStruct(ST, Meta, seen)) {
      return true;
    }
  }
  return false;
}

Type *anonymize(Type *T) {
  if (auto *PT = dyn_cast<PointerType>(T))
    return PointerType::get(anonymize(PT->getPointerElementType()),
                            PT->getAddressSpace());
  if (auto *AT = dyn_cast<ArrayType>(T))
    return ArrayType::get(anonymize(AT->getElementType()),
                          AT->getNumElements());
  if (auto *FT = dyn_cast<FunctionType>(T)) {
    SmallVector<Type *, 4> V;
    for (auto *t : FT->params())
      V.push_back(anonymize(t));
    return FunctionType::get(anonymize(FT->getReturnType()), V, FT->isVarArg());
  }
  if (auto *ST = dyn_cast<StructType>(T)) {
    if (ST->isLiteral())
      return ST;
    SmallVector<Type *, 4> V;

    for (auto *t : ST->elements()) {
      SmallPtrSet<Type *, 4> Seen;
      if (isRecursiveStruct(t, ST, Seen))
        V.push_back(t);
      else
        V.push_back(anonymize(t));
    }
    return StructType::get(ST->getContext(), V, ST->isPacked());
  }
  return T;
}

} // namespace mlirclang
