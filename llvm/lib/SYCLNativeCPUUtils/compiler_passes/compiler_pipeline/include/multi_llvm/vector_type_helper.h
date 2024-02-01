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
#ifndef MULTI_LLVM_VECTOR_TYPE_HELPER_H_INCLUDED
#define MULTI_LLVM_VECTOR_TYPE_HELPER_H_INCLUDED

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Type.h>
#include <llvm/Support/TypeSize.h>

namespace multi_llvm {

// The functions defined below are common functions to allow us to generically
// get VectorType information from a base Type class, due to either deprecation
// or removal of these in LLVM 11 (result of scalable/fixed vectors separation)

inline llvm::Type *getVectorElementType(llvm::Type *ty) {
  assert(llvm::isa<llvm::VectorType>(ty) && "Not a vector type");
  return llvm::cast<llvm::VectorType>(ty)->getElementType();
}
inline llvm::Type *getVectorElementType(const llvm::Type *ty) {
  assert(llvm::isa<llvm::VectorType>(ty) && "Not a vector type");
  return llvm::cast<llvm::VectorType>(ty)->getElementType();
}

inline uint64_t getVectorNumElements(llvm::Type *ty) {
  assert(ty->getTypeID() == llvm::Type::FixedVectorTyID &&
         "Not a fixed vector type");
  return llvm::cast<llvm::FixedVectorType>(ty)
      ->getElementCount()
      .getFixedValue();
}
inline uint64_t getVectorNumElements(const llvm::Type *ty) {
  assert(ty->getTypeID() == llvm::Type::FixedVectorTyID &&
         "Not a fixed vector type");
  return llvm::cast<llvm::FixedVectorType>(ty)
      ->getElementCount()
      .getFixedValue();
}

inline llvm::ElementCount getVectorElementCount(llvm::Type *ty) {
  return llvm::cast<llvm::VectorType>(ty)->getElementCount();
}
inline llvm::ElementCount getVectorElementCount(const llvm::Type *ty) {
  return llvm::cast<llvm::VectorType>(ty)->getElementCount();
}

inline unsigned getVectorKnownMinNumElements(llvm::Type *ty) {
  return getVectorElementCount(ty).getKnownMinValue();
}

inline unsigned getVectorKnownMinNumElements(const llvm::Type *ty) {
  return getVectorElementCount(ty).getKnownMinValue();
}
}  // namespace multi_llvm

#endif  // MULTI_LLVM_VECTOR_TYPE_HELPER_H_INCLUDED
