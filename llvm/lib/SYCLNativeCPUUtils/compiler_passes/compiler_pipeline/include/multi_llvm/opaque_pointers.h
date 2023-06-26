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
#ifndef MULTI_LLVM_OPAQUE_POINTERS_H_INCLUDED
#define MULTI_LLVM_OPAQUE_POINTERS_H_INCLUDED

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Type.h>

namespace multi_llvm {
inline bool isOpaquePointerTy(llvm::Type *Ty) {
  if (auto *PTy = llvm::dyn_cast<llvm::PointerType>(Ty)) {
    return PTy->isOpaque();
  }
  return false;
}

inline bool isOpaqueOrPointeeTypeMatches(llvm::PointerType *PTy, llvm::Type *) {
  (void)PTy;
  assert(PTy->isOpaque() && "No support for typed pointers in LLVM 15+");
  return true;
}

inline llvm::Type *getPtrElementType(llvm::PointerType *PTy) {
  if (PTy->isOpaque()) {
    return nullptr;
  }
  assert(false && "No support for typed pointers");
  return nullptr;
}

};  // namespace multi_llvm

#endif  // MULTI_LLVM_OPAQUE_POINTERS_H_INCLUDED
