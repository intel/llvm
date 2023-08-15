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
#include <multi_llvm/llvm_version.h>

namespace multi_llvm {
inline bool isOpaqueOrPointeeTypeMatches(llvm::PointerType *PTy, llvm::Type *) {
  (void)PTy;
#if LLVM_VERSION_LESS(17, 0)
  assert(PTy->isOpaque() && "No support for typed pointers in LLVM 15+");
#endif
  return true;
}

inline llvm::Type *getPtrElementType(llvm::PointerType *PTy) {
  (void)PTy;
#if LLVM_VERSION_LESS(17, 0)
  assert(PTy->isOpaque() && "No support for typed pointers in LLVM 15+");
#endif
  return nullptr;
}

};  // namespace multi_llvm

#endif  // MULTI_LLVM_OPAQUE_POINTERS_H_INCLUDED
