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

#ifndef MULTI_LLVM_IR_LIFETIME_H_INCLUDED
#define MULTI_LLVM_IR_LIFETIME_H_INCLUDED

#include <llvm/IR/IRBuilder.h>
#include <multi_llvm/llvm_version.h>

namespace multi_llvm {

#if LLVM_VERSION_LESS(22, 0) || 1
template <typename IRBuilder = llvm::IRBuilder<>>
auto CreateLifetimeStart(IRBuilder &B, llvm::Value *Ptr,
                                         llvm::ConstantInt *Size)
    -> decltype(B.CreateLifetimeStart(Ptr, Size)) {
  return B.CreateLifetimeStart(Ptr, Size);
}

template <typename IRBuilder = llvm::IRBuilder<>>
auto CreateLifetimeEnd(IRBuilder &B, llvm::Value *Ptr,
                                         llvm::ConstantInt *Size)
    -> decltype(B.CreateLifetimeEnd(Ptr, Size)) {
  return B.CreateLifetimeEnd(Ptr, Size);
}
#endif


template <typename IRBuilder = llvm::IRBuilder<>>
auto CreateLifetimeStart(IRBuilder &B, llvm::Value *Ptr,
                                         llvm::ConstantInt *Size)
    -> decltype(B.CreateLifetimeStart(Ptr)) {
  return B.CreateLifetimeStart(Ptr);
}

template <typename IRBuilder = llvm::IRBuilder<>>
auto CreateLifetimeEnd(IRBuilder &B, llvm::Value *Ptr,
                                         llvm::ConstantInt *Size)
    -> decltype(B.CreateLifetimeEnd(Ptr)) {
  return B.CreateLifetimeEnd(Ptr);
}

}  // namespace multi_llvm

#endif  // MULTI_LLVM_TARGET_TARGETINFO_H_INCLUDED

