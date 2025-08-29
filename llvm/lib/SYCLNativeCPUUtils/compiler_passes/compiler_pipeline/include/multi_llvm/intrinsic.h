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

#ifndef MULTI_LLVM_MULTI_INTRINSIC_H_INCLUDED
#define MULTI_LLVM_MULTI_INTRINSIC_H_INCLUDED

#include <llvm/IR/Intrinsics.h>
#include <multi_llvm/llvm_version.h>

namespace multi_llvm {

// Drop getAttributes workaround when LLVM 21 is minimum version
namespace detail {
template <typename... T>
auto getAttributes(T... args)
    -> decltype(llvm::Intrinsic::getAttributes(args...)) {
  return llvm::Intrinsic::getAttributes(args...);
}
template <typename... T>
auto getAttributes(T... args, llvm::FunctionType *)
    -> decltype(llvm::Intrinsic::getAttributes(args...)) {
  return llvm::Intrinsic::getAttributes(args...);
}
}  // namespace detail

namespace Intrinsic {
static inline auto getAttributes(llvm::LLVMContext &C, llvm::Intrinsic::ID ID,
                                 llvm::FunctionType *FT) {
  return detail::getAttributes<llvm::LLVMContext &, llvm::Intrinsic::ID>(C, ID,
                                                                         FT);
}
}  // namespace Intrinsic

}  // namespace multi_llvm

#endif  // MULTI_LLVM_MULTI_INTRINSIC_H_INCLUDED
