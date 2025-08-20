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

#ifndef MULTI_LLVM_LIFETIME_HELPER_H_INCLUDED
#define MULTI_LLVM_LIFETIME_HELPER_H_INCLUDED

#include <llvm/IR/IRBuilder.h>
#include <multi_llvm/llvm_version.h>

#include <type_traits>

namespace multi_llvm {

template <class T, class T2 = void> constexpr bool LifeTimeHasSizeArgT = false;

// Check for the old method where we have a size argument (ConstantInt)
template <typename IRBuilder>
static constexpr bool LifeTimeHasSizeArgT<
    IRBuilder,
    std::void_t<decltype(std::declval<IRBuilder &>().CreateLifetimeStart(
        std::declval<llvm::Value *>(), std::declval<llvm::ConstantInt *>()))>> =
    true;

static constexpr bool LifeTimeHasSizeArg() {
  return LifeTimeHasSizeArgT<llvm::IRBuilderBase>;
}
}; // namespace multi_llvm

#endif