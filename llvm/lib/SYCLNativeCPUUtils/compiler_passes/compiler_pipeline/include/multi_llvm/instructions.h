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

#ifndef MULTI_LLVM_MULTI_INSTRUCTIONS_H_INCLUDED
#define MULTI_LLVM_MULTI_INSTRUCTIONS_H_INCLUDED

#include <llvm/IR/Instructions.h>
#include <multi_llvm/llvm_version.h>

namespace multi_llvm {

namespace detail {

template <typename Base = llvm::AtomicRMWInst, typename = void>
struct AtomicRMWInst : Base {};

#if LLVM_VERSION_LESS(20, 0)
template <typename Base>
struct AtomicRMWInst<
    Base, std::enable_if_t<Base::LAST_BINOP - Base::FIRST_BINOP == 16>>
    : llvm::AtomicRMWInst {
  static constexpr BinOp USubCond = static_cast<BinOp>(BAD_BINOP + 1);
  static constexpr BinOp USubSat = static_cast<BinOp>(BAD_BINOP + 2);
  static constexpr BinOp FMaximum = static_cast<BinOp>(BAD_BINOP + 3);
  static constexpr BinOp FMinimum = static_cast<BinOp>(BAD_BINOP + 4);
};
#endif

// #if LLVM_VERSION_LESS(21, 0)
// This is enabled for now on LLVM 21 as well to allow building against older
// LLVM 21 snapshots.
template <typename Base>
struct AtomicRMWInst<
    Base, std::enable_if_t<Base::LAST_BINOP - Base::FIRST_BINOP == 18>>
    : llvm::AtomicRMWInst {
  static constexpr BinOp FMaximum = static_cast<BinOp>(BAD_BINOP + 1);
  static constexpr BinOp FMinimum = static_cast<BinOp>(BAD_BINOP + 2);
};
// #endif

}  // namespace detail

struct AtomicRMWInst : detail::AtomicRMWInst<> {};

}  // namespace multi_llvm

#endif  // MULTI_LLVM_MULTI_INSTRUCTIONS_H_INCLUDED
