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

template <typename T = llvm::AtomicRMWInst::BinOp, typename = void>
struct BinOpHelper;

// TODO Make this entirely version-based once we no longer have to account for
// older LLVM 21 snapshots that use the LLVM 20 definition of
// llvm::AtomicRMWInst::BinOp.
#define LLVM 21
#include <multi_llvm/instructions.inc>
#undef LLVM
#define LLVM 20
#include <multi_llvm/instructions.inc>
#undef LLVM
#define LLVM 19
#include <multi_llvm/instructions.inc>
#undef LLVM

}  // namespace detail

static std::optional<llvm::AtomicRMWInst::BinOp> consume_binop_with_underscore(
    llvm::StringRef &String) {
  return multi_llvm::detail::BinOpHelper<>::consume_front_with_underscore(
      String);
}

static llvm::StringRef to_string(llvm::AtomicRMWInst::BinOp BinOp) {
  return multi_llvm::detail::BinOpHelper<>::to_string(BinOp);
}

}  // namespace multi_llvm

#endif  // MULTI_LLVM_MULTI_INSTRUCTIONS_H_INCLUDED
