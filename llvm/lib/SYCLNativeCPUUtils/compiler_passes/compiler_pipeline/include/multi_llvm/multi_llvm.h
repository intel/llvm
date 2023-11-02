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
#ifndef MULTI_LLVM_MULTI_LLVM_H_INCLUDED
#define MULTI_LLVM_MULTI_LLVM_H_INCLUDED

#include <llvm/Analysis/IVDescriptors.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <multi_llvm/llvm_version.h>
#include <multi_llvm/triple.h>

namespace multi_llvm {

inline llvm::DILocation *getDILocation(unsigned Line, unsigned Column,
                                       llvm::MDNode *Scope,
                                       llvm::MDNode *InlinedAt = nullptr) {
  // If no scope is available, this is an unknown location.
  if (!Scope) return llvm::DebugLoc();
  return llvm::DILocation::get(Scope->getContext(), Line, Column, Scope,
                               InlinedAt, /*ImplicitCode*/ false);
}

/// @brief Create a binary operation corresponding to the given
/// `llvm::RecurKind` with the two provided arguments. It may not
/// necessarily return one of LLVM's in-built `BinaryOperator`s, or even one
/// operation: integer min/max operations may defer to multiple instructions or
/// intrinsics depending on the LLVM version.
///
/// @param[in] B the IRBuilder to build new instructions
/// @param[in] lhs the left-hand value for the operation
/// @param[in] rhs the right-hand value for the operation
/// @param[in] kind the kind of operation to create
/// @param[out] The binary operation.
inline llvm::Value *createBinOpForRecurKind(llvm::IRBuilder<> &B,
                                            llvm::Value *lhs, llvm::Value *rhs,
                                            llvm::RecurKind kind) {
  switch (kind) {
    default:
      break;
    case llvm::RecurKind::None:
      return nullptr;
    case llvm::RecurKind::Add:
      return B.CreateAdd(lhs, rhs);
    case llvm::RecurKind::Mul:
      return B.CreateMul(lhs, rhs);
    case llvm::RecurKind::Or:
      return B.CreateOr(lhs, rhs);
    case llvm::RecurKind::And:
      return B.CreateAnd(lhs, rhs);
    case llvm::RecurKind::Xor:
      return B.CreateXor(lhs, rhs);
    case llvm::RecurKind::FAdd:
      return B.CreateFAdd(lhs, rhs);
    case llvm::RecurKind::FMul:
      return B.CreateFMul(lhs, rhs);
  }
  assert((kind == llvm::RecurKind::FMin || kind == llvm::RecurKind::FMax ||
          kind == llvm::RecurKind::SMin || kind == llvm::RecurKind::SMax ||
          kind == llvm::RecurKind::UMin || kind == llvm::RecurKind::UMax) &&
         "Unexpected min/max kind");
  if (kind == llvm::RecurKind::FMin || kind == llvm::RecurKind::FMax) {
    return B.CreateBinaryIntrinsic(kind == llvm::RecurKind::FMin
                                       ? llvm::Intrinsic::minnum
                                       : llvm::Intrinsic::maxnum,
                                   lhs, rhs);
  }
  bool isMin = kind == llvm::RecurKind::SMin || kind == llvm::RecurKind::UMin;
  bool isSigned =
      kind == llvm::RecurKind::SMin || kind == llvm::RecurKind::SMax;
  llvm::Intrinsic::ID intrOpc =
      isMin ? (isSigned ? llvm::Intrinsic::smin : llvm::Intrinsic::umin)
            : (isSigned ? llvm::Intrinsic::smax : llvm::Intrinsic::umax);
  return B.CreateBinaryIntrinsic(intrOpc, lhs, rhs);
}
}  // namespace multi_llvm

#endif  // MULTI_LLVM_MULTI_LLVM_H_INCLUDED
