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

#ifndef MULTI_LLVM_DIBUILDER_H_INCLUDED
#define MULTI_LLVM_DIBUILDER_H_INCLUDED

#include <llvm/IR/DIBuilder.h>
#include <multi_llvm/llvm_version.h>

#include <type_traits>

namespace multi_llvm {
// TODO In order to enable the use of OCK in DPC++ which currently uses the
// older DIBuilder interface, we do not yet condition this on LLVM version, we
// dynamically detect which version of DIBuilder we have. This should be updated
// after DPC++'s next pulldown to drop the use of DIBuilderWrapperNeeded and
// base it entirely on LLVM major version.
#if LLVM_VERSION_GREATER_EQUAL(20, 0) && 0
using DIBuilder = llvm::DIBuilder;
#else
template <typename DIBuilder>
struct DIBuilderWrapper : DIBuilder {
  using DIBuilder::DIBuilder;

#if LLVM_VERSION_GREATER_EQUAL(19, 0)
  llvm::BasicBlock *getBasicBlock(llvm::InsertPosition InsertPt) {
    return InsertPt.getBasicBlock();
  }
#else
  llvm::BasicBlock *getBasicBlock(llvm::BasicBlock::iterator InsertPt) {
    // Cannot handle sentinels.
    return InsertPt->getParent();
  }
#endif

  auto insertDeclare(llvm::Value *Storage, llvm::DILocalVariable *VarInfo,
                     llvm::DIExpression *Expr, const llvm::DILocation *DL,
                     llvm::BasicBlock *InsertAtEnd) {
    return DIBuilder::insertDeclare(Storage, VarInfo, Expr, DL, InsertAtEnd);
  }

  auto insertDeclare(llvm::Value *Storage, llvm::DILocalVariable *VarInfo,
                     llvm::DIExpression *Expr, const llvm::DILocation *DL,
                     llvm::BasicBlock::iterator InsertPt) {
    auto *InsertBB = getBasicBlock(InsertPt);
    if (InsertPt == InsertBB->end()) {
      return DIBuilder::insertDeclare(Storage, VarInfo, Expr, DL, InsertBB);
    } else {
      return DIBuilder::insertDeclare(Storage, VarInfo, Expr, DL, &*InsertPt);
    }
  }

  auto insertDbgValueIntrinsic(llvm::Value *Val, llvm::DILocalVariable *VarInfo,
                               llvm::DIExpression *Expr,
                               const llvm::DILocation *DL,
                               llvm::BasicBlock *InsertAtEnd) {
    return DIBuilder::insertDbgValueIntrinsic(Val, VarInfo, Expr, DL,
                                              InsertAtEnd);
  }

  auto insertDbgValueIntrinsic(llvm::Value *Val, llvm::DILocalVariable *VarInfo,
                               llvm::DIExpression *Expr,
                               const llvm::DILocation *DL,
                               llvm::BasicBlock::iterator InsertPt) {
    auto *InsertBB = getBasicBlock(InsertPt);
    if (InsertPt == InsertBB->end()) {
      return DIBuilder::insertDbgValueIntrinsic(Val, VarInfo, Expr, DL,
                                                InsertBB);
    } else {
      return DIBuilder::insertDbgValueIntrinsic(Val, VarInfo, Expr, DL,
                                                &*InsertPt);
    }
  }
};

template <typename DIBuilder, typename = void>
static constexpr bool DIBuilderWrapperNeeded = true;

template <typename DIBuilder>
static constexpr bool DIBuilderWrapperNeeded<
    DIBuilder, std::void_t<decltype(std::declval<DIBuilder &>().insertLabel(
                   std::declval<llvm::DILabel *>(),
                   std::declval<const llvm::DILocation *>(),
                   std::declval<llvm::BasicBlock::iterator>()))>> = false;

template <typename DIBuilder>
using DIBuilderMaybeWrapped =
    std::conditional_t<DIBuilderWrapperNeeded<DIBuilder>,
                       DIBuilderWrapper<DIBuilder>, DIBuilder>;

using DIBuilder = DIBuilderMaybeWrapped<llvm::DIBuilder>;
#endif
}  // namespace multi_llvm

#endif  // MULTI_LLVM_DIBUILDER_H_INCLUDED
