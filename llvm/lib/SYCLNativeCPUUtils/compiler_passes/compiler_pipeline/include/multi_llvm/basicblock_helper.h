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
#ifndef MULTI_LLVM_BASICBLOCK_HELPER_H_INCLUDED
#define MULTI_LLVM_BASICBLOCK_HELPER_H_INCLUDED

#include <llvm/IR/BasicBlock.h>
#include <multi_llvm/llvm_version.h>

namespace multi_llvm {
inline void insertBefore(llvm::Instruction *const I,
                         const llvm::BasicBlock::iterator InsertPos) {
#if LLVM_VERSION_GREATER_EQUAL(18, 0)
  I->insertBefore(InsertPos);
#else
  I->insertBefore(&*InsertPos);
#endif
}

inline llvm::BasicBlock::iterator getFirstNonPHIIt(llvm::BasicBlock *const BB) {
#if LLVM_VERSION_GREATER_EQUAL(18, 0)
  return BB->getFirstNonPHIIt();
#else
  return BB->getFirstNonPHI()->getIterator();
#endif
}
}  // namespace multi_llvm

#endif  // MULTI_LLVM_BASICBLOCK_HELPER_H_INCLUDED
