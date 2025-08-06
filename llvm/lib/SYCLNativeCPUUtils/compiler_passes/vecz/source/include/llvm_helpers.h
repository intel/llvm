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

/// @file
///
/// @brief LLVM helper methods.

#ifndef VECZ_LLVM_HELPERS_H_INCLUDED
#define VECZ_LLVM_HELPERS_H_INCLUDED

#include <llvm/ADT/ArrayRef.h>
#include <llvm/IR/IRBuilder.h>
#include <multi_llvm/llvm_version.h>
#include <multi_llvm/vector_type_helper.h>

namespace vecz {

/// @brief Determine if the value has vector type, and return it.
///
/// @param[in] V Value to analyze.
///
/// @return Vector type of V or null.
llvm::FixedVectorType *getVectorType(llvm::Value *V);

/// @brief Get the default value for a type.
///
/// @param[in] T Type to get default value of.
/// @param[in] V Default value to use for numeric type
///
/// @return Default value, which will be undef for non-numeric types
llvm::Value *getDefaultValue(llvm::Type *T, uint64_t V = 0UL);

/// @brief Get the shuffle mask as sequence of integers.
///
/// @param[in] Shuffle Instruction
///
/// @return Array of integers representing the Shuffle mask
llvm::ArrayRef<int> getShuffleVecMask(llvm::ShuffleVectorInst *Shuffle);
}  // namespace vecz

#endif  // VECZ_LLVM_HELPERS_H_INCLUDED
