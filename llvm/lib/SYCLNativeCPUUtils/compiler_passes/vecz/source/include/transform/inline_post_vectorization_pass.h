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

/// @file
///
/// @brief Replace calls to certain builtins with an inline implementation after
/// vectorization.

#ifndef VECZ_TRANSFORM_INLINE_POST_VECTORIZATION_PASS_H_INCLUDED
#define VECZ_TRANSFORM_INLINE_POST_VECTORIZATION_PASS_H_INCLUDED

#include <llvm/IR/PassManager.h>

namespace vecz {

/// @brief This pass replaces calls to builtins that require special attention
/// after vectorization.
class InlinePostVectorizationPass
    : public llvm::PassInfoMixin<InlinePostVectorizationPass> {
public:
  /// @brief Create a new pass object.
  InlinePostVectorizationPass() {}

  /// @brief The entry point to the pass.
  /// @param[in,out] F Function to optimize.
  /// @param[in,out] AM FunctionAnalysisManager providing analyses.
  /// @returns Whether or not the pass changed anything.
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &AM);
  /// @brief Retrieve the pass's name.
  /// @return pointer to text description.
  static llvm::StringRef name() { return "Inline Post Vectorization pass"; }
};
} // namespace vecz

#endif // VECZ_TRANSFORM_INLINE_POST_VECTORIZATION_PASS_H_INCLUDED
