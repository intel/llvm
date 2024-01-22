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
/// @brief Vectorizable Function analysis.

#ifndef VECZ_ANALYSIS_VECTORIZABLE_FUNCTION_ANALYSIS_H_INCLUDED
#define VECZ_ANALYSIS_VECTORIZABLE_FUNCTION_ANALYSIS_H_INCLUDED

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/PassManager.h>

namespace vecz {

/// @brief Determines whether vectorization of a function is possible.
class VectorizableFunctionAnalysis
    : public llvm::AnalysisInfoMixin<VectorizableFunctionAnalysis> {
  friend AnalysisInfoMixin<VectorizableFunctionAnalysis>;

 public:
  /// @brief Create a new instance of the pass.
  VectorizableFunctionAnalysis() = default;

  /// @brief Type of result produced by the analysis.
  struct Result {
    /// @brief Whether the function can be vectorized.
    bool canVectorize = false;

    /// @brief If the function can not be vectorized, the value (if any) that
    /// is the cause of the problem.
    const llvm::Value *failedAt = nullptr;

    /// @brief Handle invalidation events from the new pass manager.
    ///
    /// @return false, as this analysis can never be invalidated.
    bool invalidate(llvm::Function &, const llvm::PreservedAnalyses &,
                    llvm::FunctionAnalysisManager::Invalidator &) {
      return false;
    }
  };

  /// @brief Determine whether vectorization of a function is possible.
  /// @param[in] F Function to analyze.
  /// @return VectorizationUnit corresponding to this function
  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &);

  /// @brief Return the name of the pass.
  static llvm::StringRef name() { return "Vectorizable Function analysis"; }

 private:
  /// @brief Unique pass identifier.
  static llvm::AnalysisKey Key;
};

}  // namespace vecz

#endif  // VECZ_ANALYSIS_VECTORIZABLE_FUNCTION_ANALYSIS_H_INCLUDED
