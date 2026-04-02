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

/// @file vectorization_unit_analysis.h
///
/// @brief VectorizationUnit analysis.

#ifndef VECZ_ANALYSIS_VECTORIZATION_UNIT_H_INCLUDED
#define VECZ_ANALYSIS_VECTORIZATION_UNIT_H_INCLUDED

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/PassManager.h>

#include <cassert>

#include "vectorization_context.h"
#include "vectorization_unit.h"

namespace vecz {

/// @brief Caches and returns the VectorizationUnit for a Function.
class VectorizationUnitAnalysis
    : public llvm::AnalysisInfoMixin<VectorizationUnitAnalysis> {
  friend AnalysisInfoMixin<VectorizationUnitAnalysis>;

public:
  /// @brief Create a new instance of the pass.
  VectorizationUnitAnalysis(const VectorizationContext &Ctx) : Ctx(Ctx) {}

  /// @brief Type of result produced by the analysis.
  class Result {
    VectorizationUnit *VU = nullptr;

  public:
    Result() = default;
    Result(VectorizationUnit *VU) : VU(VU) {}
    VectorizationUnit &getVU() {
      assert(hasResult());
      return *VU;
    }
    bool hasResult() { return VU; }

    /// @brief Handle invalidation events from the new pass manager.
    ///
    /// @return false, as this analysis can never be invalidated.
    bool invalidate(llvm::Function &, const llvm::PreservedAnalyses &,
                    llvm::FunctionAnalysisManager::Invalidator &) {
      return false;
    }
  };

  /// @brief Retrieve the VectorizationUnit for the requested function.
  /// @param[in] F Function to analyze.
  /// @return VectorizationUnit corresponding to this function
  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &);

  /// @brief Return the name of the pass.
  static llvm::StringRef name() { return "VectorizationUnit analysis"; }

private:
  const VectorizationContext &Ctx;
  /// @brief Unique pass identifier.
  static llvm::AnalysisKey Key;
};

/// @brief Caches and returns the VectorizationContext for a Function.
class VectorizationContextAnalysis
    : public llvm::AnalysisInfoMixin<VectorizationContextAnalysis> {
  friend AnalysisInfoMixin<VectorizationContextAnalysis>;

public:
  /// @brief Create a new instance of the pass.
  VectorizationContextAnalysis(VectorizationContext &Ctx) : Context(Ctx) {}

  /// @brief Type of result produced by the analysis.
  class Result {
    VectorizationContext &Ctx;

  public:
    Result(VectorizationContext &Ctx) : Ctx(Ctx) {}
    VectorizationContext &getContext() { return Ctx; }
    const VectorizationContext &getContext() const { return Ctx; }

    /// @brief Handle invalidation events from the new pass manager.
    ///
    /// @return false, as this analysis can never be invalidated.
    bool invalidate(llvm::Function &, const llvm::PreservedAnalyses &,
                    llvm::FunctionAnalysisManager::Invalidator &) {
      return false;
    }
  };

  /// @brief Retrieve the VectorizationContext for the requested function.
  /// @param[in] F Function to analyze.
  /// @return VectorizationContext corresponding to this function
  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &);

  /// @brief Return the name of the pass.
  static llvm::StringRef name() { return "VectorizationContext analysis"; }

private:
  VectorizationContext &Context;
  /// @brief Unique pass identifier.
  static llvm::AnalysisKey Key;
};
} // namespace vecz

#endif // VECZ_ANALYSIS_VECTORIZATION_UNIT_H_INCLUDED
