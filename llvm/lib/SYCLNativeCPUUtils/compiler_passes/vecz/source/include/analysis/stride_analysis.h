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
/// @brief Stride analysis.

#ifndef VECZ_ANALYSIS_STRIDE_ANALYSIS_H_INCLUDED
#define VECZ_ANALYSIS_STRIDE_ANALYSIS_H_INCLUDED

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Analysis/AssumptionCache.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/PassManager.h>

#include "offset_info.h"

namespace llvm {
class Function;
class Value;
}  // namespace llvm

namespace vecz {

struct UniformValueResult;

/// @brief Holds the result of Stride Analysis for a given function.
class StrideAnalysisResult {
 public:
  /// @brief The function being analyzed
  llvm::Function &F;
  /// @brief The Uniform Value Result to use during analysis
  UniformValueResult &UVR;
  /// @brief AssumptionCache for computing live bits of uniform values
  llvm::AssumptionCache &AC;

  StrideAnalysisResult(llvm::Function &f, UniformValueResult &uvr,
                       llvm::AssumptionCache &AC);

  /// @brief generate stride `ConstantInt`s or `Instruction`s for all analyzed
  /// values.
  void manifestAll(llvm::IRBuilder<> &B);

  /// @brief gets a pointer to the info struct for this value's analysis.
  OffsetInfo *getInfo(llvm::Value *V) {
    const auto find = analyzed.find(V);
    return (find != analyzed.end()) ? &find->second : nullptr;
  }

  /// @brief gets a pointer to the info struct for this value's analysis.
  const OffsetInfo *getInfo(llvm::Value *V) const {
    const auto find = analyzed.find(V);
    return (find != analyzed.end()) ? &find->second : nullptr;
  }

  /// @brief construct the offset info for the given value.
  OffsetInfo &analyze(llvm::Value *V);

  /// @brief build the strides as `Instructions` or `ConstantInts`.
  /// Strides may be needed as `llvm::Values` by transform passes, but we are
  /// not allowed to construct them during an analysis pass. However, note that
  /// information about manifested stride `Value`s will survive until the
  /// analysis is invalidated.
  const OffsetInfo &manifest(llvm::IRBuilder<> &B, llvm::Value *V) {
    const auto find = analyzed.find(V);
    assert(find != analyzed.end() &&
           "Trying to manifest unanalyzed OffsetInfo");
    return find->second.manifest(B, *this);
  }

  /// @brief gets the manifested memory stride for this value, if present.
  ///
  /// @param[in] B IRBuilder for creating new instructions/values
  /// @param[in] Ptr the pointer to calculate the stride for
  /// @param[in] EleTy the type that the pointer points to
  /// @returns the stride of the memory operation, in number of elements
  llvm::Value *buildMemoryStride(llvm::IRBuilder<> &B, llvm::Value *Ptr,
                                 llvm::Type *EleTy) const;

 private:
  /// @brief A map of values onto OffsetInfos that were already analyzed.
  llvm::DenseMap<llvm::Value *, OffsetInfo> analyzed;
};

/// @brief Analysis that determines whether pointer operands of memory
/// operations have a linear dependence on the work item ID.
class StrideAnalysis : public llvm::AnalysisInfoMixin<StrideAnalysis> {
  friend AnalysisInfoMixin<StrideAnalysis>;

 public:
  /// @brief Create a new analysis object.
  StrideAnalysis() {}

  using Result = StrideAnalysisResult;

  /// @brief Run the Stride Analysis
  ///
  /// @param[in] F Function to analyze.
  /// @param[in] AM FunctionAnalysisManager providing analyses.
  ///
  /// @return Analysis result for the function.
  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &AM);

  /// @brief Return the name of the pass.
  static llvm::StringRef name() { return "Stride analysis"; }

 private:
  /// @brief Unique identifier for the pass.
  static llvm::AnalysisKey Key;
};

}  // namespace vecz

#endif  // VECZ_ANALYSIS_STRIDE_ANALYSIS_H_INCLUDED
