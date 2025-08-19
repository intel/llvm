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
/// @brief SIMD width analysis.

#ifndef VECZ_ANALYSIS_SIMD_WIDTH_ANALYSIS_H_INCLUDED
#define VECZ_ANALYSIS_SIMD_WIDTH_ANALYSIS_H_INCLUDED

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/PassManager.h>

#include "vectorization_unit.h"

namespace vecz {

class LivenessResult;

/// @brief Choose a good SIMD width for the given function.
class SimdWidthAnalysis : public llvm::AnalysisInfoMixin<SimdWidthAnalysis> {
  friend AnalysisInfoMixin<SimdWidthAnalysis>;

 public:
  /// @brief Create a new instance of the pass.
  SimdWidthAnalysis() = default;

  /// @brief Type of result produced by the analysis.
  struct Result {
    Result(unsigned value) : value(value) {}
    unsigned value;
  };

  /// @brief Run the SIMD width analysis pass on the given function.
  /// @param[in] F Function to analyze.
  /// @param[in] AM FunctionAnalysisManager providing analyses.
  /// @return Preferred SIMD vectorization factor for the function or zero.
  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &AM);

  /// @brief Return the name of the pass.
  static llvm::StringRef name() { return "SIMD width analysis"; }

 private:
  unsigned avoidSpillImpl(llvm::Function &, llvm::FunctionAnalysisManager &,
                          unsigned MinWidth = 2);

  /// @brief Vector register width from TTI, if available.
  unsigned MaxVecRegBitWidth;

  /// @brief Unique pass identifier.
  static llvm::AnalysisKey Key;
};
}  // namespace vecz

#endif  // VECZ_ANALYSIS_SIMD_WIDTH_ANALYSIS_H_INCLUDED
