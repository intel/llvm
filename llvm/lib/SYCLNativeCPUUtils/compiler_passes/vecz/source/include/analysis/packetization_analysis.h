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
/// @brief Stride analysis.

#ifndef VECZ_ANALYSIS_PACKETIZATION_ANALYSIS_H_INCLUDED
#define VECZ_ANALYSIS_PACKETIZATION_ANALYSIS_H_INCLUDED

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/PassManager.h>

namespace llvm {
class Function;
class Value;
}  // namespace llvm

namespace vecz {

class StrideAnalysisResult;
struct UniformValueResult;

/// @brief Holds the result of Packetization Analysis for a given function.
class PacketizationAnalysisResult {
 public:
  /// @brief The function being analyzed
  llvm::Function &F;
  /// @brief The Stride Analysis Result to use during analysis
  StrideAnalysisResult &SAR;
  /// @brief The Uniform Value Result to use during analysis
  UniformValueResult &UVR;

  /// @brief Traverse the function, starting from the vector leaves, and mark
  /// instructions for packetization where needed. Note that the resulting set
  /// MAY not be exhaustive, since it is not always easy to predict where the
  /// packetizer might fail and fall back on instantiation, in which case
  /// pointers will need to be packetized regardless of linear stride.
  PacketizationAnalysisResult(llvm::Function &f, StrideAnalysisResult &sar);

  /// @brief Returns whether the packetization set is empty or not.
  bool isEmpty() const { return toPacketize.empty(); }

  /// @brief query whether the given value has been marked for packetization.
  ///
  /// @param[in] V the value to query
  /// @return true if the value was marked for packetization, false otherwise.
  bool needsPacketization(const llvm::Value *V) const {
    return toPacketize.count(V) != 0;
  }

 private:
  void markForPacketization(llvm::Value *V);

  /// @brief The set of instructions that need to be packetized.
  /// This equates to all non-uniform values except for values used only in
  /// address computations with constant linear strides.
  llvm::DenseSet<const llvm::Value *> toPacketize;
};

/// @brief Analysis that determines whether pointer operands of memory
/// operations have a linear dependence on the work item ID.
class PacketizationAnalysis
    : public llvm::AnalysisInfoMixin<PacketizationAnalysis> {
  friend AnalysisInfoMixin<PacketizationAnalysis>;

 public:
  /// @brief Create a new analysis object.
  PacketizationAnalysis() {}

  using Result = PacketizationAnalysisResult;

  /// @brief Run the Packetization Analysis
  ///
  /// @param[in] F Function to analyze.
  /// @param[in] AM FunctionAnalysisManager providing analyses.
  ///
  /// @return Analysis result for the function.
  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &AM);

  /// @brief Return the name of the pass.
  static llvm::StringRef name() { return "Packetization analysis"; }

 private:
  /// @brief Unique identifier for the pass.
  static llvm::AnalysisKey Key;
};

}  // namespace vecz

#endif  // VECZ_ANALYSIS_PACKETIZATION_ANALYSIS_H_INCLUDED
