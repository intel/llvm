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

/// @brief Analysis of control flow.

#ifndef VECZ_ANALYSIS_CONTROL_FLOW_ANALYSIS_H_INCLUDED
#define VECZ_ANALYSIS_CONTROL_FLOW_ANALYSIS_H_INCLUDED

#include <llvm/IR/PassManager.h>

namespace llvm {
class BasicBlock;
}  // namespace llvm

namespace vecz {

/// @brief Holds the results and state for CFG analysis.
struct CFGResult {
  /// @brief true if analysis failed, e.g. CFG conversion cannot be done.
  bool failed = false;
  /// @brief true if CFG conversion is needed to vectorize the function.
  bool convNeeded = false;
  /// @brief Single basic block that exits the function.
  llvm::BasicBlock *exitBB = nullptr;

  /// @brief Create new analysis results for the given function.
  CFGResult() = default;

  /// @brief Deleted copy constructor.
  CFGResult(const CFGResult &) = delete;

  /// @brief Move constructor.
  ///
  /// @param[in,out] Res Existing results to move.
  CFGResult(CFGResult &&Res) = default;

  /// @brief Access the failed flag.
  /// @return true if analysis failed.
  bool getFailed() const { return failed; }

  /// @brief Access the failed flag.
  /// @param[in] newVal New value for the flag.
  void setFailed(bool newVal) { failed = newVal; }

  /// @brief Determine whether CFG conversion is needed for the function or not.
  bool isConversionNeeded() const { return convNeeded; }
  /// @brief Set whether CFG conversion is needed for the function or not.
  /// @param[in] newVal Whether conversion is needed or not.
  void setConversionNeeded(bool newVal) { convNeeded = newVal; }

  /// @brief Single block in the function that returns to the caller or null.
  llvm::BasicBlock *getExitBlock() const { return exitBB; }
};

/// @brief Analysis that determines whether a function can have divergent
/// control flow and so whether CFG conversion is needed or not.
class CFGAnalysis : public llvm::AnalysisInfoMixin<CFGAnalysis> {
 public:
  /// @brief Create a new CFG analysis object.
  CFGAnalysis() = default;

  /// @brief Type of the analaysis result.
  using Result = CFGResult;

  /// @brief Perform CFG analysis on the function to determine whether control
  /// flow conversion is required and possible or not.
  ///
  /// @param[in,out] F Function to analyze.
  /// @param[in,out] AM FunctionAnalysisManager providing analyses
  ///
  /// @return CFG analysis result.
  CFGResult run(llvm::Function &F, llvm::FunctionAnalysisManager &AM);

  /// @brief Analysis name.
  static llvm::StringRef name() { return "CFG analysis"; }

 private:
  friend llvm::AnalysisInfoMixin<CFGAnalysis>;
  /// @brief Unique identifier for the analysis.
  static llvm::AnalysisKey Key;
};

}  // namespace vecz

#endif  // VECZ_ANALYSIS_CONTROL_FLOW_ANALYSIS_H_INCLUDED
