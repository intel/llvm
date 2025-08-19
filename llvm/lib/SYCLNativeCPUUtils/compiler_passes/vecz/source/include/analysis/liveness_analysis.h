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

/// @file liveness_analysis.h
///
/// @brief Live Variable Set Analysis

#ifndef VECZ_ANALYSIS_LIVENESS_ANALYSIS_H
#define VECZ_ANALYSIS_LIVENESS_ANALYSIS_H

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/PassManager.h>

namespace llvm {
class Loop;
class LoopInfo;
class Function;
class BasicBlock;
class Value;
}  // namespace llvm

namespace vecz {
class VectorizationUnit;

struct BlockLivenessInfo {
  using LiveSet = llvm::SmallVector<llvm::Value *, 16>;

  LiveSet LiveIn;
  LiveSet LiveOut;
  size_t MaxRegistersInBlock = 0;
};

class LivenessResult {
 public:
  LivenessResult(llvm::Function &F) : F(F) {}

  LivenessResult() = delete;
  LivenessResult(const LivenessResult &) = delete;
  LivenessResult(LivenessResult &&) = default;
  ~LivenessResult() = default;

  void recalculate();

  size_t getMaxLiveVirtualRegisters() const;
  const BlockLivenessInfo &getBlockInfo(const llvm::BasicBlock *) const;

 private:
  class Impl;

  llvm::Function &F;

  size_t maxNumberOfLiveValues;

  llvm::DenseMap<const llvm::BasicBlock *, BlockLivenessInfo> BlockInfos;
};

/// Analysis pass to perform liveness analysis and estimate register pressure by
/// counting the number of live virtual registers in a function.
///
/// Values in a basic block's live set are guaranteed to be in program order.
class LivenessAnalysis : public llvm::AnalysisInfoMixin<LivenessAnalysis> {
  friend llvm::AnalysisInfoMixin<LivenessAnalysis>;

 public:
  using Result = LivenessResult;

  LivenessAnalysis() = default;

  /// @brief Return the name of the pass.
  static llvm::StringRef name() { return "Liveness analysis"; }

  /// Estimate the number of registers needed by F by counting the number of
  /// live values.
  ///
  /// Assumes a reducible CFG. In OpenCL 1.2 whether or not irreducible control
  /// flow is illegal is implementation defined.
  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &);

  /// @brief Unique pass identifier.
  static llvm::AnalysisKey Key;
};

}  // namespace vecz

#endif  // VECZ_ANALYSIS_LIVENESS_ANALYSIS_H
