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
/// @brief Factory functions for some Vecz support passes

#ifndef VECZ_TRANSFORM_PASSES_H_INCLUDED
#define VECZ_TRANSFORM_PASSES_H_INCLUDED

#include <llvm/IR/PassManager.h>
#include <llvm/Transforms/Scalar/LoopPassManager.h>

namespace compiler {
namespace utils {
class BuiltinInfo;
}  // namespace utils
}  // namespace compiler

namespace vecz {
class SimplifyInfiniteLoopPass
    : public llvm::PassInfoMixin<SimplifyInfiniteLoopPass> {
 public:
  SimplifyInfiniteLoopPass() = default;

  llvm::PreservedAnalyses run(llvm::Loop &L, llvm::LoopAnalysisManager &,
                              llvm::LoopStandardAnalysisResults &,
                              llvm::LPMUpdater &);
};

/// @brief This pass replaces calls to builtins that require special attention
/// (e.g. there is no scalar or vector equivalent) with inline implementations.
class BuiltinInliningPass : public llvm::PassInfoMixin<BuiltinInliningPass> {
 public:
  /// @brief Create a new pass object.
  BuiltinInliningPass() = default;

  /// @brief The entry point to the pass.
  /// @param[in,out] M Module to optimize.
  /// @param[in,out] AM ModuleAnalysisManager providing analyses.
  /// @return Preserved analyses.
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);

  /// @brief Retrieve the pass's name.
  /// @return pointer to text description.
  static llvm::StringRef name() { return "OpenCL builtin inlining pass"; }

 private:
  /// @brief Process a call site, inlining it or marking it as needing inlining
  /// if required.
  ///
  /// @param[in] CI Call site to inspect.
  /// @param[out] NeedLLVMInline Whether the call site needs LLVM inlining.
  ///
  /// @return New return value for the call instruction.
  llvm::Value *processCallSite(llvm::CallInst *CI, bool &NeedLLVMInline);
};

/// @brief This pass tries to remove unecessary allocas that are not optimized
/// away by LLVM's Mem2Reg pass, for example in the presence of bitcasts. It is
/// however much simpler than LLVM's.
class BasicMem2RegPass : public llvm::PassInfoMixin<BasicMem2RegPass> {
 public:
  BasicMem2RegPass() {};

  /// @brief The entry point to the pass.
  /// @param[in,out] F Function to optimize.
  /// @param[in,out] AM FunctionAnalysisManager providing analyses.
  /// @return Preserved analyses.
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &AM);
  /// @brief Retrieve the pass's name.
  /// @return pointer to text description.
  static llvm::StringRef name() { return "Basic Mem2Reg Pass"; }

 private:
  /// @brief Determine whether the alloca can be promoted or not.
  ///
  /// This is the case when it is inside the entry block, there is at most one
  /// store to it and all other users are loads (possibly through bitcasts).
  /// The store must also be in the entry block and precede all loads.
  ///
  /// @param[in] Alloca Alloca instruction to analyze.
  /// @return true if the alloca can be promoted, false otherwise.
  bool canPromoteAlloca(llvm::AllocaInst *Alloca) const;
  /// @brief Try to promote the alloca, remove store users and replacing load
  /// users by the stored values. The alloca itself isn't touched.
  /// @param[in] Alloca Alloca instruction to promote.
  /// @return true if the alloca was promoted, false otherwise.
  bool promoteAlloca(llvm::AllocaInst *Alloca) const;
};

class PreLinearizePass : public llvm::PassInfoMixin<PreLinearizePass> {
 public:
  PreLinearizePass() = default;

  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &AM);

  static llvm::StringRef name() { return "Prepare for SPMD linearization"; }
};

/// @brief Wraps llvm's LoopRotatePass but retricts the range of loops on which
/// it works.
class VeczLoopRotatePass : public llvm::PassInfoMixin<VeczLoopRotatePass> {
 public:
  VeczLoopRotatePass() {}

  llvm::PreservedAnalyses run(llvm::Loop &L, llvm::LoopAnalysisManager &,
                              llvm::LoopStandardAnalysisResults &,
                              llvm::LPMUpdater &);

  static llvm::StringRef name() { return "Vecz Loop Rotation Wrapper"; };
};

class RemoveIntPtrPass : public llvm::PassInfoMixin<RemoveIntPtrPass> {
 public:
  RemoveIntPtrPass() = default;

  static llvm::StringRef name() { return "Remove IntPtr instructions"; }

  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &);
};

class SquashSmallVectorsPass
    : public llvm::PassInfoMixin<SquashSmallVectorsPass> {
 public:
  SquashSmallVectorsPass() = default;

  static llvm::StringRef name() { return "Squash Small Vectors"; }

  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &);
};

/// @brief Try to replace or remove masked memory operations that are trivially
/// not needed or can be converted to non-masked operations.
class SimplifyMaskedMemOpsPass
    : public llvm::PassInfoMixin<SimplifyMaskedMemOpsPass> {
 public:
  /// @brief Create a new pass object.
  SimplifyMaskedMemOpsPass() = default;

  /// @brief Replace masked memory operations that use 'all true' masks by
  /// regular memory operations, and remove masked operations that use 'all
  /// false' masks.
  ///
  /// @param[in] F Function to optimize.
  /// @param[in] AM FunctionAnalysisManager providing analyses.
  ///
  /// @return Preserved analyses.
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &AM);

  /// @brief Pass name.
  static llvm::StringRef name() { return "Simplify masked memory operations"; }
};

/// @brief reassociate uniform binary operators and split branches
class UniformReassociationPass
    : public llvm::PassInfoMixin<UniformReassociationPass> {
 public:
  UniformReassociationPass() = default;

  static llvm::StringRef name() { return "Reassociate uniform binops"; }

  llvm::PreservedAnalyses run(llvm::Function &,
                              llvm::FunctionAnalysisManager &);
};

/// @brief Removes uniform divergence reductions created by CFG conversion
class DivergenceCleanupPass
    : public llvm::PassInfoMixin<DivergenceCleanupPass> {
 public:
  /// @brief Create a new pass object.
  DivergenceCleanupPass() = default;

  /// @brief Remove uniform divergence reductions.
  ///
  /// @param[in] F Function to optimize.
  /// @param[in] AM FunctionAnalysisManager providing analyses.
  ///
  /// @return Preserved analyses.
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &AM);

  /// @brief Pass name.
  static llvm::StringRef name() {
    return "Remove uniform divergence reductions";
  }
};

}  // namespace vecz

#endif  // VECZ_TRANSFORM_PASSES_H_INCLUDED
