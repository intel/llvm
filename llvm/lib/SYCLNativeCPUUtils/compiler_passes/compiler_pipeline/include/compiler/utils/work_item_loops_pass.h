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
/// Work-item loops pass, splitting into "barrier regions"

#ifndef COMPILER_UTILS_WORK_ITEM_LOOPS_PASS_H_INCLUDED
#define COMPILER_UTILS_WORK_ITEM_LOOPS_PASS_H_INCLUDED

#include <compiler/utils/barrier_regions.h>
#include <compiler/utils/metadata.h>
#include <compiler/utils/vectorization_factor.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/PassManager.h>

#include <string>

namespace llvm {
class DominatorTree;
}

namespace compiler {
namespace utils {

class BuiltinInfo;
class BarrierWithLiveVars;

struct WorkItemLoopsPassOptions {
  /// @brief Set to true if the pass should add extra alloca
  /// instructions to preserve the values of variables between barriers.
  bool IsDebug = false;
  /// @brief Set to true if the pass should forcibly omit scalar
  /// tail loops from wrapped vector kernels, even if the local work-group size
  /// is not known to be a multiple of the vectorization factor.
  bool ForceNoTail = false;
};

/// @brief The "work-item loops" pass.
///
/// This pass adds loops around implicitly SIMT kernels such that the original
/// kernel is wrapped in a new function that runs over each work-item in the
/// work-group and calls the original kernel: the scheduling model thus becomes
/// explicit.
///
/// The work-item loops pass assumes that:
///
/// * Any functions containing barrier-like functions have already been inlined
/// into the kernel entry points
/// * the IDs of pairs of barrier-like functions align between 'main' and 'tail
/// kernels.
///
/// Both of these can be achieved by first running the PrepareBarriersPass.
///
/// The pass will query a kernel function for the `reqd_work_group_size`
/// metadata and optimize accordingly in the presence of it.
///
/// Runs over all kernels with "kernel entry point" metadata. Work-item orders
/// are sourced from the "work item order" function metadata on each kernel.
class WorkItemLoopsPass final : public llvm::PassInfoMixin<WorkItemLoopsPass> {
 public:
  /// @brief Constructor.
  WorkItemLoopsPass(const WorkItemLoopsPassOptions &Options)
      : IsDebug(Options.IsDebug), ForceNoTail(Options.ForceNoTail) {}

  llvm::PreservedAnalyses run(llvm::Module &, llvm::ModuleAnalysisManager &);

 private:
  /// @brief Make the work-item-loop wrapper function.
  /// This creates a wrapper function that iterates over a work group, calling
  /// the kernel for each work item, respecting the semantics of any barriers
  /// present. The wrapped kernel may be a scalar kernel, a vectorized kernel,
  /// or both. When the wrapped kernel wraps both a vector and scalar kernel,
  /// all vectorized work items will be executed first, and the scalar tail
  /// last.
  ///
  /// The wrapper function is created as a new function suffixed by
  /// ".mux-barrier-wrapper". The original unwrapped kernel(s)s will be left in
  /// the Module, but marked as internal linkage so later passes can remove
  /// them if uncalled once inlined into the wrapper function.
  ///
  /// When wrapping only a scalar kernel, or only a vector kernel, pass the
  /// same Barrier object as both Barrier input parameters.
  ///
  /// @param[in] barrierMain the Barrier object of the main kernel function
  /// @param[in] barrierTail the Barrier object of the tail kernel function
  /// (may be nullptr).
  /// @param[in] baseName the base name to use on the new wrapper function
  /// @param[in] M the module the kernels live in
  /// @param[in] BI BuiltinInfo providing builtin information
  /// @return The new wrapper function
  llvm::Function *makeWrapperFunction(BarrierWithLiveVars &barrierMain,
                                      BarrierWithLiveVars *barrierTail,
                                      llvm::StringRef baseName, llvm::Module &M,
                                      BuiltinInfo &BI);

  const bool IsDebug;
  const bool ForceNoTail;
};
}  // namespace utils
}  // namespace compiler

#endif  // COMPILER_UTILS_WORK_ITEM_LOOPS_PASS_H_INCLUDED
