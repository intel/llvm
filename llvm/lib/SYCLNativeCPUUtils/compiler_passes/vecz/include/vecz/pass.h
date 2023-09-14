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
/// @brief Vecz passes header.

#ifndef VECZ_PASS_H
#define VECZ_PASS_H

#include <compiler/utils/vectorization_factor.h>
#include <llvm/IR/PassManager.h>

#include <cstdint>
#include <optional>

#include "vecz/vecz_choices.h"

namespace llvm {
class ModulePass;
class StringRef;
class Module;
class TargetMachine;
}  // namespace llvm

namespace compiler {
namespace utils {
class BuiltinInfo;
}  // namespace utils
}  // namespace compiler

namespace vecz {
/// @addtogroup vecz
/// @{

struct VeczPassOptions {
  VeczPassOptions() : vecz_auto(false), vec_dim_idx(0), local_size(0) {}

  /// @brief boolean choices such as double support, partial scalarization
  vecz::VectorizationChoices choices;

  /// @brief vectorization factor, including known min and scalable flag
  compiler::utils::VectorizationFactor factor;

  /// @brief automatically work out factor
  bool vecz_auto;

  /// @brief Index of vectorization dimension to use (0 => x, 1 => y, 2 => z).
  uint32_t vec_dim_idx;

  /// @brief local_size Value specifying the local size for the function (0 is
  /// unknown)
  uint64_t local_size;
};

/// @brief Returns the vectorization options that would vectorize the provided
/// function to its required sub-group size.
std::optional<VeczPassOptions> getReqdSubgroupSizeOpts(llvm::Function &);

/// @brief Returns the vectorization options that would vectorize the provided
/// function to its required sub-group size (if set) or one of the device's
/// sub-group sizes.
///
/// Only returns options if the function uses sub-group operations, as
/// determined by the SubGroupAnalysis pass.
///
/// Tries to find a good fit that produces one of the device's sub-group sizes,
/// preferring ones which fit the known local work-group size and powers of
/// two. The device's sub-group sizes can be sorted such that preferable sizes
/// are placed towards the front.
std::optional<VeczPassOptions> getAutoSubgroupSizeOpts(
    llvm::Function &, llvm::ModuleAnalysisManager &);

/// @brief Analysis pass which determines on which functions @ref RunVeczPass
/// should operate.
class VeczPassOptionsAnalysis
    : public llvm::AnalysisInfoMixin<VeczPassOptionsAnalysis> {
  using VeczPassOptionsCallbackFn =
      std::function<bool(llvm::Function &, llvm::ModuleAnalysisManager &,
                         llvm::SmallVectorImpl<VeczPassOptions> &)>;
  friend AnalysisInfoMixin<VeczPassOptionsAnalysis>;
  static llvm::AnalysisKey Key;
  VeczPassOptionsCallbackFn queryFunc =
      [](llvm::Function &F, llvm::ModuleAnalysisManager &,
         llvm::SmallVectorImpl<VeczPassOptions> &Opts) -> bool {
    if (F.getCallingConv() != llvm::CallingConv::SPIR_KERNEL) {
      return false;
    }
    // TODO what are our defaults, here?
    Opts.emplace_back();
    return true;
  };

 public:
  VeczPassOptionsAnalysis() = default;
  /// @brief explicit constructor which uses the given callback to determine
  /// whether vectorization should be performed on the passed function. If the
  /// default constructor is used, all functions with a SPIR calling convention
  /// will be vectorized
  explicit VeczPassOptionsAnalysis(VeczPassOptionsCallbackFn queryFunc)
      : queryFunc(queryFunc) {}
  using Result = VeczPassOptionsCallbackFn;
  Result run(llvm::Module &, llvm::ModuleAnalysisManager &) {
    return queryFunc;
  }
};

/// @brief A helper pass which can be used to inspect and test the
/// vectorization options set on a per-function basis.
class VeczPassOptionsPrinterPass
    : public llvm::PassInfoMixin<VeczPassOptionsPrinterPass> {
  llvm::raw_ostream &OS;

 public:
  explicit VeczPassOptionsPrinterPass(llvm::raw_ostream &OS) : OS(OS) {}

  llvm::PreservedAnalyses run(llvm::Module &, llvm::ModuleAnalysisManager &);
};

/// @brief A new-style module pass that provides a wrapper for using the
/// the ComputeAorta IR vectorizer. This vectorizes kernels
/// to vectorization factor specified when the pass is created. In our case this
/// is typically the local size in the first dimension but there are other
/// factors to consider when picking the vectorization factor, like being a
/// power of 2. This pass queries the @ref `VeczShouldRunOnFunctionAnalysis`, so
/// if you do not wish all kernels to be vectorized, you must ensure your pass
/// manager's ModuleAnalysisManager is configured with a custom @ref
/// `VeczShouldRunOnFunctionAnalysis`
class RunVeczPass : public llvm::PassInfoMixin<RunVeczPass> {
 public:
  /// @brief llvm's entry point for the PassManager
  llvm::PreservedAnalyses run(llvm::Module &, llvm::ModuleAnalysisManager &);
};

/// @}
}  // namespace vecz

#endif  // VECZ_PASS_H
