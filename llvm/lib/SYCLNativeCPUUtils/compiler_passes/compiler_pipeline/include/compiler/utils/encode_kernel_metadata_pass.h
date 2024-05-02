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
/// EncodeKernelMetadataPass pass.

#ifndef COMPILER_UTILS_ENCODE_KERNEL_METADATA_PASS_H_INCLUDED
#define COMPILER_UTILS_ENCODE_KERNEL_METADATA_PASS_H_INCLUDED

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/PassManager.h>

#include <optional>

namespace compiler {
namespace utils {

/// @brief Sets up the per-function mux metadata used by later passes.
/// Transfers per-module !opencl.kernel metadata to mux kernel metadata.
struct TransferKernelMetadataPass
    : public llvm::PassInfoMixin<TransferKernelMetadataPass> {
  explicit TransferKernelMetadataPass() {}

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
};

struct EncodeKernelMetadataPassOptions {
  std::string KernelName;
  std::optional<std::array<uint64_t, 3>> LocalSizes = std::nullopt;
};

struct EncodeKernelMetadataPass
    : public llvm::PassInfoMixin<EncodeKernelMetadataPass> {
  EncodeKernelMetadataPass(EncodeKernelMetadataPassOptions Options)
      : KernelName(Options.KernelName), LocalSizes(Options.LocalSizes) {}

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);

 private:
  std::string KernelName;
  std::optional<std::array<uint64_t, 3>> LocalSizes;
};
}  // namespace utils
}  // namespace compiler

#endif  // COMPILER_UTILS_ENCODE_KERNEL_METADATA_PASS_H_INCLUDED
