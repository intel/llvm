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
/// Prepare barriers pass.

#ifndef COMPILER_UTILS_PREPARE_BARRIERS_PASS_H_INCLUDED
#define COMPILER_UTILS_PREPARE_BARRIERS_PASS_H_INCLUDED

#include <llvm/IR/PassManager.h>

namespace compiler {
namespace utils {

/// @brief Pass for ensuring consistent barrier handling.
///
/// It inlines functions that contain barriers and gives each barrier call a
/// unique ID as metadata to ensure consistent handling of barriers in
/// different versions of the kernel (i.e. Scalar vs Vector). Run before Vecz
/// for mixed wrapper kernels made up of multiple kernels to work.
///
/// Runs over all kernels with "kernel entry point" metadata.
class PrepareBarriersPass final
    : public llvm::PassInfoMixin<PrepareBarriersPass> {
 public:
  llvm::PreservedAnalyses run(llvm::Module &, llvm::ModuleAnalysisManager &);
};
}  // namespace utils
}  // namespace compiler

#endif  // COMPILER_UTILS_PREPARE_BARRIERS_PASS_H_INCLUDED
