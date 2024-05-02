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
/// Replace local module-scope variables pass.

#ifndef COMPILER_UTILS_REPLACE_LOCAL_MODULE_SCOPE_VARIABLES_PASS_H_INCLUDED
#define COMPILER_UTILS_REPLACE_LOCAL_MODULE_SCOPE_VARIABLES_PASS_H_INCLUDED

#include <llvm/IR/PassManager.h>

namespace compiler {
namespace utils {

/// @brief __local address space automatic variables are represented in the
/// LLVM module as global variables with address space 3. This pass identifies
/// these variables and places them into a struct allocated (via alloca) in a
/// newly created wrapper function. A pointer to the struct is then passed
/// via a parameter to the original kernel.
///
/// Runs over all kernels with "kernel" metadata.
class ReplaceLocalModuleScopeVariablesPass final
    : public llvm::PassInfoMixin<ReplaceLocalModuleScopeVariablesPass> {
 public:
  llvm::PreservedAnalyses run(llvm::Module &, llvm::ModuleAnalysisManager &);
};
}  // namespace utils
}  // namespace compiler

#endif  // COMPILER_UTILS_REPLACE_LOCAL_MODULE_SCOPE_VARIABLES_PASS_H_INCLUDED
