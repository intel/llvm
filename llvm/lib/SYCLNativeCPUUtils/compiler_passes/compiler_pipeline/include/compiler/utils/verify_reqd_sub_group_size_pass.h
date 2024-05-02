
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

#ifndef COMPILER_UTILS_VERIFY_REQD_SUB_GROUP_SIZE_PASS_H_INCLUDED
#define COMPILER_UTILS_VERIFY_REQD_SUB_GROUP_SIZE_PASS_H_INCLUDED

#include <llvm/IR/PassManager.h>

namespace compiler {
namespace utils {

/// @addtogroup utils
/// @{

/// @brief This pass checks that any kernels with required sub-group sizes are
/// using sub-group sizes that are marked as legal by the device.
///
/// Raises a compile diagnostic on kernel which breaches this rule.
class VerifyReqdSubGroupSizeLegalPass
    : public llvm::PassInfoMixin<VerifyReqdSubGroupSizeLegalPass> {
 public:
  VerifyReqdSubGroupSizeLegalPass() = default;
  llvm::PreservedAnalyses run(llvm::Module &, llvm::ModuleAnalysisManager &);
};

/// @brief This pass checks that any kernels with required sub-group sizes have
/// had those sub-group sizes successfully satisfied by the compiler.
///
/// Raises a compile diagnostic on kernel which breaches this rule.
class VerifyReqdSubGroupSizeSatisfiedPass
    : public llvm::PassInfoMixin<VerifyReqdSubGroupSizeSatisfiedPass> {
 public:
  VerifyReqdSubGroupSizeSatisfiedPass() = default;
  llvm::PreservedAnalyses run(llvm::Module &, llvm::ModuleAnalysisManager &);
};

/// @}
}  // namespace utils
}  // namespace compiler

#endif  // COMPILER_UTILS_VERIFY_REQD_SUB_GROUP_SIZE_PASS_H_INCLUDED
