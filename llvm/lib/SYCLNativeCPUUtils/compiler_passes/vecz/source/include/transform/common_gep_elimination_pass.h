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
/// @brief Remove duplicate GEP instructions.

#ifndef VECZ_TRANSFORM_COMMON_GEP_ELIMINATION_PASS_H_INCLUDED
#define VECZ_TRANSFORM_COMMON_GEP_ELIMINATION_PASS_H_INCLUDED

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/PassManager.h>

namespace vecz {

class VectorizationUnit;

/// @brief This pass removes every duplicate GEP instruction before the
/// packetization pass.
class CommonGEPEliminationPass
    : public llvm::PassInfoMixin<CommonGEPEliminationPass> {
 public:
  static void *ID() { return (void *)&PassID; };

  /// @brief Remove duplicate GEP instructions.
  ///
  /// @param[in] F Function to optimize.
  /// @param[in] AM FunctionAnalysisManager providing analyses.
  ///
  /// @return Preserved passes.
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &AM);

  /// @brief Pass name.
  static llvm::StringRef name() { return "Common GEP Elimination pass"; }

 private:
  /// @brief Identifier for the pass.
  static char PassID;
};
}  // namespace vecz

#endif  // VECZ_TRANSFORM_COMMON_GEP_ELIMINATION_PASS_H_INCLUDED
