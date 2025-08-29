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

/// @file
///
/// @brief Function scalarizer.

#ifndef VECZ_TRANSFORM_SCALARIZATION_PASS_H_INCLUDED
#define VECZ_TRANSFORM_SCALARIZATION_PASS_H_INCLUDED

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/PassManager.h>

namespace llvm {
class Function;
}  // namespace llvm

namespace vecz {

class VectorizationUnit;

/// \addtogroup scalarization Scalarization Stage
/// @{
/// \ingroup vecz

/// @brief Scalarization pass where vector instructions that need it are
/// scalarized, starting from leaves.
class ScalarizationPass : public llvm::PassInfoMixin<ScalarizationPass> {
 public:
  /// @brief Create a new scalarizaation pass.
  ScalarizationPass();

  /// @brief Unique identifier for the pass.
  static void *ID() { return (void *)&PassID; }

  /// @brief Scalarize the given function.
  ///
  /// @param[in] F Function to scalarize.
  /// @param[in] AM FunctionAnalysisManager providing analyses.
  ///
  /// @return Preserved analyses.
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &AM);

  /// @brief Name of the pass.
  static llvm::StringRef name() { return "Function scalarization"; }

 private:
  static char PassID;
};

/// @}
}  // namespace vecz

#endif  // VECZ_TRANSFORM_SCALARIZATION_PASS_H_INCLUDED
