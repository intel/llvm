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
/// @brief Function packetizer.

#ifndef VECZ_TRANSFORM_PACKETIZATION_PASS_H_INCLUDED
#define VECZ_TRANSFORM_PACKETIZATION_PASS_H_INCLUDED

#include <llvm/IR/PassManager.h>

namespace vecz {

class VectorizationUnit;

/// \addtogroup packetization Packetization Stage
/// @{
/// \ingroup vecz

/// @brief Vectorization pass where scalar instructions that need it are
/// packetized, starting from leaves.
class PacketizationPass : public llvm::PassInfoMixin<PacketizationPass> {
 public:
  /// @brief Create a new packetization pass object.
  PacketizationPass() = default;

  /// @brief Create a new packetization pass object.
  ///
  /// @param[in] P Pass to move.
  PacketizationPass(PacketizationPass &&P) = default;

  // Mark default copy constructor as deleted
  PacketizationPass(const PacketizationPass &) = delete;

  /// @brief Deleted move assignment operator.
  ///
  /// Also deletes the copy assignment operator.
  PacketizationPass &operator=(PacketizationPass &&) = delete;

  /// @brief Unique identifier for the pass.
  static void *ID() { return (void *)&PassID; }

  /// @brief Packetize the given function, duplicating its behaviour (defined
  /// values and side effects) for each lane of a SIMD packet.
  ///
  /// @param[in] F Function to packetize.
  /// @param[in] AM FunctionAnalysisManager providing analyses.
  ///
  /// @return Preserved passes.
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &AM);

  /// @brief Pass name.
  static llvm::StringRef name() { return "Function packetization"; }

  /// @brief Unique identifier for the pass.
  static char PassID;
};

/// @}
}  // namespace vecz

#endif  // VECZ_TRANSFORM_PACKETIZATION_PASS_H_INCLUDED
