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
/// Helper functions for working with sub_group and work_group functions.

#ifndef COMPILER_UTILS_GROUP_COLLECTIVE_HELPERS_H_INCLUDED
#define COMPILER_UTILS_GROUP_COLLECTIVE_HELPERS_H_INCLUDED

#include <llvm/Analysis/IVDescriptors.h>

namespace llvm {
class Constant;
class Function;
class Type;
}  // namespace llvm

namespace compiler {
namespace utils {
/// @brief Utility function for retrieving the neutral value of a
/// reduction/scan operation. A neutral value is one that does not affect the
/// result of a given operation, e.g., adding 0 or multiplying by 1.
///
/// @param[in] Kind The kind of scan/reduction operation
/// @param[in] Ty The type of the returned neutral value. Must match the type
/// assumed by @a Kind, e.g., a floating-point type for floating-point
/// operations.
///
/// @return The neutral value, or nullptr if unhandled.
llvm::Constant *getNeutralVal(llvm::RecurKind Kind, llvm::Type *Ty);

/// @brief Utility function for retrieving the identity value of a
/// reduction/scan operation. The identity value is one that is expected to be
/// found in the first element of an exclusive scan. It is equal to the neutral
/// value (see @ref getNeutralVal) in all cases except in floating-point
/// min/max, where -INF/+INF is the expected identity and in floating-point
/// addition, where 0.0 (not -0.0 which is the neutral value) is the expected
/// identity.
///
/// @param[in] Kind The kind of scan/reduction operation
/// @param[in] Ty The type of the returned neutral value. Must match the type
/// assumed by @a Kind, e.g., a floating-point type for floating-point
/// operations.
///
/// @return The neutral value, or nullptr if unhandled.
llvm::Constant *getIdentityVal(llvm::RecurKind Kind, llvm::Type *Ty);

/// @brief Represents a work-group or sub-group collective operation.
struct GroupCollective {
  /// @brief The different operation types a group collective can represent.
  enum class OpKind {
    All,
    Any,
    Reduction,
    ScanInclusive,
    ScanExclusive,
    Broadcast,
    Shuffle,
    ShuffleUp,
    ShuffleDown,
    ShuffleXor,
  };

  /// @brief The possible scopes of a group collective.
  enum class ScopeKind { WorkGroup, SubGroup, VectorGroup };

  /// @brief The operation type of the group collective.
  OpKind Op = OpKind::All;
  /// @brief The scope of the group collective operation.
  ScopeKind Scope = ScopeKind::WorkGroup;
  /// @brief The llvm recurrence operation this can be mapped to. For broadcasts
  /// this will be None.
  llvm::RecurKind Recurrence = llvm::RecurKind::None;
  /// @brief True if the operation is logical, rather than bitwise.
  bool IsLogical = false;
  /// @brief Returns true for Any/All type collective operations.
  bool isAnyAll() const { return Op == OpKind::Any || Op == OpKind::All; }
  /// @brief Returns true for inclusive/exclusive scan collective operations.
  bool isScan() const {
    return Op == OpKind::ScanExclusive || Op == OpKind::ScanInclusive;
  }
  /// @brief Returns true for reduction collective operations.
  bool isReduction() const { return Op == OpKind::Reduction; }
  /// @brief Returns true for broadcast collective operations.
  bool isBroadcast() const { return Op == OpKind::Broadcast; }
  bool isShuffleLike() const {
    return Op == OpKind::Shuffle || Op == OpKind::ShuffleUp ||
           Op == OpKind::ShuffleDown || Op == OpKind::ShuffleXor;
  }
  /// @brief Returns true for sub-group collective operations.
  bool isSubGroupScope() const { return Scope == ScopeKind::SubGroup; }
  /// @brief Returns true for work-group collective operations.
  bool isWorkGroupScope() const { return Scope == ScopeKind::WorkGroup; }
};
}  // namespace utils
}  // namespace compiler

#endif  // COMPILER_UTILS_GROUP_COLLECTIVE_HELPERS_H_INCLUDED
