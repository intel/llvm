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
/// @brief SIMD packets hold a value for each lane.

#ifndef VECZ_SIMD_PACKET_H_INCLUDED
#define VECZ_SIMD_PACKET_H_INCLUDED

#include "debugging.h"

namespace llvm {
class Value;
}

namespace vecz {

/// @brief Represents the status of lanes within a packet. The most common
/// status would be that a lane can be either enabled or disabled.
struct PacketMask {
  /// @brief Create a new mask where all lanes are disabled.
  explicit PacketMask() : Value(0) {}
  /// @brief Create a new mask using an existing bit field.
  explicit PacketMask(uint64_t Mask) : Value(Mask) {}

  /// @brief Determine whether the given lane is enabled or not.
  /// @param[in] Lane Index of the lane to test.
  /// @return true if the lane is enabled, false otherwise.
  bool isEnabled(unsigned Lane) const {
    assert(Lane < CHAR_BIT * sizeof(Value) &&
           "Invalid lane, possible mask overflow");
    return (Value & (1ull << Lane)) != 0ull;
  }

  /// @brief Enable the given lane.
  /// @param[in] Lane Index of the lane to enable.
  void enable(unsigned Lane) {
    assert(Lane < CHAR_BIT * sizeof(Value) &&
           "Invalid lane, possible mask overflow");
    Value |= (1ull << Lane);
  }

  /// @brief Disable the given lane.
  /// @param[in] Lane Index of the lane to disable.
  void disable(unsigned Lane) {
    assert(Lane < CHAR_BIT * sizeof(Value) &&
           "Invalid lane, possible mask overflow");
    Value &= ~(1ull << Lane);
  }
  /// @brief Enable multiple lanes [0: NumLanes)
  /// @param[in] NumLanes Number of lanes to enable.
  void enableAll(unsigned NumLanes);

  /// @brief Bit field that describes which lanes are enabled.
  /// NOTE: The length of bitfield is limited to sizeof(uint64_t) * CHAR_BIT(8)
  uint64_t Value;
};

/// @brief Packet of LLVM values (e.g. instructions), one for each SIMD lane.
struct SimdPacket : public llvm::SmallVector<llvm::Value *, 4> {
  using SmallVector::SmallVector;

  /// @brief Return the value at the given index.
  /// @param[in] Index Index of the value to return.
  /// @return Value at the given index or null.
  llvm::Value *at(unsigned Index) const;
  /// @brief Set the value at the given index and enable the corresponding lane.
  /// @param[in] Index Index of the value to set.
  /// @param[in] V Value to store at the given index.
  void set(unsigned Index, llvm::Value *V);
  /// @brief Copy all enabled lanes from the other packet and update the mask.
  /// @param[in] Other Packet to copy values from.
  /// @return Reference to the current packet.
  SimdPacket &update(const SimdPacket &Other);

  /// @brief Bitmask of lanes that are 'enabled' in this packet.
  /// This can mean different things depending on the context:
  /// * By default, only lanes that are 'enabled' have a valid value.
  /// * When scalarizing, only lanes that are 'enabled' will be scalarized.
  PacketMask Mask;
};

}  // namespace vecz

#endif  // VECZ_SIMD_PACKET_H_INCLUDED
