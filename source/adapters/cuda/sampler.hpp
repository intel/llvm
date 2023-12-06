//===--------- sampler.hpp - CUDA Adapter ---------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <ur/ur.hpp>

/// Implementation of samplers for CUDA
///
/// Sampler property layout:
/// |     <bits>     | <usage>
/// -----------------------------------
/// |  31 30 ... 12  | N/A
/// |       11       | mip filter mode
/// |    10 9 8      | addressing mode 3
/// |     7 6 5      | addressing mode 2
/// |     4 3 2      | addressing mode 1
/// |       1        | filter mode
/// |       0        | normalize coords
struct ur_sampler_handle_t_ {
  std::atomic_uint32_t RefCount;
  uint32_t Props;
  float MinMipmapLevelClamp;
  float MaxMipmapLevelClamp;
  float MaxAnisotropy;
  ur_context_handle_t Context;

  ur_sampler_handle_t_(ur_context_handle_t Context)
      : RefCount(1), Props(0), MinMipmapLevelClamp(0.0f),
        MaxMipmapLevelClamp(0.0f), MaxAnisotropy(0.0f), Context(Context) {}

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }

  uint32_t decrementReferenceCount() noexcept { return --RefCount; }

  uint32_t getReferenceCount() const noexcept { return RefCount; }

  ur_bool_t isNormalizedCoords() const noexcept {
    return static_cast<ur_bool_t>(Props & 0b1);
  }

  ur_sampler_filter_mode_t getFilterMode() const noexcept {
    return static_cast<ur_sampler_filter_mode_t>((Props >> 1) & 0b1);
  }

  ur_sampler_addressing_mode_t getAddressingMode() const noexcept {
    return static_cast<ur_sampler_addressing_mode_t>((Props >> 2) & 0b111);
  }

  ur_sampler_addressing_mode_t getAddressingModeDim(size_t i) const noexcept {
    return static_cast<ur_sampler_addressing_mode_t>((Props >> (2 + (i * 3))) &
                                                     0b111);
  }

  ur_sampler_filter_mode_t getMipFilterMode() const noexcept {
    return static_cast<ur_sampler_filter_mode_t>((Props >> 11) & 0b1);
  }
};
