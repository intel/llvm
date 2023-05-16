//===--------- sampler.hpp - HIP Adapter ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include <ur/ur.hpp>

#include "context.hpp"

/// Implementation of samplers for HIP
///
/// Sampler property layout:
/// | 31 30 ... 6 5 |      4 3 2      |     1      |         0        |
/// |      N/A      | addressing mode | fiter mode | normalize coords |
struct ur_sampler_handle_t_ : _ur_object {
  std::atomic_uint32_t refCount_;
  uint32_t props_;
  ur_context_handle_t context_;

  ur_sampler_handle_t_(ur_context_handle_t context)
      : refCount_(1), props_(0), context_(context) {}

  uint32_t increment_reference_count() noexcept { return ++refCount_; }

  uint32_t decrement_reference_count() noexcept { return --refCount_; }

  uint32_t get_reference_count() const noexcept { return refCount_; }
};
