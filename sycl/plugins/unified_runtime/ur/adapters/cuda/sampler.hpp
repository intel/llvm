//===--------- sampler.hpp - CUDA Adapter ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include <ur/ur.hpp>

/// Implementation of samplers for CUDA
///
/// Sampler property layout:
/// | 31 30 ... 6 5 |      4 3 2      |     1      |         0        |
/// |      N/A      | addressing mode | fiter mode | normalize coords |
struct ur_sampler_handle_t_ {
  std::atomic_uint32_t RefCount;
  uint32_t Props;
  ur_context_handle_t Context;

  ur_sampler_handle_t_(ur_context_handle_t Context)
      : RefCount(1), Props(0), Context(Context) {}

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }

  uint32_t decrementReferenceCount() noexcept { return --RefCount; }

  uint32_t getReferenceCount() const noexcept { return RefCount; }
};
