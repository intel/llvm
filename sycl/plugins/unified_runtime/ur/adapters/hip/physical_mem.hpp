//===---------- physical_mem.hpp - HIP Adapter ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "common.hpp"
#include "device.hpp"
#include "platform.hpp"

/// UR queue mapping on physical memory allocations used in virtual memory
/// management.
/// TODO: Implement.
///
struct ur_physical_mem_handle_t_ {
  std::atomic_uint32_t RefCount;

  ur_physical_mem_handle_t_() : RefCount(1) {}

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }

  uint32_t decrementReferenceCount() noexcept { return --RefCount; }

  uint32_t getReferenceCount() const noexcept { return RefCount; }
};
