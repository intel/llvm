//===---------- physical_mem.hpp - HIP Adapter ----------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "common.hpp"
#include "common/ur_ref_counter.hpp"

#include "device.hpp"
#include "platform.hpp"

/// UR queue mapping on physical memory allocations used in virtual memory
/// management.
/// TODO: Implement.
///
struct ur_physical_mem_handle_t_ : ur::hip::handle_base {
  std::atomic_uint32_t RefCount;

  ur_physical_mem_handle_t_() : handle_base(), RefCount(1) {}

  UR_ReferenceCounter &getRefCounter() noexcept { return RefCounter; }

private:
  UR_ReferenceCounter RefCounter;
};
