/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file sanitizer_allocator.hpp
 *
 */

#pragma once

#include "sanitizer_common.hpp"

namespace ur_sanitizer_layer {

enum class AllocType {
  UNKNOWN,
  DEVICE_USM,
  SHARED_USM,
  HOST_USM,
  MEM_BUFFER,
  DEVICE_GLOBAL
};

inline const char *ToString(AllocType Type) {
  switch (Type) {
  case AllocType::DEVICE_USM:
    return "Device USM";
  case AllocType::HOST_USM:
    return "Host USM";
  case AllocType::SHARED_USM:
    return "Shared USM";
  case AllocType::MEM_BUFFER:
    return "Memory Buffer";
  case AllocType::DEVICE_GLOBAL:
    return "Device Global";
  default:
    return "Unknown Type";
  }
}

// Allocating USM with validation, so that we can ensure the allocated addresses
// satisfy the assumption we made for shadow memory
ur_result_t SafeAllocate(ur_context_handle_t Context, ur_device_handle_t Device,
                         uptr Size, const ur_usm_desc_t *Properties,
                         ur_usm_pool_handle_t Pool, AllocType Type,
                         void **Allocated);

} // namespace ur_sanitizer_layer
