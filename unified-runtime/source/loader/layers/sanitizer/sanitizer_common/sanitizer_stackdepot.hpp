/*
 *
 * Copyright (C) 2025 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file sanitizer_stackdepot.hpp
 *
 */

#pragma once

#include "sanitizer_stacktrace.hpp"

namespace ur_sanitizer_layer {

const uint32_t kHeapTypeCount = 4;
enum HeapType { DeviceUSM, HostUSM, SharedUSM, Local };

inline const char *ToString(HeapType Type) {
  switch (Type) {
  case HeapType::DeviceUSM:
    return "Device USM";
  case HeapType::HostUSM:
    return "Host USM";
  case HeapType::SharedUSM:
    return "Shared USM";
  case HeapType::Local:
    return "Local Memory";
  default:
    return "Unknown Heap Type";
  }
}

// Save/load stack with corresponding stack id
void StackDepotPut(uint32_t Id, StackTrace &Stack);
StackTrace StackDepotGet(uint32_t Id);

} // namespace ur_sanitizer_layer
