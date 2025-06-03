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

enum class HeapType { DeviceUSM, HostUSM, SharedUSM, Local };

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

uint32_t StackDepotPut(StackTrace &Stack, HeapType Type);
StackTrace StackDepotGet(uint32_t Id, HeapType Type);

} // namespace ur_sanitizer_layer
