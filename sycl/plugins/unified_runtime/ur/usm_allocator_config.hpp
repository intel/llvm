//===--- usm_allocator_config.hpp -configuration for USM memory allocator---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef USM_ALLOCATOR_CONFIG
#define USM_ALLOCATOR_CONFIG

#include "usm_allocator.hpp"

#include <string>

namespace usm_settings {

enum MemType { Host, Device, Shared, SharedReadOnly, All };

// Reads and stores configuration for all instances of USM allocator
class USMAllocatorConfig {
public:
  size_t EnableBuffers = 1;

  USMAllocatorParameters Configs[MemType::All];

  // String names of memory types for printing in limits traces.
  static constexpr const char *MemTypeNames[MemType::All] = {
      "Host", "Device", "Shared", "SharedReadOnly"};

  USMAllocatorConfig();
};
} // namespace usm_settings

#endif
