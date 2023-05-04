//===--------- ur_level_zero_mem.cpp - Level Zero Adapter -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "ur_level_zero_mem.hpp"

bool ShouldUseUSMAllocator() {
  // Enable allocator by default if it's not explicitly disabled
  return std::getenv("SYCL_PI_LEVEL_ZERO_DISABLE_USM_ALLOCATOR") == nullptr;
}
const bool UseUSMAllocator = ShouldUseUSMAllocator();