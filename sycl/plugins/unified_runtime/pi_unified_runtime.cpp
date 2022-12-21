//===--- pi_unified_runtime.cpp - Unified Runtime PI Plugin  -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------===//

#include <pi2ur.hpp>

extern "C" {
__SYCL_EXPORT pi_result piPlatformsGet(pi_uint32 num_entries,
                                       pi_platform *platforms,
                                       pi_uint32 *num_platforms) {
  return pi2ur::piPlatformsGet(num_entries, platforms, num_platforms);
}
} // extern "C
