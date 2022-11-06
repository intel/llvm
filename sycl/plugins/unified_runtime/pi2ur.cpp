//===---------------- pi2ur.cpp - PI API to UR API  --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------===//

// This thin layer performs conversion from PI API to Unified Runtime API
// TODO: remove when SYCL RT is changed to talk in UR directly

#include <pi2ur.hpp>

// Early exits on any error
#define HANDLE_ERRORS(urCall)                                                  \
  if (auto Result = urCall)                                                    \
    return ur2piResult(Result);

__SYCL_EXPORT pi_result piPlatformsGet(pi_uint32 num_entries,
                                       pi_platform *platforms,
                                       pi_uint32 *num_platforms) {

  // https://spec.oneapi.io/unified-runtime/latest/core/api.html#zerplatformget

  uint32_t Count = num_entries;
  auto phPlatforms = reinterpret_cast<zer_platform_handle_t *>(platforms);
  HANDLE_ERRORS(zerPlatformGet(&Count, phPlatforms));
  if (*num_platforms) {
    *num_platforms = Count;
  }
  return PI_SUCCESS;
}

__SYCL_EXPORT pi_result piPlatformGetInfo(pi_platform platform,
                                          pi_platform_info param_name,
                                          size_t param_value_size,
                                          void *param_value,
                                          size_t *param_value_size_ret) {
  die("Unified Runtime: piPlatformGetInfo is not implemented");
}
