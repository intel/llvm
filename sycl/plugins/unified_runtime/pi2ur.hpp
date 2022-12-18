//===---------------- pi2ur.hpp - PI API to UR API  --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------===//
#pragma once

#include <unordered_map>

#include "zer_api.h"
#include <sycl/detail/pi.h>
#include <ur/ur.hpp>

// Map of UR error codes to PI error codes
static pi_result ur2piResult(zer_result_t urResult) {

  // TODO: replace "global lifetime" objects with a non-trivial d'tor with
  // either pointers to such objects (which would be allocated and dealocated
  // during init and teardown) or objects with trivial d'tor.
  // E.g. for this case we could have an std::array with sorted values.
  //
  static std::unordered_map<zer_result_t, pi_result> ErrorMapping = {
      {ZER_RESULT_SUCCESS, PI_SUCCESS},
      {ZER_RESULT_ERROR_UNKNOWN, PI_ERROR_UNKNOWN},
      {ZER_RESULT_ERROR_DEVICE_LOST, PI_ERROR_DEVICE_NOT_FOUND},
      {ZER_RESULT_INVALID_OPERATION, PI_ERROR_INVALID_OPERATION},
      {ZER_RESULT_INVALID_PLATFORM, PI_ERROR_INVALID_PLATFORM},
      {ZER_RESULT_ERROR_INVALID_ARGUMENT, PI_ERROR_INVALID_ARG_VALUE},
      {ZER_RESULT_INVALID_VALUE, PI_ERROR_INVALID_VALUE},
      {ZER_RESULT_INVALID_EVENT, PI_ERROR_INVALID_EVENT},
      {ZER_RESULT_INVALID_BINARY, PI_ERROR_INVALID_BINARY},
      {ZER_RESULT_INVALID_KERNEL_NAME, PI_ERROR_INVALID_KERNEL_NAME},
      {ZER_RESULT_ERROR_INVALID_FUNCTION_NAME, PI_ERROR_BUILD_PROGRAM_FAILURE},
      {ZER_RESULT_INVALID_WORK_GROUP_SIZE, PI_ERROR_INVALID_WORK_GROUP_SIZE},
      {ZER_RESULT_ERROR_MODULE_BUILD_FAILURE, PI_ERROR_BUILD_PROGRAM_FAILURE},
      {ZER_RESULT_ERROR_OUT_OF_DEVICE_MEMORY, PI_ERROR_OUT_OF_RESOURCES},
      {ZER_RESULT_ERROR_OUT_OF_HOST_MEMORY, PI_ERROR_OUT_OF_HOST_MEMORY}};

  auto It = ErrorMapping.find(urResult);
  if (It == ErrorMapping.end()) {
    return PI_ERROR_UNKNOWN;
  }
  return It->second;
}

// Early exits on any error
#define HANDLE_ERRORS(urCall)                                                  \
  if (auto Result = urCall)                                                    \
    return ur2piResult(Result);

namespace pi2ur {
inline pi_result piPlatformsGet(pi_uint32 num_entries, pi_platform *platforms,
                                pi_uint32 *num_platforms) {

  // https://spec.oneapi.io/unified-runtime/latest/core/api.html#zerplatformget

  uint32_t Count = num_entries;
  auto phPlatforms = reinterpret_cast<zer_platform_handle_t *>(platforms);
  HANDLE_ERRORS(zerPlatformGet(&Count, phPlatforms));
  if (num_platforms) {
    *num_platforms = Count;
  }
  return PI_SUCCESS;
}

inline pi_result piPlatformGetInfo(pi_platform platform,
                                   pi_platform_info param_name,
                                   size_t param_value_size, void *param_value,
                                   size_t *param_value_size_ret) {
  (void)platform;
  (void)param_name;
  (void)param_value_size;
  (void)param_value;
  (void)param_value_size_ret;
  die("Unified Runtime: piPlatformGetInfo is not implemented");
}
} // namespace pi2ur
