//===---------------- pi2ur.cpp - PI API to UR API  --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------===//

#include <unordered_map>

#include "zer_api.h"
#include <sycl/detail/pi.h>

// Map of UR error codes to PI error codes
static pi_result ur2piResult(zer_result_t urResult) {

  // TODO: replace "global lifetime" objects with a non-trivial d'tor with
  // either pointers to such objects (which would be allocated and dealocated
  // during init and teardown) or objects with trivial d'tor.
  // E.g. for this case we could have an std::array with sorted values.
  //
  static std::unordered_map<zer_result_t, pi_result> ErrorMapping = {
      {ZER_RESULT_SUCCESS, PI_SUCCESS},
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
