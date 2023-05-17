//===--------- common.hpp - OpenCL Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "common.hpp"

// Global variables for ZER_EXT_RESULT_ADAPTER_SPECIFIC_ERROR
thread_local ur_result_t ErrorMessageCode = UR_RESULT_SUCCESS;
thread_local char ErrorMessage[MaxMessageSize];

// Utility function for setting a message and warning
[[maybe_unused]] void setErrorMessage(const char *message,
                                      ur_result_t error_code) {
  assert(strlen(message) <= MaxMessageSize);
  strcpy(ErrorMessage, message);
  ErrorMessageCode = error_code;
}

// Returns plugin specific error and warning messages; common implementation
// that can be shared between adapters
ur_result_t urGetLastResult(ur_platform_handle_t, const char **ppMessage) {
  *ppMessage = &ErrorMessage[0];
  return ErrorMessageCode;
}

ur_result_t map_cl_error_to_ur(cl_int result) {

  switch (result) {
  case CL_SUCCESS:
    return UR_RESULT_SUCCESS;
  case CL_OUT_OF_HOST_MEMORY:
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  case CL_INVALID_VALUE:
    return UR_RESULT_ERROR_INVALID_VALUE;
  case CL_INVALID_PLATFORM:
    return UR_RESULT_ERROR_INVALID_PLATFORM;
  default:
    return UR_RESULT_ERROR_UNKNOWN;
  }
}
