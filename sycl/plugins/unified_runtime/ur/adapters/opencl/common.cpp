//===--------- common.hpp - OpenCL Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "common.hpp"

#include <sycl/detail/pi.h>

namespace cl {
// Global variables for ZER_EXT_RESULT_ADAPTER_SPECIFIC_ERROR
thread_local ur_result_t ErrorMessageCode = UR_RESULT_SUCCESS;
thread_local char ErrorMessage[cl::MaxMessageSize];

// Utility function for setting a message and warning
[[maybe_unused]] void setErrorMessage(const char *message,
                                      ur_result_t error_code) {
  assert(strlen(message) <= cl::MaxMessageSize);
  strcpy(cl::ErrorMessage, message);
  ErrorMessageCode = error_code;
}
} // namespace cl

// Returns plugin specific error and warning messages; common implementation
// that can be shared between adapters
ur_result_t urGetLastResult(ur_platform_handle_t, const char **ppMessage) {
  *ppMessage = &cl::ErrorMessage[0];
  return cl::ErrorMessageCode;
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

/// Common API for getting the native handle of a UR object
///
/// \param urObj is the UR object to get the native handle of
/// \param nativeHandle is a pointer to be set to the native handle
///
/// PI_SUCCESS
ur_result_t urGetNativeHandle(void *urObj, ur_native_handle_t *nativeHandle) {
  UR_ASSERT(nativeHandle, UR_RESULT_ERROR_INVALID_NULL_POINTER)
  *nativeHandle = reinterpret_cast<ur_native_handle_t>(urObj);
  return UR_RESULT_SUCCESS;
}

cl_ext::ExtFuncPtrCacheT *ExtFuncPtrCache;
