//===--------- common.hpp - OpenCL Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "common.hpp"

namespace cl_adapter {

/* Global variables for urPlatformGetLastError() */
thread_local int32_t ErrorMessageCode = 0;
thread_local char ErrorMessage[MaxMessageSize];

[[maybe_unused]] void setErrorMessage(const char *Message, int32_t ErrorCode) {
  assert(strlen(Message) <= cl_adapter::MaxMessageSize);
  strcpy(cl_adapter::ErrorMessage, Message);
  ErrorMessageCode = ErrorCode;
}
} // namespace cl_adapter

ur_result_t mapCLErrorToUR(cl_int Result) {
  switch (Result) {
  case CL_SUCCESS:
    return UR_RESULT_SUCCESS;
  case CL_OUT_OF_HOST_MEMORY:
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  case CL_INVALID_VALUE:
  case CL_INVALID_BUILD_OPTIONS:
    return UR_RESULT_ERROR_INVALID_VALUE;
  case CL_INVALID_PLATFORM:
    return UR_RESULT_ERROR_INVALID_PLATFORM;
  case CL_DEVICE_NOT_FOUND:
    return UR_RESULT_ERROR_DEVICE_NOT_FOUND;
  case CL_INVALID_OPERATION:
    return UR_RESULT_ERROR_INVALID_OPERATION;
  case CL_INVALID_ARG_VALUE:
    return UR_RESULT_ERROR_INVALID_ARGUMENT;
  case CL_INVALID_EVENT:
    return UR_RESULT_ERROR_INVALID_EVENT;
  case CL_INVALID_EVENT_WAIT_LIST:
    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
  case CL_INVALID_BINARY:
    return UR_RESULT_ERROR_INVALID_BINARY;
  case CL_INVALID_KERNEL_NAME:
    return UR_RESULT_ERROR_INVALID_KERNEL_NAME;
  case CL_BUILD_PROGRAM_FAILURE:
    return UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE;
  case CL_INVALID_WORK_GROUP_SIZE:
    return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
  case CL_INVALID_WORK_ITEM_SIZE:
    return UR_RESULT_ERROR_INVALID_WORK_ITEM_SIZE;
  case CL_INVALID_WORK_DIMENSION:
    return UR_RESULT_ERROR_INVALID_WORK_DIMENSION;
  case CL_OUT_OF_RESOURCES:
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  case CL_INVALID_MEM_OBJECT:
    return UR_RESULT_ERROR_INVALID_MEM_OBJECT;
  default:
    return UR_RESULT_ERROR_UNKNOWN;
  }
}

void cl_adapter::die(const char *Message) {
  std::cerr << "ur_die: " << Message << "\n";
  std::terminate();
}

/// Common API for getting the native handle of a UR object
///
/// \param URObj is the UR object to get the native handle of
/// \param NativeHandle is a pointer to be set to the native handle
///
/// UR_RESULT_SUCCESS
ur_result_t getNativeHandle(void *URObj, ur_native_handle_t *NativeHandle) {
  *NativeHandle = reinterpret_cast<ur_native_handle_t>(URObj);
  return UR_RESULT_SUCCESS;
}
