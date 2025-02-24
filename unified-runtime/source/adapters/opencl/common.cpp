//===--------- common.hpp - OpenCL Adapter ---------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"
#include "logger/ur_logger.hpp"
namespace cl_adapter {

/* Global variables for urAdapterGetLastError() */
thread_local int32_t ErrorMessageCode = 0;
thread_local char ErrorMessage[MaxMessageSize]{};

[[maybe_unused]] void setErrorMessage(const char *Message, int32_t ErrorCode) {
  assert(strlen(Message) < cl_adapter::MaxMessageSize);
  // Copy at most MaxMessageSize - 1 bytes to ensure the resultant string is
  // always null terminated.
  strncpy(cl_adapter::ErrorMessage, Message, MaxMessageSize - 1);

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
  case CL_INVALID_PROGRAM:
    return UR_RESULT_ERROR_INVALID_PROGRAM;
  case CL_INVALID_PROGRAM_EXECUTABLE:
    return UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE;
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
  case CL_INVALID_QUEUE_PROPERTIES:
    return UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES;
  case CL_INVALID_BUFFER_SIZE:
    return UR_RESULT_ERROR_INVALID_BUFFER_SIZE;
  case CL_INVALID_IMAGE_SIZE:
    return UR_RESULT_ERROR_INVALID_IMAGE_SIZE;
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
  case CL_INVALID_IMAGE_DESCRIPTOR:
    return UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR;
  case CL_IMAGE_FORMAT_NOT_SUPPORTED:
    return UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT;
  case CL_PROFILING_INFO_NOT_AVAILABLE:
    return UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE;
  case CL_LINK_PROGRAM_FAILURE:
    return UR_RESULT_ERROR_PROGRAM_LINK_FAILURE;
  case CL_INVALID_ARG_INDEX:
    return UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX;
  case CL_INVALID_COMMAND_BUFFER_KHR:
    return UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP;
  case CL_INVALID_SYNC_POINT_WAIT_LIST_KHR:
    return UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP;
  case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
    return UR_RESULT_ERROR_IN_EVENT_LIST_EXEC_STATUS;
  case CL_DEVICE_NOT_AVAILABLE:
    return UR_RESULT_ERROR_DEVICE_NOT_AVAILABLE;
  case CL_INVALID_KERNEL_ARGS:
    return UR_RESULT_ERROR_INVALID_KERNEL_ARGS;
  case CL_INVALID_COMMAND_QUEUE:
    return UR_RESULT_ERROR_INVALID_QUEUE;
  case CL_INVALID_ARG_SIZE:
    return UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE;
  case CL_INVALID_SPEC_ID:
    return UR_RESULT_ERROR_INVALID_SPEC_ID;
  default:
    return UR_RESULT_ERROR_UNKNOWN;
  }
}

void cl_adapter::die(const char *Message) {
  logger::always("ur_die: {}", Message);
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

cl_int getDeviceCommandBufferUpdateCapabilities(
    cl_device_id Dev,
    ur_device_command_buffer_update_capability_flags_t &UpdateCapabilities) {

  UpdateCapabilities = 0;

  size_t ExtSize = 0;
  CL_RETURN_ON_FAILURE(
      clGetDeviceInfo(Dev, CL_DEVICE_EXTENSIONS, 0, nullptr, &ExtSize));

  std::string ExtStr(ExtSize, '\0');
  CL_RETURN_ON_FAILURE(clGetDeviceInfo(Dev, CL_DEVICE_EXTENSIONS, ExtSize,
                                       ExtStr.data(), nullptr));

  std::string SupportedExtensions(ExtStr.c_str());
  if (ExtStr.find("cl_khr_command_buffer_mutable_dispatch") ==
      std::string::npos) {
    return CL_SUCCESS;
  }

  cl_mutable_dispatch_fields_khr MutableCapabilities;
  CL_RETURN_ON_FAILURE(clGetDeviceInfo(
      Dev, CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR,
      sizeof(MutableCapabilities), &MutableCapabilities, nullptr));

  if (!(MutableCapabilities & CL_MUTABLE_DISPATCH_EXEC_INFO_KHR)) {
    return CL_SUCCESS;
  }

  if (MutableCapabilities & CL_MUTABLE_DISPATCH_ARGUMENTS_KHR) {
    UpdateCapabilities |=
        UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_KERNEL_ARGUMENTS;
  }
  if (MutableCapabilities & CL_MUTABLE_DISPATCH_GLOBAL_SIZE_KHR) {
    UpdateCapabilities |=
        UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_GLOBAL_WORK_SIZE;
  }
  if (MutableCapabilities & CL_MUTABLE_DISPATCH_LOCAL_SIZE_KHR) {
    UpdateCapabilities |=
        UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_LOCAL_WORK_SIZE;
  }
  if (MutableCapabilities & CL_MUTABLE_DISPATCH_GLOBAL_OFFSET_KHR) {
    UpdateCapabilities |=
        UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_GLOBAL_WORK_OFFSET;
  }

  return CL_SUCCESS;
}
