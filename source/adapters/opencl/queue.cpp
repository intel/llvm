//===--------- memory.cpp - OpenCL Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "common.hpp"
#include "platform.hpp"

cl_command_queue_info mapURQueueInfoToCL(const ur_queue_info_t PropName) {

  switch (PropName) {
  case UR_QUEUE_INFO_CONTEXT:
    return CL_QUEUE_CONTEXT;
  case UR_QUEUE_INFO_DEVICE:
    return CL_QUEUE_DEVICE;
  case UR_QUEUE_INFO_DEVICE_DEFAULT:
    return CL_QUEUE_DEVICE_DEFAULT;
  case UR_QUEUE_INFO_FLAGS:
    return CL_QUEUE_PROPERTIES;
  case UR_QUEUE_INFO_REFERENCE_COUNT:
    return CL_QUEUE_REFERENCE_COUNT;
  case UR_QUEUE_INFO_SIZE:
    return CL_QUEUE_SIZE;
  default:
    return -1;
  }
}

cl_command_queue_properties
convertURQueuePropertiesToCL(const ur_queue_properties_t *URQueueProperties) {
  cl_command_queue_properties CLCommandQueueProperties = 0;

  if (URQueueProperties->flags & UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
    CLCommandQueueProperties |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
  }
  if (URQueueProperties->flags & UR_QUEUE_FLAG_PROFILING_ENABLE) {
    CLCommandQueueProperties |= CL_QUEUE_PROFILING_ENABLE;
  }
  if (URQueueProperties->flags & UR_QUEUE_FLAG_ON_DEVICE) {
    CLCommandQueueProperties |= CL_QUEUE_ON_DEVICE;
  }
  if (URQueueProperties->flags & UR_QUEUE_FLAG_ON_DEVICE_DEFAULT) {
    CLCommandQueueProperties |= CL_QUEUE_ON_DEVICE_DEFAULT;
  }

  return CLCommandQueueProperties;
}

ur_queue_flags_t
mapCLQueuePropsToUR(const cl_command_queue_properties &Properties) {
  ur_queue_flags_t Flags = 0;
  if (Properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
    Flags |= UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE;
  }
  if (Properties & CL_QUEUE_PROFILING_ENABLE) {
    Flags |= UR_QUEUE_FLAG_PROFILING_ENABLE;
  }
  if (Properties & CL_QUEUE_ON_DEVICE) {
    Flags |= UR_QUEUE_FLAG_ON_DEVICE;
  }
  if (Properties & CL_QUEUE_ON_DEVICE_DEFAULT) {
    Flags |= UR_QUEUE_FLAG_ON_DEVICE_DEFAULT;
  }
  return Flags;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueCreate(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_queue_properties_t *pProperties, ur_queue_handle_t *phQueue) {

  cl_platform_id CurPlatform;
  CL_RETURN_ON_FAILURE_AND_SET_NULL(
      clGetDeviceInfo(cl_adapter::cast<cl_device_id>(hDevice),
                      CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &CurPlatform,
                      nullptr),
      phQueue);

  cl_command_queue_properties CLProperties =
      pProperties ? convertURQueuePropertiesToCL(pProperties) : 0;

  // Properties supported by OpenCL backend.
  const cl_command_queue_properties SupportByOpenCL =
      CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE |
      CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT;

  oclv::OpenCLVersion Version;
  CL_RETURN_ON_FAILURE_AND_SET_NULL(
      cl_adapter::getPlatformVersion(CurPlatform, Version), phQueue);

  cl_int RetErr = CL_INVALID_OPERATION;

  if (Version < oclv::V2_0) {
    *phQueue = cl_adapter::cast<ur_queue_handle_t>(
        clCreateCommandQueue(cl_adapter::cast<cl_context>(hContext),
                             cl_adapter::cast<cl_device_id>(hDevice),
                             CLProperties & SupportByOpenCL, &RetErr));
    CL_RETURN_ON_FAILURE(RetErr);
    return UR_RESULT_SUCCESS;
  }

  /* TODO: Add support for CL_QUEUE_PRIORITY_KHR */
  cl_queue_properties CreationFlagProperties[] = {
      CL_QUEUE_PROPERTIES, CLProperties & SupportByOpenCL, 0};
  *phQueue =
      cl_adapter::cast<ur_queue_handle_t>(clCreateCommandQueueWithProperties(
          cl_adapter::cast<cl_context>(hContext),
          cl_adapter::cast<cl_device_id>(hDevice), CreationFlagProperties,
          &RetErr));
  CL_RETURN_ON_FAILURE(RetErr);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueGetInfo(ur_queue_handle_t hQueue,
                                                   ur_queue_info_t propName,
                                                   size_t propSize,
                                                   void *pPropValue,
                                                   size_t *pPropSizeRet) {
  if (propName == UR_QUEUE_INFO_EMPTY) {
    // OpenCL doesn't provide API to check the status of the queue.
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }
  cl_command_queue_info CLCommandQueueInfo = mapURQueueInfoToCL(propName);

  // Unfortunately the size of cl_bitfield (unsigned long) doesn't line up with
  // our enums (forced to be sizeof(uint32_t)) so this needs special handling.
  if (propName == UR_QUEUE_INFO_FLAGS) {
    UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

    cl_command_queue_properties QueueProperties = 0;
    CL_RETURN_ON_FAILURE(clGetCommandQueueInfo(
        cl_adapter::cast<cl_command_queue>(hQueue), CLCommandQueueInfo,
        sizeof(QueueProperties), &QueueProperties, nullptr));

    return ReturnValue(mapCLQueuePropsToUR(QueueProperties));
  } else {
    size_t CheckPropSize = 0;
    cl_int RetErr = clGetCommandQueueInfo(
        cl_adapter::cast<cl_command_queue>(hQueue), CLCommandQueueInfo,
        propSize, pPropValue, &CheckPropSize);
    if (pPropValue && CheckPropSize != propSize) {
      return UR_RESULT_ERROR_INVALID_SIZE;
    }
    CL_RETURN_ON_FAILURE(RetErr);
    if (pPropSizeRet) {
      *pPropSizeRet = CheckPropSize;
    }
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urQueueGetNativeHandle(ur_queue_handle_t hQueue, ur_queue_native_desc_t *,
                       ur_native_handle_t *phNativeQueue) {
  return getNativeHandle(hQueue, phNativeQueue);
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueCreateWithNativeHandle(
    ur_native_handle_t hNativeQueue,
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] const ur_queue_native_properties_t *pProperties,
    ur_queue_handle_t *phQueue) {

  *phQueue = reinterpret_cast<ur_queue_handle_t>(hNativeQueue);
  cl_int RetErr =
      clRetainCommandQueue(cl_adapter::cast<cl_command_queue>(hNativeQueue));
  CL_RETURN_ON_FAILURE(RetErr);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueFinish(ur_queue_handle_t hQueue) {
  cl_int RetErr = clFinish(cl_adapter::cast<cl_command_queue>(hQueue));
  CL_RETURN_ON_FAILURE(RetErr);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueFlush(ur_queue_handle_t hQueue) {
  cl_int RetErr = clFinish(cl_adapter::cast<cl_command_queue>(hQueue));
  CL_RETURN_ON_FAILURE(RetErr);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueRetain(ur_queue_handle_t hQueue) {
  cl_int RetErr =
      clRetainCommandQueue(cl_adapter::cast<cl_command_queue>(hQueue));
  CL_RETURN_ON_FAILURE(RetErr);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueRelease(ur_queue_handle_t hQueue) {
  cl_int RetErr =
      clReleaseCommandQueue(cl_adapter::cast<cl_command_queue>(hQueue));
  CL_RETURN_ON_FAILURE(RetErr);
  return UR_RESULT_SUCCESS;
}
