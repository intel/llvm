//===--------- memory.cpp - OpenCL Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "common.hpp"
#include "platform.hpp"

#include <sycl/detail/cl.h>

cl_command_queue_info map_ur_queue_info_to_cl(const ur_queue_info_t propName) {
  switch (propName) {
  case UR_QUEUE_INFO_CONTEXT:
    return CL_QUEUE_CONTEXT;
    break;
  case UR_QUEUE_INFO_DEVICE:
    return CL_QUEUE_DEVICE;
    break;
  case UR_QUEUE_INFO_DEVICE_DEFAULT:
    return CL_QUEUE_DEVICE_DEFAULT;
    break;
  case UR_QUEUE_INFO_FLAGS:
    return CL_QUEUE_PROPERTIES_ARRAY;
    break;
  case UR_QUEUE_INFO_REFERENCE_COUNT:
    return CL_QUEUE_REFERENCE_COUNT;
    break;
  case UR_QUEUE_INFO_SIZE:
    return CL_QUEUE_SIZE;
    break;
  default:
    return -1;
    break;
  }
}

cl_command_queue_properties convert_ur_queue_properties_to_cl(
    const ur_queue_properties_t *urQueueProperties) {
  cl_command_queue_properties clCommandQueueProperties = 0;

  if (urQueueProperties->flags & UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
    clCommandQueueProperties |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
  }
  if (urQueueProperties->flags & UR_QUEUE_FLAG_PROFILING_ENABLE) {
    clCommandQueueProperties |= CL_QUEUE_PROFILING_ENABLE;
  }
  if (urQueueProperties->flags & UR_QUEUE_FLAG_ON_DEVICE) {
    clCommandQueueProperties |= CL_QUEUE_ON_DEVICE;
  }
  if (urQueueProperties->flags & UR_QUEUE_FLAG_ON_DEVICE_DEFAULT) {
    clCommandQueueProperties |= CL_QUEUE_ON_DEVICE_DEFAULT;
  }
  if (urQueueProperties->flags & UR_QUEUE_FLAG_PRIORITY_LOW) {
    clCommandQueueProperties |= CL_QUEUE_PRIORITY_LOW_KHR;
  }
  if (urQueueProperties->flags & UR_QUEUE_FLAG_PRIORITY_HIGH) {
    clCommandQueueProperties |= CL_QUEUE_PRIORITY_HIGH_KHR;
  }

  return clCommandQueueProperties;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueCreate(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_queue_properties_t *pProperties, ur_queue_handle_t *phQueue) {
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  cl_platform_id curPlatform;
  CL_RETURN_ON_FAILURE_AND_SET_NULL(
      clGetDeviceInfo(cl_adapter::cast<cl_device_id>(hDevice),
                      CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &curPlatform,
                      nullptr),
      phQueue);

  cl_command_queue_properties clProperties =
      convert_ur_queue_properties_to_cl(pProperties);

  // Check that unexpected bits are not set.
  assert(!(clProperties &
           ~(CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
             CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_ON_DEVICE |
             CL_QUEUE_ON_DEVICE_DEFAULT)));

  // Properties supported by OpenCL backend.
  cl_command_queue_properties SupportByOpenCL =
      CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE |
      CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT;
  
  OCLV::OpenCLVersion version;
  CL_RETURN_ON_FAILURE_AND_SET_NULL(cl_adapter::getPlatformVersion(curPlatform, version),
                                    phQueue);

  cl_int ret_err = CL_INVALID_OPERATION;

  if (version >= OCLV::V2_0) {
    *phQueue = cl_adapter::cast<ur_queue_handle_t>(
        clCreateCommandQueue(cl_adapter::cast<cl_context>(hContext),
                             cl_adapter::cast<cl_device_id>(hDevice),
                             clProperties & SupportByOpenCL, &ret_err));
    CL_RETURN_ON_FAILURE(ret_err);
    return UR_RESULT_SUCCESS;
  }

  cl_queue_properties CreationFlagProperties[] = {
      CL_QUEUE_PROPERTIES, clProperties & SupportByOpenCL, 0};
  *phQueue =
      cl_adapter::cast<ur_queue_handle_t>(clCreateCommandQueueWithProperties(
          cl_adapter::cast<cl_context>(hContext),
          cl_adapter::cast<cl_device_id>(hDevice), CreationFlagProperties,
          &ret_err));
  CL_RETURN_ON_FAILURE(ret_err);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueGetInfo(ur_queue_handle_t hQueue,
                                                   ur_queue_info_t propName,
                                                   size_t propSize,
                                                   void *pPropValue,
                                                   size_t *pPropSizeRet) {
  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  if (propName == UR_QUEUE_INFO_EMPTY) {
    // OpenCL doesn't provide API to check the status of the queue.
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  cl_command_queue_info clCommandQueueInfo = map_ur_queue_info_to_cl(propName);

  cl_int ret_err = clGetCommandQueueInfo(
      cl_adapter::cast<cl_command_queue>(hQueue), clCommandQueueInfo, propSize,
      pPropValue, pPropSizeRet);
  CL_RETURN_ON_FAILURE(ret_err);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueGetNativeHandle(
    ur_queue_handle_t hQueue, ur_native_handle_t *phNativeQueue) {
  return urGetNativeHandle(hQueue, phNativeQueue);
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueCreateWithNativeHandle(
    ur_native_handle_t hNativeQueue, ur_context_handle_t hContext,
    ur_device_handle_t hDevice, const ur_queue_native_properties_t *pProperties,
    ur_queue_handle_t *phQueue) {
  UR_ASSERT(hNativeQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(phQueue, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  (void)hContext;
  (void)hDevice;
  (void)pProperties;
  *phQueue = reinterpret_cast<ur_queue_handle_t>(hNativeQueue);
  cl_int ret_err =
      clRetainCommandQueue(cl_adapter::cast<cl_command_queue>(hNativeQueue));
  CL_RETURN_ON_FAILURE(ret_err);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueFinish(ur_queue_handle_t hQueue) {
  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  cl_int ret_err = clFinish(cl_adapter::cast<cl_command_queue>(hQueue));
  CL_RETURN_ON_FAILURE(ret_err);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueFlush(ur_queue_handle_t hQueue) {
  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  cl_int ret_err = clFinish(cl_adapter::cast<cl_command_queue>(hQueue));
  CL_RETURN_ON_FAILURE(ret_err);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueRetain(ur_queue_handle_t hQueue) {
  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  cl_int ret_err =
      clRetainCommandQueue(cl_adapter::cast<cl_command_queue>(hQueue));
  CL_RETURN_ON_FAILURE(ret_err);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueRelease(ur_queue_handle_t hQueue) {
  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  cl_int ret_err =
      clReleaseCommandQueue(cl_adapter::cast<cl_command_queue>(hQueue));
  CL_RETURN_ON_FAILURE(ret_err);
  return UR_RESULT_SUCCESS;
}
