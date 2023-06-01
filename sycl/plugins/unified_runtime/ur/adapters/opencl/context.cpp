//===--------- context.hpp - OpenCL Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "common.hpp"

#include <sycl/detail/cl.h>

UR_APIEXPORT ur_result_t UR_APICALL urContextCreate(
    uint32_t DeviceCount, const ur_device_handle_t *phDevices,
    const ur_context_properties_t *, ur_context_handle_t *phContext) {

  UR_ASSERT(phDevices, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(phContext, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  cl_int ret;
  *phContext = cl_adapter::cast<ur_context_handle_t>(
      clCreateContext(nullptr, cl_adapter::cast<cl_uint>(DeviceCount),
                      cl_adapter::cast<const cl_device_id *>(phDevices),
                      nullptr, nullptr, cl_adapter::cast<cl_int *>(&ret)));

  return map_cl_error_to_ur(ret);
}

cl_int map_ur_context_info_to_cl(ur_context_info_t urPropName) {

  cl_int cl_propName;
  switch (urPropName) {
  case UR_CONTEXT_INFO_NUM_DEVICES:
    cl_propName = CL_CONTEXT_NUM_DEVICES;
    break;
  case UR_CONTEXT_INFO_DEVICES:
    cl_propName = CL_CONTEXT_DEVICES;
    break;
  case UR_CONTEXT_INFO_REFERENCE_COUNT:
    cl_propName = CL_CONTEXT_REFERENCE_COUNT;
    break;
  default:
    cl_propName = -1;
  }

  return cl_propName;
}

UR_APIEXPORT ur_result_t UR_APICALL
urContextGetInfo(ur_context_handle_t hContext, ur_context_info_t propName,
                 size_t propSize, void *pPropValue, size_t *pPropSizeRet) {

  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);
  const cl_int cl_propName = map_ur_context_info_to_cl(propName);

  switch (static_cast<uint32_t>(propName)) {
  /* 2D USM memops are not supported. */
  case UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT:
  case UR_CONTEXT_INFO_USM_FILL2D_SUPPORT: {
    return ReturnValue(false);
  }
  case UR_CONTEXT_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES:
  case UR_CONTEXT_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES:
  case UR_CONTEXT_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES:
  case UR_CONTEXT_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES: {
    /* These queries should be dealt with in context_impl.cpp by calling the
     * queries of each device separately and building the intersection set. */
    cl_adapter::setErrorMessage("These queries should have never come here.",
                                UR_RESULT_ERROR_INVALID_ARGUMENT);
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }
  case UR_CONTEXT_INFO_NUM_DEVICES:
  case UR_CONTEXT_INFO_DEVICES:
  case UR_CONTEXT_INFO_REFERENCE_COUNT: {

    CL_RETURN_ON_FAILURE(
        clGetContextInfo(cl_adapter::cast<cl_context>(hContext), cl_propName,
                         propSize, pPropValue, pPropSizeRet));
    return UR_RESULT_SUCCESS;
  }
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL
urContextRelease(ur_context_handle_t hContext) {

  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  cl_int ret = clReleaseContext(cl_adapter::cast<cl_context>(hContext));
  return map_cl_error_to_ur(ret);
}

UR_APIEXPORT ur_result_t UR_APICALL
urContextRetain(ur_context_handle_t hContext) {

  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  cl_int ret = clRetainContext(cl_adapter::cast<cl_context>(hContext));
  return map_cl_error_to_ur(ret);
}

UR_APIEXPORT ur_result_t UR_APICALL urContextGetNativeHandle(
    ur_context_handle_t hContext, ur_native_handle_t *phNativeContext) {

  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(phNativeContext, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  *phNativeContext = reinterpret_cast<ur_native_handle_t>(hContext);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urContextCreateWithNativeHandle(
    ur_native_handle_t hNativeContext, uint32_t, const ur_device_handle_t *,
    const ur_context_native_properties_t *, ur_context_handle_t *phContext) {

  UR_ASSERT(hNativeContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  *phContext = reinterpret_cast<ur_context_handle_t>(hNativeContext);
  return UR_RESULT_SUCCESS;
}
