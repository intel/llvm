//===--------- platform.hpp - OpenCL Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "common.hpp"

#include <sycl/detail/cl.h>

cl_int map_ur_platform_info_to_cl(ur_platform_info_t urPropName) {

  cl_int cl_propName;
  switch (urPropName) {
  case UR_PLATFORM_INFO_NAME:
    cl_propName = CL_PLATFORM_NAME;
    break;
  case UR_PLATFORM_INFO_VENDOR_NAME:
    cl_propName = CL_PLATFORM_VENDOR;
    break;
  case UR_PLATFORM_INFO_VERSION:
    cl_propName = CL_PLATFORM_VERSION;
    break;
  case UR_PLATFORM_INFO_EXTENSIONS:
    cl_propName = CL_PLATFORM_EXTENSIONS;
    break;
  case UR_PLATFORM_INFO_PROFILE:
    cl_propName = CL_PLATFORM_PROFILE;
    break;
  default:
    cl_propName = -1;
  }

  return cl_propName;
}

UR_DLLEXPORT ur_result_t UR_APICALL
urPlatformGetInfo(ur_platform_handle_t hPlatform, ur_platform_info_t propName,
                  size_t propSize, void *pPropValue, size_t *pSizeRet) {

  UR_ASSERT(hPlatform, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UrReturnHelper ReturnValue(propSize, pPropValue, pSizeRet);
  const cl_int cl_propName = map_ur_platform_info_to_cl(propName);

  switch (static_cast<uint32_t>(propName)) {
  case UR_PLATFORM_INFO_BACKEND:
    return ReturnValue(UR_PLATFORM_BACKEND_OPENCL);
  case UR_PLATFORM_INFO_NAME:
  case UR_PLATFORM_INFO_VENDOR_NAME:
  case UR_PLATFORM_INFO_VERSION:
  case UR_PLATFORM_INFO_EXTENSIONS:
  case UR_PLATFORM_INFO_PROFILE: {
    CL_RETURN_ON_FAILURE(clGetPlatformInfo(cl::cast<cl_platform_id>(hPlatform),
                                           cl_propName, propSize, pPropValue,
                                           pSizeRet));
    return UR_RESULT_SUCCESS;
  }
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }
}

UR_DLLEXPORT ur_result_t UR_APICALL urPlatformGetApiVersion(
    ur_platform_handle_t hPlatform, ur_api_version_t *pVersion) {
  UR_ASSERT(hPlatform, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pVersion, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  *pVersion = UR_API_VERSION_CURRENT;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL
urPlatformGet(uint32_t NumEntries, ur_platform_handle_t *phPlatforms,
              uint32_t *pNumPlatforms) {

  UR_ASSERT(phPlatforms || pNumPlatforms, UR_RESULT_ERROR_INVALID_VALUE);
  UR_ASSERT(!phPlatforms || NumEntries > 0, UR_RESULT_ERROR_INVALID_SIZE);

  cl_int result = clGetPlatformIDs(cl::cast<cl_uint>(NumEntries),
                                   cl::cast<cl_platform_id *>(phPlatforms),
                                   cl::cast<cl_uint *>(pNumPlatforms));

  /* Absorb the CL_PLATFORM_NOT_FOUND_KHR and just return 0 in num_platforms */
  if (result == CL_PLATFORM_NOT_FOUND_KHR) {
    result = CL_SUCCESS;
    if (pNumPlatforms) {
      *pNumPlatforms = 0;
    }
  }

  return map_cl_error_to_ur(result);
}

UR_APIEXPORT ur_result_t UR_APICALL urPlatformGetNativeHandle(
    ur_platform_handle_t hPlatform, ur_native_handle_t *phNativePlatform) {

  UR_ASSERT(hPlatform, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(phNativePlatform, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  *phNativePlatform = reinterpret_cast<ur_native_handle_t>(hPlatform);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urPlatformCreateWithNativeHandle(
    ur_native_handle_t hNativePlatform, ur_platform_handle_t *phPlatform) {

  UR_ASSERT(hNativePlatform, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  *phPlatform = reinterpret_cast<ur_platform_handle_t>(hNativePlatform);
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urInit(ur_device_init_flags_t) {
  return UR_RESULT_SUCCESS;
}

// This API is called by Sycl RT to notify the end of the plugin lifetime.
// Windows: dynamically loaded plugins might have been unloaded already
// when this is called. Sycl RT holds onto the PI plugin so it can be
// called safely. But this is not transitive. If the PI plugin in turn
// dynamically loaded a different DLL, that may have been unloaded.
// TODO: add a global variable lifetime management code here (see
// pi_level_zero.cpp for reference).
UR_DLLEXPORT ur_result_t UR_APICALL urTearDown(void *pParams) {
  UR_ASSERT(pParams, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  if (cl_ext::ExtFuncPtrCache) {
    delete cl_ext::ExtFuncPtrCache;
    cl_ext::ExtFuncPtrCache = nullptr;
  }
  return UR_RESULT_SUCCESS;
}
