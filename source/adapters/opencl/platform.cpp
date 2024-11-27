//===--------- platform.cpp - OpenCL Adapter ---------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "platform.hpp"

ur_result_t cl_adapter::getPlatformVersion(cl_platform_id Plat,
                                           oclv::OpenCLVersion &Version) {

  size_t PlatVerSize = 0;
  CL_RETURN_ON_FAILURE(
      clGetPlatformInfo(Plat, CL_PLATFORM_VERSION, 0, nullptr, &PlatVerSize));

  std::string PlatVer(PlatVerSize, '\0');
  CL_RETURN_ON_FAILURE(clGetPlatformInfo(Plat, CL_PLATFORM_VERSION, PlatVerSize,
                                         PlatVer.data(), nullptr));

  Version = oclv::OpenCLVersion(PlatVer);
  if (!Version.isValid()) {
    return UR_RESULT_ERROR_INVALID_PLATFORM;
  }

  return UR_RESULT_SUCCESS;
}

static cl_int mapURPlatformInfoToCL(ur_platform_info_t URPropName) {

  switch (URPropName) {
  case UR_PLATFORM_INFO_NAME:
    return CL_PLATFORM_NAME;
  case UR_PLATFORM_INFO_VENDOR_NAME:
    return CL_PLATFORM_VENDOR;
  case UR_PLATFORM_INFO_VERSION:
    return CL_PLATFORM_VERSION;
  case UR_PLATFORM_INFO_EXTENSIONS:
    return CL_PLATFORM_EXTENSIONS;
  case UR_PLATFORM_INFO_PROFILE:
    return CL_PLATFORM_PROFILE;
  default:
    return -1;
  }
}

UR_DLLEXPORT ur_result_t UR_APICALL
urPlatformGetInfo(ur_platform_handle_t hPlatform, ur_platform_info_t propName,
                  size_t propSize, void *pPropValue, size_t *pSizeRet) {

  UrReturnHelper ReturnValue(propSize, pPropValue, pSizeRet);
  const cl_int CLPropName = mapURPlatformInfoToCL(propName);

  switch (static_cast<uint32_t>(propName)) {
  case UR_PLATFORM_INFO_BACKEND:
    return ReturnValue(UR_PLATFORM_BACKEND_OPENCL);
  case UR_PLATFORM_INFO_NAME:
  case UR_PLATFORM_INFO_VENDOR_NAME:
  case UR_PLATFORM_INFO_VERSION:
  case UR_PLATFORM_INFO_EXTENSIONS:
  case UR_PLATFORM_INFO_PROFILE: {
    CL_RETURN_ON_FAILURE(
        clGetPlatformInfo(cl_adapter::cast<cl_platform_id>(hPlatform),
                          CLPropName, propSize, pPropValue, pSizeRet));
    return UR_RESULT_SUCCESS;
  }
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }
}

UR_DLLEXPORT ur_result_t UR_APICALL
urPlatformGetApiVersion([[maybe_unused]] ur_platform_handle_t hPlatform,
                        ur_api_version_t *pVersion) {
  *pVersion = UR_API_VERSION_CURRENT;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urPlatformGet(ur_adapter_handle_t *, uint32_t, uint32_t NumEntries,
              ur_platform_handle_t *phPlatforms, uint32_t *pNumPlatforms) {

  cl_int Result =
      clGetPlatformIDs(cl_adapter::cast<cl_uint>(NumEntries),
                       cl_adapter::cast<cl_platform_id *>(phPlatforms),
                       cl_adapter::cast<cl_uint *>(pNumPlatforms));

  /* Absorb the CL_PLATFORM_NOT_FOUND_KHR and just return 0 in num_platforms */
  if (Result == CL_PLATFORM_NOT_FOUND_KHR) {
    Result = CL_SUCCESS;
    if (pNumPlatforms) {
      *pNumPlatforms = 0;
    }
  }

  /* INVALID_VALUE is returned when the size is invalid, special case it here */
  if (Result == CL_INVALID_VALUE && phPlatforms != nullptr && NumEntries == 0) {
    return UR_RESULT_ERROR_INVALID_SIZE;
  }

  return mapCLErrorToUR(Result);
}

UR_APIEXPORT ur_result_t UR_APICALL urPlatformGetNativeHandle(
    ur_platform_handle_t hPlatform, ur_native_handle_t *phNativePlatform) {
  *phNativePlatform = reinterpret_cast<ur_native_handle_t>(hPlatform);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urPlatformCreateWithNativeHandle(
    ur_native_handle_t hNativePlatform, ur_adapter_handle_t,
    const ur_platform_native_properties_t *, ur_platform_handle_t *phPlatform) {
  *phPlatform = reinterpret_cast<ur_platform_handle_t>(hNativePlatform);
  return UR_RESULT_SUCCESS;
}

// Returns plugin specific backend option.
// Current support is only for optimization options.
// Return '-cl-opt-disable' for pFrontendOption = -O0 and '' for others.
UR_APIEXPORT ur_result_t UR_APICALL
urPlatformGetBackendOption(ur_platform_handle_t, const char *pFrontendOption,
                           const char **ppPlatformOption) {
  using namespace std::literals;
  if (pFrontendOption == nullptr)
    return UR_RESULT_SUCCESS;
  if (pFrontendOption == ""sv) {
    *ppPlatformOption = "";
    return UR_RESULT_SUCCESS;
  }
  // Return '-cl-opt-disable' for frontend_option = -O0 and '' for others.
  if (!strcmp(pFrontendOption, "-O0")) {
    *ppPlatformOption = "-cl-opt-disable";
    return UR_RESULT_SUCCESS;
  }
  if (pFrontendOption == "-O1"sv || pFrontendOption == "-O2"sv ||
      pFrontendOption == "-O3"sv) {
    *ppPlatformOption = "";
    return UR_RESULT_SUCCESS;
  }
  if (pFrontendOption == "-ftarget-compile-fast"sv) {
    *ppPlatformOption = "-igc_opts 'PartitionUnit=1,SubroutineThreshold=50000'";
    return UR_RESULT_SUCCESS;
  }
  if (pFrontendOption == "-foffload-fp32-prec-div"sv ||
      pFrontendOption == "-foffload-fp32-prec-sqrt"sv) {
    *ppPlatformOption = "-cl-fp32-correctly-rounded-divide-sqrt";
    return UR_RESULT_SUCCESS;
  }
  return UR_RESULT_ERROR_INVALID_VALUE;
}
