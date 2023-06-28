//===--------- platform.cpp - Native CPU Adapter ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "platform.hpp"
#include "common.hpp"

#include "ur/ur.hpp"
#include "ur_api.h"

#include <iostream>

UR_APIEXPORT ur_result_t UR_APICALL
urPlatformGet(uint32_t NumEntries, ur_platform_handle_t *phPlatforms,
              uint32_t *pNumPlatforms) {

  UR_ASSERT(pNumPlatforms || phPlatforms, UR_RESULT_ERROR_INVALID_VALUE);

  if (pNumPlatforms) {
    *pNumPlatforms = 1;
  }

  if (NumEntries == 0) {
    if (phPlatforms != nullptr) {
      if (PrintTrace) {
        std::cerr << "Invalid argument combination for piPlatformsGet\n";
      }
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
    return UR_RESULT_SUCCESS;
  }
  if (phPlatforms && NumEntries > 0) {
    static ur_platform_handle_t_ ThePlatform;
    *phPlatforms = &ThePlatform;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urPlatformGetInfo(ur_platform_handle_t hPlatform, ur_platform_info_t propName,
                  size_t propSize, void *pParamValue, size_t *pSizeRet) {

  if (hPlatform == nullptr) {
    return UR_RESULT_ERROR_INVALID_PLATFORM;
  }
  UrReturnHelper ReturnValue(propSize, pParamValue, pSizeRet);

  switch (propName) {
  case UR_PLATFORM_INFO_NAME:
    return ReturnValue("SYCL_NATIVE_CPU");

  case UR_PLATFORM_INFO_VENDOR_NAME:
    return ReturnValue("tbd");

  case UR_PLATFORM_INFO_VERSION:
    return ReturnValue("0.1");

  case UR_PLATFORM_INFO_PROFILE:
    return ReturnValue("FULL_PROFILE");

  case UR_PLATFORM_INFO_EXTENSIONS:
    return ReturnValue("");

  case UR_PLATFORM_INFO_BACKEND:
    // TODO(alcpz): PR with this enum value at
    // https://github.com/oneapi-src/unified-runtime
    return ReturnValue(UR_PLATFORM_BACKEND_NATIVE_CPU);
  default:
    DIE_NO_IMPLEMENTATION;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urInit(ur_device_init_flags_t) {
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urTearDown(void *) {
  return UR_RESULT_SUCCESS;
}

// TODO
UR_APIEXPORT ur_result_t UR_APICALL urPlatformGetBackendOption(
    ur_platform_handle_t hPlatform, const char *pFrontendOption,
    const char **ppPlatformOption) {
  std::ignore = hPlatform;
  if (pFrontendOption == nullptr) {
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  }

  std::ignore = ppPlatformOption;
  return UR_RESULT_SUCCESS;
}
