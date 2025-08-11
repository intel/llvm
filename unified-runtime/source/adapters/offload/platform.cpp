//===----------- platform.cpp - LLVM Offload Adapter  ---------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>
#include <unordered_set>
#include <ur/ur.hpp>
#include <ur_api.h>

#include "adapter.hpp"
#include "device.hpp"
#include "ur2offload.hpp"

UR_APIEXPORT ur_result_t UR_APICALL
urPlatformGet(ur_adapter_handle_t, uint32_t NumEntries,
              ur_platform_handle_t *phPlatforms, uint32_t *pNumPlatforms) {

  if (pNumPlatforms) {
    *pNumPlatforms = Adapter->Platforms.size();
  }

  if (phPlatforms) {
    size_t PlatformIndex = 0;
    for (auto &Platform : Adapter->Platforms) {
      phPlatforms[PlatformIndex++] = Platform.get();
      if (PlatformIndex == NumEntries) {
        break;
      }
    }
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urPlatformGetInfo(ur_platform_handle_t hPlatform, ur_platform_info_t propName,
                  size_t propSize, void *pPropValue, size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  ol_platform_info_t olInfo;
  switch (propName) {
  case UR_PLATFORM_INFO_NAME:
    olInfo = OL_PLATFORM_INFO_NAME;
    break;
  case UR_PLATFORM_INFO_VENDOR_NAME:
    olInfo = OL_PLATFORM_INFO_VENDOR_NAME;
    break;
  case UR_PLATFORM_INFO_VERSION:
    olInfo = OL_PLATFORM_INFO_VERSION;
    break;
  case UR_PLATFORM_INFO_EXTENSIONS:
    return ReturnValue("");
  case UR_PLATFORM_INFO_PROFILE:
    return ReturnValue("FULL_PROFILE");
  case UR_PLATFORM_INFO_BACKEND:
    return ReturnValue(UR_BACKEND_OFFLOAD);
  case UR_PLATFORM_INFO_ADAPTER:
    return ReturnValue(Adapter);
    break;
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }

  if (pPropSizeRet) {
    OL_RETURN_ON_ERR(olGetPlatformInfoSize(hPlatform->OffloadPlatform, olInfo,
                                           pPropSizeRet));
  }

  if (pPropValue) {
    OL_RETURN_ON_ERR(olGetPlatformInfo(hPlatform->OffloadPlatform, olInfo,
                                       propSize, pPropValue));
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urPlatformGetBackendOption(ur_platform_handle_t, const char *pFrontendOption,
                           const char **ppPlatformOption) {
  using namespace std::literals;
  if (pFrontendOption == nullptr)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  if (pFrontendOption == "-O0"sv || pFrontendOption == "-O1"sv ||
      pFrontendOption == "-O2"sv || pFrontendOption == "-O3"sv ||
      pFrontendOption == ""sv) {
    *ppPlatformOption = "";
    return UR_RESULT_SUCCESS;
  }
  return UR_RESULT_ERROR_INVALID_VALUE;
}

UR_APIEXPORT ur_result_t UR_APICALL urPlatformGetNativeHandle(
    ur_platform_handle_t hAdapter, ur_native_handle_t *phNativePlatform) {
  *phNativePlatform =
      reinterpret_cast<ur_native_handle_t>(hAdapter->OffloadPlatform);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urPlatformCreateWithNativeHandle(
    ur_native_handle_t hNativePlatform, ur_adapter_handle_t hAdapter,
    const ur_platform_native_properties_t *, ur_platform_handle_t *phPlatform) {

  auto Found = std::find_if(
      hAdapter->Platforms.begin(), hAdapter->Platforms.end(), [&](auto &P) {
        return P->OffloadPlatform ==
               reinterpret_cast<ol_platform_handle_t>(hNativePlatform);
      });
  if (Found != hAdapter->Platforms.end()) {
    *phPlatform = Found->get();
    return UR_RESULT_SUCCESS;
  }

  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urPlatformGetApiVersion(ur_platform_handle_t, ur_api_version_t *pVersion) {
  *pVersion = UR_API_VERSION_CURRENT;
  return UR_RESULT_SUCCESS;
}
