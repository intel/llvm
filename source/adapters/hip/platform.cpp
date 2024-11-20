//===--------- platform.cpp - HIP Adapter ---------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "platform.hpp"
#include "context.hpp"

UR_APIEXPORT ur_result_t UR_APICALL
urPlatformGetInfo(ur_platform_handle_t, ur_platform_info_t propName,
                  size_t propSize, void *pPropValue, size_t *pSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pSizeRet);

  switch (propName) {
  case UR_PLATFORM_INFO_NAME:
    return ReturnValue("AMD HIP BACKEND");
  case UR_PLATFORM_INFO_VENDOR_NAME:
    return ReturnValue("AMD Corporation");
  case UR_PLATFORM_INFO_PROFILE:
    return ReturnValue("FULL PROFILE");
  case UR_PLATFORM_INFO_VERSION: {
    std::string Version;
    UR_CHECK_ERROR(getHipVersionString(Version));
    return ReturnValue(Version.c_str());
  }
  case UR_PLATFORM_INFO_BACKEND: {
    return ReturnValue(UR_PLATFORM_BACKEND_HIP);
  }
  case UR_PLATFORM_INFO_EXTENSIONS: {
    return ReturnValue("");
  }
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }

  return UR_RESULT_SUCCESS;
}

/// Obtains the HIP platform.
/// There is only one HIP platform, and contains all devices on the system.
/// Triggers the HIP Driver initialization (hipInit) the first time, so this
/// must be the first UR API called.
UR_APIEXPORT ur_result_t UR_APICALL
urPlatformGet(ur_adapter_handle_t *, uint32_t, uint32_t NumEntries,
              ur_platform_handle_t *phPlatforms, uint32_t *pNumPlatforms) {

  try {
    static std::once_flag InitFlag;
    static uint32_t NumPlatforms = 1;
    static ur_platform_handle_t_ Platform;

    UR_ASSERT(phPlatforms || pNumPlatforms, UR_RESULT_ERROR_INVALID_VALUE);
    UR_ASSERT(!phPlatforms || NumEntries > 0, UR_RESULT_ERROR_INVALID_VALUE);

    ur_result_t Result = UR_RESULT_SUCCESS;

    std::call_once(
        InitFlag,
        [](ur_result_t &Err) {
          if (hipInit(0) != hipSuccess) {
            NumPlatforms = 0;
            return;
          }
          int NumDevices = 0;
          Err = UR_RESULT_SUCCESS;
          UR_CHECK_ERROR(hipGetDeviceCount(&NumDevices));
          if (NumDevices == 0) {
            NumPlatforms = 0;
            return;
          }
          try {
            for (auto i = 0u; i < static_cast<uint32_t>(NumDevices); ++i) {
              hipDevice_t Device;
              UR_CHECK_ERROR(hipDeviceGet(&Device, i));
              UR_CHECK_ERROR(hipSetDevice(i));
              hipEvent_t EvBase;
              UR_CHECK_ERROR(hipEventCreate(&EvBase));

              // Use the default stream to record base event counter
              UR_CHECK_ERROR(hipEventRecord(EvBase, 0));
              Platform.Devices.emplace_back(
                  new ur_device_handle_t_{Device, EvBase, &Platform, i});

              ScopedDevice Active(Platform.Devices.front().get());
            }
          } catch (const std::bad_alloc &) {
            // Signal out-of-memory situation
            Platform.Devices.clear();
            Err = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
          } catch (ur_result_t CatchErr) {
            // Clear and rethrow to allow retry
            Platform.Devices.clear();
            Err = CatchErr;
            throw CatchErr;
          } catch (...) {
            Err = UR_RESULT_ERROR_OUT_OF_RESOURCES;
            throw;
          }
        },
        Result);

    if (pNumPlatforms != nullptr) {
      *pNumPlatforms = NumPlatforms;
    }

    if (phPlatforms != nullptr) {
      *phPlatforms = &Platform;
    }

    return Result;
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL
urPlatformGetApiVersion(ur_platform_handle_t, ur_api_version_t *pVersion) {
  *pVersion = UR_API_VERSION_CURRENT;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urPlatformGetNativeHandle(
    ur_platform_handle_t hPlatform, ur_native_handle_t *phNativePlatform) {
  std::ignore = hPlatform;
  std::ignore = phNativePlatform;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urPlatformCreateWithNativeHandle(
    ur_native_handle_t, ur_adapter_handle_t,
    const ur_platform_native_properties_t *, ur_platform_handle_t *) {
  // There is no HIP equivalent to ur_platform_handle_t
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

// Get CUDA plugin specific backend option.
// Current support is only for optimization options.
// Return empty string for cuda.
// TODO: Determine correct string to be passed.
UR_APIEXPORT ur_result_t UR_APICALL
urPlatformGetBackendOption(ur_platform_handle_t, const char *pFrontendOption,
                           const char **ppPlatformOption) {
  using namespace std::literals;
  if (pFrontendOption == "-O0"sv || pFrontendOption == "-O1"sv ||
      pFrontendOption == "-O2"sv || pFrontendOption == "-O3"sv ||
      pFrontendOption == ""sv) {
    *ppPlatformOption = "";
    return UR_RESULT_SUCCESS;
  }
  if (pFrontendOption == "-foffload-fp32-prec-div"sv ||
      pFrontendOption == "-foffload-fp32-prec-sqrt"sv) {
    *ppPlatformOption = "";
    return UR_RESULT_SUCCESS;
  }
  return UR_RESULT_ERROR_INVALID_VALUE;
}
