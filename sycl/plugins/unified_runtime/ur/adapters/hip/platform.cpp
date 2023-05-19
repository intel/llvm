//===--------- platform.cpp - HIP Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "platform.hpp"

hipEvent_t ur_platform_handle_t_::evBase_{nullptr};

UR_DLLEXPORT ur_result_t UR_APICALL urPlatformGetInfo(
    ur_platform_handle_t hPlatform, ur_platform_info_t PlatformInfoType,
    size_t Size, void *pPlatformInfo, size_t *pSizeRet) {

  UR_ASSERT(hPlatform, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UrReturnHelper ReturnValue(Size, pPlatformInfo, pSizeRet);

  switch (PlatformInfoType) {
  case UR_PLATFORM_INFO_NAME:
    return ReturnValue("AMD HIP BACKEND");
  case UR_PLATFORM_INFO_VENDOR_NAME:
    return ReturnValue("AMD Corporation");
  case UR_PLATFORM_INFO_PROFILE:
    return ReturnValue("FULL PROFILE");
  case UR_PLATFORM_INFO_VERSION: {
    auto version = getHipVersionString();
    return ReturnValue(version.c_str());
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
///
/// However because multiple devices in a context is not currently supported,
/// place each device in a separate platform.
///
UR_DLLEXPORT ur_result_t UR_APICALL
urPlatformGet(uint32_t NumEntries, ur_platform_handle_t *phPlatforms,
              uint32_t *pNumPlatforms) {

  try {
    static std::once_flag initFlag;
    static uint32_t numPlatforms = 1;
    static std::vector<ur_platform_handle_t_> platformIds;

    UR_ASSERT(phPlatforms || pNumPlatforms, UR_RESULT_ERROR_INVALID_VALUE);
    UR_ASSERT(!phPlatforms || NumEntries > 0, UR_RESULT_ERROR_INVALID_VALUE);

    ur_result_t err = UR_RESULT_SUCCESS;

    std::call_once(
        initFlag,
        [](ur_result_t &err) {
          if (hipInit(0) != hipSuccess) {
            numPlatforms = 0;
            return;
          }
          int numDevices = 0;
          err = UR_CHECK_ERROR(hipGetDeviceCount(&numDevices));
          if (numDevices == 0) {
            numPlatforms = 0;
            return;
          }
          try {
            // make one platform per device
            numPlatforms = numDevices;
            platformIds.resize(numDevices);

            for (int i = 0; i < numDevices; ++i) {
              hipDevice_t device;
              err = UR_CHECK_ERROR(hipDeviceGet(&device, i));
              platformIds[i].devices_.emplace_back(
                  new ur_device_handle_t_{device, &platformIds[i]});
            }
          } catch (const std::bad_alloc &) {
            // Signal out-of-memory situation
            for (int i = 0; i < numDevices; ++i) {
              platformIds[i].devices_.clear();
            }
            platformIds.clear();
            err = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
          } catch (...) {
            // Clear and rethrow to allow retry
            for (int i = 0; i < numDevices; ++i) {
              platformIds[i].devices_.clear();
            }
            platformIds.clear();
            throw;
          }
        },
        err);

    if (pNumPlatforms != nullptr) {
      *pNumPlatforms = numPlatforms;
    }

    if (phPlatforms != nullptr) {
      for (unsigned i = 0; i < std::min(NumEntries, numPlatforms); ++i) {
        phPlatforms[i] = &platformIds[i];
      }
    }

    return err;
  } catch (ur_result_t err) {
    return err;
  } catch (...) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }
}

UR_DLLEXPORT ur_result_t UR_APICALL urPlatformGetApiVersion(
    ur_platform_handle_t hDriver, ur_api_version_t *pVersion) {
  UR_ASSERT(hDriver, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pVersion, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  *pVersion = UR_API_VERSION_CURRENT;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urInit(ur_device_init_flags_t) {
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urTearDown(void *) {
  return UR_RESULT_SUCCESS;
}

// Returns plugin specific backend option.
// Current support is only for optimization options.
// Return empty string for cuda.
// TODO: Determine correct string to be passed.
UR_APIEXPORT ur_result_t UR_APICALL urPlatformGetBackendOption(
    ur_platform_handle_t hPlatform, const char *pFrontendOption,
    const char **ppPlatformOption) {
  (void)hPlatform;
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
