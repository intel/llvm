//===--------- platform.cpp - HIP Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "platform.hpp"

hipEvent_t ur_platform_handle_t_::EvBase{nullptr};

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
    detail::ur::assertion(getHipVersionString(Version) == hipSuccess);
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
///
/// However because multiple devices in a context is not currently supported,
/// place each device in a separate platform.
UR_APIEXPORT ur_result_t UR_APICALL
urPlatformGet(uint32_t NumEntries, ur_platform_handle_t *phPlatforms,
              uint32_t *pNumPlatforms) {

  try {
    static std::once_flag InitFlag;
    static uint32_t NumPlatforms = 1;
    static std::vector<ur_platform_handle_t_> PlatformIds;

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
          Err = UR_CHECK_ERROR(hipGetDeviceCount(&NumDevices));
          if (NumDevices == 0) {
            NumPlatforms = 0;
            return;
          }
          try {
            // make one platform per device
            NumPlatforms = NumDevices;
            PlatformIds.resize(NumDevices);

            for (int i = 0; i < NumDevices; ++i) {
              hipDevice_t Device;
              Err = UR_CHECK_ERROR(hipDeviceGet(&Device, i));
              PlatformIds[i].Devices.emplace_back(
                  new ur_device_handle_t_{Device, &PlatformIds[i]});
            }
          } catch (const std::bad_alloc &) {
            // Signal out-of-memory situation
            for (int i = 0; i < NumDevices; ++i) {
              PlatformIds[i].Devices.clear();
            }
            PlatformIds.clear();
            Err = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
          } catch (...) {
            // Clear and rethrow to allow retry
            for (int i = 0; i < NumDevices; ++i) {
              PlatformIds[i].Devices.clear();
            }
            PlatformIds.clear();
            throw;
          }
        },
        Result);

    if (pNumPlatforms != nullptr) {
      *pNumPlatforms = NumPlatforms;
    }

    if (phPlatforms != nullptr) {
      for (unsigned i = 0; i < std::min(NumEntries, NumPlatforms); ++i) {
        phPlatforms[i] = &PlatformIds[i];
      }
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

UR_APIEXPORT ur_result_t UR_APICALL urInit(ur_device_init_flags_t) {
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urTearDown(void *) {
  return UR_RESULT_SUCCESS;
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
  return UR_RESULT_ERROR_INVALID_VALUE;
}
