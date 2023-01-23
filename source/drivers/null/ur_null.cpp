/*
 *
 * Copyright (C) 2019-2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file ur_null.cpp
 *
 */
#include "ur_null.h"

namespace driver {
//////////////////////////////////////////////////////////////////////////
context_t d_context;

//////////////////////////////////////////////////////////////////////////
context_t::context_t() {
  //////////////////////////////////////////////////////////////////////////
  urDdiTable.Platform.pfnGet = [](uint32_t NumEntries,
                                  ur_platform_handle_t *phPlatforms,
                                  uint32_t *pNumPlatforms) {
    if (phPlatforms != nullptr && NumEntries != 1)
      return UR_RESULT_ERROR_INVALID_SIZE;
    if (pNumPlatforms != nullptr)
      *pNumPlatforms = 1;
    if (nullptr != phPlatforms)
      *reinterpret_cast<void **>(phPlatforms) = d_context.get();
    return UR_RESULT_SUCCESS;
  };

  //////////////////////////////////////////////////////////////////////////
  urDdiTable.Platform.pfnGetApiVersion = [](ur_platform_handle_t,
                                            ur_api_version_t *version) {
    *version = d_context.version;
    return UR_RESULT_SUCCESS;
  };

  //////////////////////////////////////////////////////////////////////////
  urDdiTable.Platform.pfnGetInfo =
      [](ur_platform_handle_t hPlatform, ur_platform_info_t PlatformInfoType,
         size_t Size, void *pPlatformInfo, size_t *pSizeRet) {
        if (!hPlatform) {
          return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        switch (PlatformInfoType) {
        case UR_PLATFORM_INFO_NAME: {
          const char null_platform_name[] = "UR_NULL_PLATFORM";
          if (pSizeRet) {
            *pSizeRet = sizeof(null_platform_name);
          }
          if (pPlatformInfo && Size != sizeof(null_platform_name)) {
            return UR_RESULT_ERROR_INVALID_SIZE;
          }
          if (pPlatformInfo) {
#if defined(_WIN32)
            strncpy_s(reinterpret_cast<char *>(pPlatformInfo), Size,
                      null_platform_name, sizeof(null_platform_name));
#else
            strncpy(reinterpret_cast<char *>(pPlatformInfo), null_platform_name,
                    Size);
#endif
          }
        } break;

        default:
          return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        return UR_RESULT_SUCCESS;
      };

  //////////////////////////////////////////////////////////////////////////
  urDdiTable.Device.pfnGet =
      [](ur_platform_handle_t hPlatform, ur_device_type_t DevicesType,
         uint32_t NumEntries, ur_device_handle_t *phDevices,
         uint32_t *pNumDevices) {
        (void)DevicesType;
        if (hPlatform == nullptr)
          return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        if (UR_DEVICE_TYPE_VPU < DevicesType)
          return UR_RESULT_ERROR_INVALID_ENUMERATION;
        if (phDevices != nullptr && NumEntries != 1)
          return UR_RESULT_ERROR_INVALID_SIZE;
        if (pNumDevices != nullptr)
          *pNumDevices = 1;
        if (nullptr != phDevices)
          *reinterpret_cast<void **>(phDevices) = d_context.get();
        return UR_RESULT_SUCCESS;
      };

  //////////////////////////////////////////////////////////////////////////
  urDdiTable.Device.pfnGetInfo = [](ur_device_handle_t hDevice,
                                    ur_device_info_t infoType, size_t propSize,
                                    void *pDeviceInfo, size_t *pPropSizeRet) {
    switch (infoType) {
    case UR_DEVICE_INFO_TYPE:
      if (pDeviceInfo && propSize != sizeof(ur_device_type_t))
        return UR_RESULT_ERROR_INVALID_SIZE;

      if (pDeviceInfo != nullptr) {
        *reinterpret_cast<ur_device_type_t *>(pDeviceInfo) = UR_DEVICE_TYPE_GPU;
      }
      if (pPropSizeRet != nullptr) {
        *pPropSizeRet = sizeof(ur_device_type_t);
      }
      break;

    case UR_DEVICE_INFO_NAME: {
      char deviceName[] = "Null Device";
      if (pDeviceInfo && propSize < sizeof(deviceName)) {
        return UR_RESULT_ERROR_INVALID_SIZE;
      }
      if (pDeviceInfo != nullptr) {
#if defined(_WIN32)
        strncpy_s(reinterpret_cast<char *>(pDeviceInfo), propSize, deviceName,
                  sizeof(deviceName));
#else
        strncpy(reinterpret_cast<char *>(pDeviceInfo), deviceName, propSize);
#endif
      }
      if (pPropSizeRet != nullptr) {
        *pPropSizeRet = sizeof(deviceName);
      }
    } break;

    default:
      return UR_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return UR_RESULT_SUCCESS;
  };
}
} // namespace driver
