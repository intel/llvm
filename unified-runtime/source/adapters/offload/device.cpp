//===----------- device.cpp - LLVM Offload Adapter  -----------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>
#include <ur/ur.hpp>
#include <ur_api.h>

#include "device.hpp"
#include "platform.hpp"
#include "ur2offload.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urDeviceGet(ur_platform_handle_t hPlatform,
                                                ur_device_type_t,
                                                uint32_t NumEntries,
                                                ur_device_handle_t *phDevices,
                                                uint32_t *pNumDevices) {
  if (pNumDevices) {
    *pNumDevices = static_cast<uint32_t>(hPlatform->Devices.size());
  }

  size_t NumDevices =
      std::min(static_cast<uint32_t>(hPlatform->Devices.size()), NumEntries);

  for (size_t I = 0; I < NumDevices; I++) {
    phDevices[I] = &hPlatform->Devices[I];
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceGetInfo(ur_device_handle_t hDevice,
                                                    ur_device_info_t propName,
                                                    size_t propSize,
                                                    void *pPropValue,
                                                    size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  ol_device_info_t olInfo;
  switch (propName) {
  case UR_DEVICE_INFO_NAME:
    olInfo = OL_DEVICE_INFO_NAME;
    break;
  case UR_DEVICE_INFO_PARENT_DEVICE:
    return ReturnValue(nullptr);
  case UR_DEVICE_INFO_VERSION:
    return ReturnValue("");
  case UR_DEVICE_INFO_EXTENSIONS:
    return ReturnValue("");
  case UR_DEVICE_INFO_USE_NATIVE_ASSERT:
    return ReturnValue(false);
  case UR_DEVICE_INFO_TYPE:
    olInfo = OL_DEVICE_INFO_TYPE;
    break;
  case UR_DEVICE_INFO_VENDOR:
    olInfo = OL_DEVICE_INFO_VENDOR;
    break;
  case UR_DEVICE_INFO_DRIVER_VERSION:
    olInfo = OL_DEVICE_INFO_DRIVER_VERSION;
    break;
  case UR_DEVICE_INFO_PLATFORM:
    return ReturnValue(hDevice->Platform);
    break;
  case UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT:
    return ReturnValue(UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS);
  case UR_DEVICE_INFO_BUILD_ON_SUBDEVICE:
    return ReturnValue(false);
  default:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }

  if (pPropSizeRet) {
    if (auto Res =
            olGetDeviceInfoSize(hDevice->OffloadDevice, olInfo, pPropSizeRet)) {
      return offloadResultToUR(Res);
    }
  }

  if (pPropValue) {
    if (auto Res = olGetDeviceInfo(hDevice->OffloadDevice, olInfo, propSize,
                                   pPropValue)) {
      return offloadResultToUR(Res);
    }
    // Need to explicitly map this type
    if (olInfo == OL_DEVICE_INFO_TYPE) {
      auto urPropPtr = reinterpret_cast<ur_device_type_t *>(pPropValue);
      auto olPropPtr = reinterpret_cast<ol_device_type_t *>(pPropValue);

      switch (*olPropPtr) {
      case OL_DEVICE_TYPE_CPU:
        *urPropPtr = UR_DEVICE_TYPE_CPU;
        break;
      case OL_DEVICE_TYPE_GPU:
        *urPropPtr = UR_DEVICE_TYPE_GPU;
        break;
      default:
        break;
      }
    }
  }

  return UR_RESULT_SUCCESS;
}

// Device partitioning is not supported in Offload, and won't be for some time.
// This means urDeviceRetain/Release are no-ops because all devices are root
// devices.

UR_APIEXPORT ur_result_t UR_APICALL urDeviceRetain(ur_device_handle_t) {
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceRelease(ur_device_handle_t) {
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urDevicePartition(ur_device_handle_t, const ur_device_partition_properties_t *,
                  uint32_t, ur_device_handle_t *, uint32_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceSelectBinary(
    ur_device_handle_t hDevice, const ur_device_binary_t *pBinaries,
    uint32_t NumBinaries, uint32_t *pSelectedBinary) {

  ol_platform_backend_t Backend;
  olGetPlatformInfo(hDevice->Platform->OffloadPlatform,
                    OL_PLATFORM_INFO_BACKEND, sizeof(Backend), &Backend);

  const char *ImageTarget = UR_DEVICE_BINARY_TARGET_UNKNOWN;
  if (Backend == OL_PLATFORM_BACKEND_CUDA) {
    ImageTarget = UR_DEVICE_BINARY_TARGET_NVPTX64;
  } else if (Backend == OL_PLATFORM_BACKEND_AMDGPU) {
    ImageTarget = UR_DEVICE_BINARY_TARGET_AMDGCN;
  }

  for (uint32_t i = 0; i < NumBinaries; ++i) {
    if (strcmp(pBinaries[i].pDeviceTargetSpec, ImageTarget) == 0) {
      *pSelectedBinary = i;
      return UR_RESULT_SUCCESS;
    }
  }

  // No image can be loaded for the given device
  return UR_RESULT_ERROR_INVALID_BINARY;
}
