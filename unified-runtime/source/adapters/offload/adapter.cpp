//===----------- adapter.cpp - LLVM Offload Plugin  -----------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <atomic>
#include <cstdint>
#include <OffloadAPI.h>
#include <unordered_set>

#include "adapter.hpp"
#include "ur/ur.hpp"
#include "ur_api.h"

ur_adapter_handle_t_ Adapter{};

// Initialize liboffload and perform the initial platform and device discovery
ur_result_t ur_adapter_handle_t_::init() {
  auto Res = olInit();
  (void)Res;

  // Discover every platform that isn't the host platform.
  // Use an unordered_set to deduplicate platforms we discover multiple times
  // from different devices.
  // Also discover the host device. We only expect one so don't need to worry
  // about overwriting it.
  Res = olIterateDevices(
      [](ol_device_handle_t D, void *UserData) {
        auto Adapter = static_cast<ur_adapter_handle_t>(UserData);
        ol_platform_handle_t Platform;
        olGetDeviceInfo(D, OL_DEVICE_INFO_PLATFORM, sizeof(Platform),
                        &Platform);
        ol_platform_backend_t Backend;
        olGetPlatformInfo(Platform, OL_PLATFORM_INFO_BACKEND, sizeof(Backend),
                          &Backend);
        if (Backend == OL_PLATFORM_BACKEND_HOST) {
          Adapter->HostDevice = D;
        } else if (Backend != OL_PLATFORM_BACKEND_UNKNOWN) {
          Adapter->Platforms.insert(Platform);
        }
        return false;
      },
      this);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterGet(
    uint32_t, ur_adapter_handle_t *phAdapters, uint32_t *pNumAdapters) {
  if (phAdapters) {
    if (++Adapter.RefCount == 1) {
      Adapter.init();
    }
    *phAdapters = &Adapter;
  }
  if (pNumAdapters) {
    *pNumAdapters = 1;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterRelease(ur_adapter_handle_t) {
  if (--Adapter.RefCount == 0) {
    // This can crash when tracing is enabled.
    // olShutDown();
  };
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterRetain(ur_adapter_handle_t) {
  Adapter.RefCount++;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterGetInfo(ur_adapter_handle_t,
                                                     ur_adapter_info_t propName,
                                                     size_t propSize,
                                                     void *pPropValue,
                                                     size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_ADAPTER_INFO_BACKEND:
    return ReturnValue(UR_BACKEND_CUDA); // TODO: Return a proper value
  case UR_ADAPTER_INFO_REFERENCE_COUNT:
    return ReturnValue(Adapter.RefCount.load());
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }

  return UR_RESULT_SUCCESS;
}
