//===----------- adapter.cpp - LLVM Offload Adapter  ----------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>
#include <atomic>
#include <cstdint>
#include <unordered_set>

#include "adapter.hpp"
#include "device.hpp"
#include "platform.hpp"
#include "ur/ur.hpp"
#include "ur_api.h"

ur_adapter_handle_t_ Adapter{};

// Initialize liboffload and perform the initial platform and device discovery
ur_result_t ur_adapter_handle_t_::init() {
  auto Res = olInit();
  (void)Res;

  // Discover every platform and device
  Res = olIterateDevices(
      [](ol_device_handle_t D, void *UserData) {
        auto *Platforms =
            reinterpret_cast<decltype(Adapter.Platforms) *>(UserData);

        ol_platform_handle_t Platform;
        olGetDeviceInfo(D, OL_DEVICE_INFO_PLATFORM, sizeof(Platform),
                        &Platform);
        ol_platform_backend_t Backend;
        olGetPlatformInfo(Platform, OL_PLATFORM_INFO_BACKEND, sizeof(Backend),
                          &Backend);
        if (Backend == OL_PLATFORM_BACKEND_HOST) {
          Adapter.HostDevice = D;
        } else if (Backend != OL_PLATFORM_BACKEND_UNKNOWN) {
          auto URPlatform =
              std::find_if(Platforms->begin(), Platforms->end(), [&](auto &P) {
                return P.OffloadPlatform == Platform;
              });

          if (URPlatform == Platforms->end()) {
            URPlatform =
                Platforms->insert(URPlatform, ur_platform_handle_t_(Platform));
          }

          URPlatform->Devices.push_back(ur_device_handle_t_{&*URPlatform, D});
        }
        return false;
      },
      &Adapter.Platforms);

  (void)Res;

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
    return ReturnValue(UR_BACKEND_OFFLOAD);
  case UR_ADAPTER_INFO_REFERENCE_COUNT:
    return ReturnValue(Adapter.RefCount.load());
  case UR_ADAPTER_INFO_VERSION:
    return ReturnValue(1);
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterGetLastError(ur_adapter_handle_t,
                                                          const char **,
                                                          int32_t *) {
  // This only needs to write out the error if another entry point has returned
  // "ADAPTER_SPECIFIC", which we never do
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterSetLoggerCallback(
    ur_adapter_handle_t, ur_logger_callback_t pfnLoggerCallback,
    void *pUserData, ur_logger_level_t level = UR_LOGGER_LEVEL_QUIET) {

  Adapter.Logger.setCallbackSink(pfnLoggerCallback, pUserData, level);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urAdapterSetLoggerCallbackLevel(ur_adapter_handle_t, ur_logger_level_t level) {

  Adapter.Logger.setCallbackLevel(level);

  return UR_RESULT_SUCCESS;
}
