//===--------- platform.cpp - CUDA Adapter --------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "platform.hpp"
#include "adapter.hpp"
#include "common.hpp"
#include "context.hpp"
#include "device.hpp"

#include <cassert>
#include <cuda.h>
#include <sstream>

UR_APIEXPORT ur_result_t UR_APICALL urPlatformGetInfo(
    ur_platform_handle_t hPlatform, ur_platform_info_t PlatformInfoType,
    size_t Size, void *pPlatformInfo, size_t *pSizeRet) {

  UR_ASSERT(hPlatform, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UrReturnHelper ReturnValue(Size, pPlatformInfo, pSizeRet);

  switch (PlatformInfoType) {
  case UR_PLATFORM_INFO_NAME:
    return ReturnValue("NVIDIA CUDA BACKEND");
  case UR_PLATFORM_INFO_VENDOR_NAME:
    return ReturnValue("NVIDIA Corporation");
  case UR_PLATFORM_INFO_PROFILE:
    return ReturnValue("FULL PROFILE");
  case UR_PLATFORM_INFO_VERSION: {
    auto Version = getCudaVersionString();
    return ReturnValue(Version.c_str());
  }
  case UR_PLATFORM_INFO_EXTENSIONS: {
    return ReturnValue("");
  }
  case UR_PLATFORM_INFO_BACKEND: {
    return ReturnValue(UR_PLATFORM_BACKEND_CUDA);
  }
  case UR_PLATFORM_INFO_ADAPTER: {
    return ReturnValue(&adapter);
  }
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }

  return UR_RESULT_SUCCESS;
}

/// Obtains the CUDA platform.
/// There is only one CUDA platform, and contains all devices on the system.
/// Triggers the CUDA Driver initialization (cuInit) the first time, so this
/// must be the first PI API called.
UR_APIEXPORT ur_result_t UR_APICALL
urPlatformGet(ur_adapter_handle_t *, uint32_t, uint32_t NumEntries,
              ur_platform_handle_t *phPlatforms, uint32_t *pNumPlatforms) {

  try {
    static std::once_flag InitFlag;
    static uint32_t NumPlatforms = 1;
    static ur_platform_handle_t_ Platform;

    UR_ASSERT(phPlatforms || pNumPlatforms, UR_RESULT_ERROR_INVALID_VALUE);
    UR_ASSERT(!phPlatforms || NumEntries > 0, UR_RESULT_ERROR_INVALID_SIZE);

    ur_result_t Result = UR_RESULT_SUCCESS;

    std::call_once(
        InitFlag,
        [](ur_result_t &Result) {
          CUresult InitResult = cuInit(0);
          if (InitResult == CUDA_ERROR_NO_DEVICE) {
            return; // No devices found which is not an error so exit early.
          }
          UR_CHECK_ERROR(InitResult);
          int NumDevices = 0;
          UR_CHECK_ERROR(cuDeviceGetCount(&NumDevices));
          try {
            for (int i = 0; i < NumDevices; ++i) {
              CUdevice Device;
              UR_CHECK_ERROR(cuDeviceGet(&Device, i));
              CUcontext Context;
              UR_CHECK_ERROR(cuDevicePrimaryCtxRetain(&Context, Device));

              ScopedContext Active(Context); // Set native ctx as active
              CUevent EvBase;
              UR_CHECK_ERROR(cuEventCreate(&EvBase, CU_EVENT_DEFAULT));

              // Use default stream to record base event counter
              UR_CHECK_ERROR(cuEventRecord(EvBase, 0));

              Platform.Devices.emplace_back(
                  new ur_device_handle_t_{Device, Context, EvBase, &Platform,
                                          static_cast<uint32_t>(i)});
            }
          } catch (const std::bad_alloc &) {
            // Signal out-of-memory situation
            for (int i = 0; i < NumDevices; ++i) {
              Platform.Devices.clear();
            }
            Result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
          } catch (ur_result_t Err) {
            // Clear and rethrow to allow retry
            for (int i = 0; i < NumDevices; ++i) {
              Platform.Devices.clear();
            }
            Result = Err;
            throw Err;
          } catch (...) {
            Result = UR_RESULT_ERROR_OUT_OF_RESOURCES;
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

UR_APIEXPORT ur_result_t UR_APICALL urPlatformGetApiVersion(
    ur_platform_handle_t hDriver, ur_api_version_t *pVersion) {
  std::ignore = hDriver;
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
  // There is no CUDA equivalent to ur_platform_handle_t
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

// Get CUDA plugin specific backend option.
// Current support is only for optimization options.
// Return empty string for cuda.
// TODO: Determine correct string to be passed.
UR_APIEXPORT ur_result_t UR_APICALL urPlatformGetBackendOption(
    ur_platform_handle_t hPlatform, const char *pFrontendOption,
    const char **ppPlatformOption) {
  std::ignore = hPlatform;
  using namespace std::literals;
  if (pFrontendOption == nullptr)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
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
