//===--------- platform.cpp - CUDA Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "platform.hpp"
#include "common.hpp"
#include "context.hpp"
#include "device.hpp"

#include <cassert>
#include <cuda.h>
#include <sstream>

ur_result_t urPlatformGetInfo(ur_platform_handle_t hPlatform,
                              ur_platform_info_t PlatformInfoType, size_t Size,
                              void *pPlatformInfo, size_t *pSizeRet) {

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
    auto version = getCudaVersionString();
    return ReturnValue(version.c_str());
  }
  case UR_PLATFORM_INFO_EXTENSIONS: {
    return ReturnValue("");
  }
  case UR_PLATFORM_INFO_BACKEND: {
    return ReturnValue(UR_PLATFORM_BACKEND_CUDA);
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
///
/// However because multiple devices in a context is not currently supported,
/// place each device in a separate platform.
///
ur_result_t urPlatformGet(uint32_t NumEntries,
                          ur_platform_handle_t *phPlatforms,
                          uint32_t *pNumPlatforms) {

  try {
    static std::once_flag initFlag;
    static uint32_t numPlatforms = 1;
    static std::vector<ur_platform_handle_t_> platformIds;

    UR_ASSERT(phPlatforms || pNumPlatforms, UR_RESULT_ERROR_INVALID_VALUE);
    UR_ASSERT(!phPlatforms || NumEntries > 0, UR_RESULT_ERROR_INVALID_SIZE);

    ur_result_t err = UR_RESULT_SUCCESS;

    std::call_once(
        initFlag,
        [](ur_result_t &err) {
          if (cuInit(0) != CUDA_SUCCESS) {
            numPlatforms = 0;
            return;
          }
          int numDevices = 0;
          err = UR_CHECK_ERROR(cuDeviceGetCount(&numDevices));
          if (numDevices == 0) {
            numPlatforms = 0;
            return;
          }
          try {
            // make one platform per device
            numPlatforms = numDevices;
            platformIds.resize(numDevices);

            for (int i = 0; i < numDevices; ++i) {
              CUdevice device;
              err = UR_CHECK_ERROR(cuDeviceGet(&device, i));
              CUcontext context;
              err = UR_CHECK_ERROR(cuDevicePrimaryCtxRetain(&context, device));

              ScopedContext active(context);
              CUevent evBase;
              err = UR_CHECK_ERROR(cuEventCreate(&evBase, CU_EVENT_DEFAULT));

              // Use default stream to record base event counter
              err = UR_CHECK_ERROR(cuEventRecord(evBase, 0));

              platformIds[i].devices_.emplace_back(new ur_device_handle_t_{
                  device, context, evBase, &platformIds[i]});
              {
                const auto &dev = platformIds[i].devices_.back().get();
                size_t maxWorkGroupSize = 0u;
                size_t maxThreadsPerBlock[3] = {};
                ur_result_t retError = urDeviceGetInfo(
                    dev, UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES,
                    sizeof(maxThreadsPerBlock), maxThreadsPerBlock, nullptr);
                if (retError != UR_RESULT_SUCCESS) {
                  throw retError;
                }

                retError = urDeviceGetInfo(
                    dev, UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE,
                    sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr);
                if (retError != UR_RESULT_SUCCESS) {
                  throw retError;
                }

                dev->save_max_work_item_sizes(sizeof(maxThreadsPerBlock),
                                              maxThreadsPerBlock);
                dev->save_max_work_group_size(maxWorkGroupSize);
              }
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

ur_result_t urPlatformGetApiVersion(ur_platform_handle_t hDriver,
                                    ur_api_version_t *pVersion) {
  UR_ASSERT(hDriver, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pVersion, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  *pVersion = UR_API_VERSION_CURRENT;
  return UR_RESULT_SUCCESS;
}

ur_result_t urInit(ur_device_init_flags_t) { return UR_RESULT_SUCCESS; }

ur_result_t urTearDown(void *) { return UR_RESULT_SUCCESS; }
