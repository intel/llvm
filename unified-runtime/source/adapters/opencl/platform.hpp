//===--------- platform.hpp - OpenCL Adapter ---------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "common.hpp"
#include "device.hpp"

#include <vector>

struct ur_platform_handle_t_ : cl_adapter::ur_handle_t_ {
  using native_type = cl_platform_id;
  native_type CLPlatform = nullptr;
  std::vector<std::unique_ptr<ur_device_handle_t_>> Devices;

  ur_platform_handle_t_(native_type Plat)
      : cl_adapter::ur_handle_t_(), CLPlatform(Plat) {}

  ~ur_platform_handle_t_() {
    for (auto &Dev : Devices) {
      Dev.reset();
    }
    Devices.clear();
  }

  template <typename T>
  ur_result_t getExtFunc(T CachedExtFunc, const char *FuncName, T *Fptr) {
    if (!CachedExtFunc) {
      // TODO: check that the function is available
      CachedExtFunc = reinterpret_cast<T>(
          clGetExtensionFunctionAddressForPlatform(CLPlatform, FuncName));
      if (!CachedExtFunc) {
        return UR_RESULT_ERROR_INVALID_VALUE;
      }
    }
    *Fptr = CachedExtFunc;
    return UR_RESULT_SUCCESS;
  }

  ur_result_t InitDevices() {
    if (Devices.empty()) {
      cl_uint DeviceNum = 0;
      cl_int Res = clGetDeviceIDs(CLPlatform, CL_DEVICE_TYPE_ALL, 0, nullptr,
                                  &DeviceNum);

      // Absorb the CL_DEVICE_NOT_FOUND and just return 0 in num_devices
      if (Res == CL_DEVICE_NOT_FOUND) {
        return UR_RESULT_SUCCESS;
      }

      CL_RETURN_ON_FAILURE(Res);

      std::vector<cl_device_id> CLDevices(DeviceNum);
      Res = clGetDeviceIDs(CLPlatform, CL_DEVICE_TYPE_ALL, DeviceNum,
                           CLDevices.data(), nullptr);

      // Absorb the CL_DEVICE_NOT_FOUND and just return 0 in num_devices
      if (Res == CL_DEVICE_NOT_FOUND) {
        return UR_RESULT_SUCCESS;
      }

      CL_RETURN_ON_FAILURE(Res);

      try {
        Devices.resize(DeviceNum);
        for (size_t i = 0; i < DeviceNum; i++) {
          Devices[i] = std::make_unique<ur_device_handle_t_>(CLDevices[i], this,
                                                             nullptr);
        }
      } catch (std::bad_alloc &) {
        return UR_RESULT_ERROR_OUT_OF_RESOURCES;
      } catch (...) {
        return UR_RESULT_ERROR_UNKNOWN;
      }
    }

    return UR_RESULT_SUCCESS;
  }

  ur_result_t getPlatformVersion(oclv::OpenCLVersion &Version) {
    size_t PlatVerSize = 0;
    CL_RETURN_ON_FAILURE(clGetPlatformInfo(CLPlatform, CL_PLATFORM_VERSION, 0,
                                           nullptr, &PlatVerSize));

    std::string PlatVer(PlatVerSize, '\0');
    CL_RETURN_ON_FAILURE(clGetPlatformInfo(
        CLPlatform, CL_PLATFORM_VERSION, PlatVerSize, PlatVer.data(), nullptr));

    Version = oclv::OpenCLVersion(PlatVer);
    if (!Version.isValid()) {
      return UR_RESULT_ERROR_INVALID_PLATFORM;
    }

    return UR_RESULT_SUCCESS;
  }
};
