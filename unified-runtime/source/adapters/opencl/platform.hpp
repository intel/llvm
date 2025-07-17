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

#include <vector>

struct ur_device_handle_t_;

struct ur_platform_handle_t_ : ur::opencl::handle_base {
  using native_type = cl_platform_id;
  native_type CLPlatform = nullptr;
  std::vector<std::unique_ptr<ur_device_handle_t_>> Devices;
  std::map<cl_device_id, ur_device_handle_t> SubDevices;
  std::mutex SubDevicesLock;

  ur_platform_handle_t_(native_type Plat) : handle_base(), CLPlatform(Plat) {}

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

  ur_result_t InitDevices();

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
