//===--------- device.hpp - OpenCL Adapter ---------------------------===//
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
#include "platform.hpp"

struct ur_device_handle_t_ {
  using native_type = cl_device_id;
  native_type CLDevice;
  ur_platform_handle_t Platform;
  cl_device_type Type = 0;
  ur_device_handle_t ParentDevice = nullptr;
  std::atomic<uint32_t> RefCount = 0;
  bool IsNativeHandleOwned = true;

  ur_device_handle_t_(native_type Dev, ur_platform_handle_t Plat,
                      ur_device_handle_t Parent)
      : CLDevice(Dev), Platform(Plat), ParentDevice(Parent) {
    RefCount = 1;
    if (Parent) {
      Type = Parent->Type;
    } else {
      clGetDeviceInfo(CLDevice, CL_DEVICE_TYPE, sizeof(cl_device_type), &Type,
                      nullptr);
    }
  }

  ~ur_device_handle_t_() {
    if (ParentDevice && IsNativeHandleOwned) {
      clReleaseDevice(CLDevice);
    }
  }

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }

  uint32_t decrementReferenceCount() noexcept { return --RefCount; }

  uint32_t getReferenceCount() const noexcept { return RefCount; }

  ur_result_t getDeviceVersion(oclv::OpenCLVersion &Version) {
    size_t DevVerSize = 0;
    CL_RETURN_ON_FAILURE(
        clGetDeviceInfo(CLDevice, CL_DEVICE_VERSION, 0, nullptr, &DevVerSize));

    std::string DevVer(DevVerSize, '\0');
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(CLDevice, CL_DEVICE_VERSION,
                                         DevVerSize, DevVer.data(), nullptr));

    Version = oclv::OpenCLVersion(DevVer);
    if (!Version.isValid()) {
      return UR_RESULT_ERROR_INVALID_DEVICE;
    }

    return UR_RESULT_SUCCESS;
  }

  bool isIntelFPGAEmuDevice() {
    size_t NameSize = 0;
    CL_RETURN_ON_FAILURE(
        clGetDeviceInfo(CLDevice, CL_DEVICE_NAME, 0, nullptr, &NameSize));
    std::string NameStr(NameSize, '\0');
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(CLDevice, CL_DEVICE_NAME, NameSize,
                                         NameStr.data(), nullptr));

    return NameStr.find("Intel(R) FPGA Emulation Device") != std::string::npos;
  }

  ur_result_t checkDeviceExtensions(const std::vector<std::string> &Exts,
                                    bool &Supported) {
    size_t ExtSize = 0;
    CL_RETURN_ON_FAILURE(
        clGetDeviceInfo(CLDevice, CL_DEVICE_EXTENSIONS, 0, nullptr, &ExtSize));

    std::string ExtStr(ExtSize, '\0');

    CL_RETURN_ON_FAILURE(clGetDeviceInfo(CLDevice, CL_DEVICE_EXTENSIONS,
                                         ExtSize, ExtStr.data(), nullptr));

    Supported = true;
    for (const std::string &Ext : Exts) {
      if (!(Supported = (ExtStr.find(Ext) != std::string::npos))) {
        // The Intel FPGA emulation device does actually support these, even if
        // it doesn't report them.
        if (isIntelFPGAEmuDevice() &&
            (Ext == "cl_intel_device_attribute_query" ||
             Ext == "cl_intel_required_subgroup_size" ||
             Ext == "cl_khr_subgroups")) {
          Supported = true;
          continue;
        }
        break;
      }
    }

    return UR_RESULT_SUCCESS;
  }
};
