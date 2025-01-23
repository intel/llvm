//===--------- context.hpp - OpenCL Adapter ---------------------------===//
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

struct ur_context_handle_t_ : cl_adapter::ur_handle_t_ {
  using native_type = cl_context;
  native_type CLContext;
  std::vector<ur_device_handle_t> Devices;
  uint32_t DeviceCount;
  std::atomic<uint32_t> RefCount = 0;
  bool IsNativeHandleOwned = true;

  ur_context_handle_t_(native_type Ctx, uint32_t DevCount,
                       const ur_device_handle_t *phDevices)
      : cl_adapter::ur_handle_t_(), CLContext(Ctx), DeviceCount(DevCount) {
    for (uint32_t i = 0; i < DeviceCount; i++) {
      Devices.emplace_back(phDevices[i]);
      urDeviceRetain(phDevices[i]);
    }
    RefCount = 1;
  }

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }

  uint32_t decrementReferenceCount() noexcept { return --RefCount; }

  uint32_t getReferenceCount() const noexcept { return RefCount; }

  static ur_result_t makeWithNative(native_type Ctx, uint32_t DevCount,
                                    const ur_device_handle_t *phDevices,
                                    ur_context_handle_t &Context);
  ~ur_context_handle_t_() {
    for (uint32_t i = 0; i < DeviceCount; i++) {
      urDeviceRelease(Devices[i]);
    }
    if (IsNativeHandleOwned) {
      clReleaseContext(CLContext);
    }
  }
};
