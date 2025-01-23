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

#include "adapter.hpp"
#include "common.hpp"
#include "device.hpp"

#include <vector>

struct ur_context_handle_t_ : ur::opencl::handle_base {
  using native_type = cl_context;
  native_type CLContext;
  std::vector<ur_device_handle_t> Devices;
  uint32_t DeviceCount;
  std::atomic<uint32_t> RefCount = 0;
  bool IsNativeHandleOwned = true;

  ur_context_handle_t_(native_type Ctx, uint32_t DevCount,
                       const ur_device_handle_t *phDevices)
      : handle_base(), CLContext(Ctx), DeviceCount(DevCount) {
    for (uint32_t i = 0; i < DeviceCount; i++) {
      Devices.emplace_back(phDevices[i]);
      urDeviceRetain(phDevices[i]);
    }
    // The context retains a reference to the adapter so it can clear the
    // function ptr cache on destruction
    urAdapterRetain(ur::cl::getAdapter());
    RefCount = 1;
  }

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }

  uint32_t decrementReferenceCount() noexcept { return --RefCount; }

  uint32_t getReferenceCount() const noexcept { return RefCount; }

  static ur_result_t makeWithNative(native_type Ctx, uint32_t DevCount,
                                    const ur_device_handle_t *phDevices,
                                    ur_context_handle_t &Context);
  ~ur_context_handle_t_() {
    // If we're reasonably sure this context is about to be destroyed we should
    // clear the ext function pointer cache. This isn't foolproof sadly but it
    // should drastically reduce the chances of the pathological case described
    // in the comments in common.hpp.
    ur::cl::getAdapter()->fnCache.clearCache(CLContext);
    urAdapterRelease(ur::cl::getAdapter());

    for (uint32_t i = 0; i < DeviceCount; i++) {
      urDeviceRelease(Devices[i]);
    }
    if (IsNativeHandleOwned) {
      clReleaseContext(CLContext);
    }
  }
};
