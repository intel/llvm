//===----------- context.hpp - LLVM Offload Adapter  ----------------------===//
//
// Copyright (C) 2025 Intel Corporation
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
#include <OffloadAPI.h>
#include <unordered_map>
#include <ur_api.h>

struct ur_context_handle_t_ : RefCounted {
  ur_context_handle_t_(const ur_device_handle_t *Devs, size_t NumDevices)
      : Devices{Devs, Devs + NumDevices} {
    for (auto Device : Devices) {
      urDeviceRetain(Device);
    }
  }
  ~ur_context_handle_t_() {
    for (auto Device : Devices) {
      urDeviceRelease(Device);
    }
  }

  std::vector<ur_device_handle_t> Devices;

  // Gets the index of the device relative to other devices in the context
  size_t getDeviceIndex(ur_device_handle_t hDevice) {
    auto It = std::find(Devices.begin(), Devices.end(), hDevice);
    assert(It != Devices.end());
    return std::distance(Devices.begin(), It);
  }

  bool containsDevice(ur_device_handle_t Device) {
    return std::find(Devices.begin(), Devices.end(), Device) != Devices.end();
  }
};
