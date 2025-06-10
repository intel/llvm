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

#include "common.hpp"
#include <OffloadAPI.h>
#include <unordered_map>
#include <ur_api.h>

struct ur_context_handle_t_ : RefCounted {
  ur_context_handle_t_(ur_device_handle_t hDevice) : Device{hDevice} {
    urDeviceRetain(Device);
  }
  ~ur_context_handle_t_() { urDeviceRelease(Device); }

  ur_device_handle_t Device;
  std::unordered_map<void *, ol_alloc_type_t> AllocTypeMap;
};
