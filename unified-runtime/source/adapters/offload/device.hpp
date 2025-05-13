//===----------- device.hpp - LLVM Offload Adapter  -----------------------===//
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
#include <ur_api.h>

struct ur_device_handle_t_ : ur::offload::handle_base {
  ur_device_handle_t_(ur_platform_handle_t Platform,
                      ol_device_handle_t OffloadDevice)
      : handle_base(), Platform(Platform), OffloadDevice(OffloadDevice) {}

  ur_platform_handle_t Platform;
  ol_device_handle_t OffloadDevice;
};
