//===----------- device.hpp - LLVM Offload Adapter  -----------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "common.hpp"
#include <OffloadAPI.h>
#include <unified-runtime/ur_api.h>

struct ur_device_handle_t_ : ur::offload::handle_base {
  ur_device_handle_t_(ur_platform_handle_t Platform,
                      ol_device_handle_t OffloadDevice)
      : handle_base(), Platform(Platform), OffloadDevice(OffloadDevice) {}

  ur_platform_handle_t Platform;
  ol_device_handle_t OffloadDevice;
};
