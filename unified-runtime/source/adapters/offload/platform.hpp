//===----------- platform.hpp - LLVM Offload Adapter  ---------------------===//
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
#include <vector>

struct ur_platform_handle_t_ : ur::offload::handle_base {
  ur_platform_handle_t_(ol_platform_handle_t OffloadPlatform)
      : handle_base(), OffloadPlatform(OffloadPlatform) {};

  ol_platform_handle_t OffloadPlatform;
  std::vector<std::unique_ptr<ur_device_handle_t_>> Devices;
};
