//===----------- platform.hpp - LLVM Offload Adapter  ---------------------===//
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
#include <vector>

struct ur_platform_handle_t_ : ur::offload::handle_base {
  ur_platform_handle_t_(ol_platform_handle_t OffloadPlatform)
      : handle_base(), OffloadPlatform(OffloadPlatform) {};

  ol_platform_handle_t OffloadPlatform;
  std::vector<ur_device_handle_t_> Devices;
};
