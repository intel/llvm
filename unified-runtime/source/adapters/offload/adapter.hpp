//===----------- adapter.hpp - LLVM Offload Adapter  ----------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <atomic>
#include <cstdint>
#include <unordered_set>

#include <OffloadAPI.h>

#include "common.hpp"
#include "logger/ur_logger.hpp"
#include "platform.hpp"

struct ur_adapter_handle_t_ : ur::offload::handle_base {
  std::atomic_uint32_t RefCount = 0;
  logger::Logger &Logger = logger::get_logger("offload");
  ol_device_handle_t HostDevice = nullptr;
  std::vector<ur_platform_handle_t_> Platforms;

  ur_result_t init();
};

extern ur_adapter_handle_t_ Adapter;
