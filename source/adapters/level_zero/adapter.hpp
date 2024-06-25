//===--------- adapters.hpp - Level Zero Adapter --------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "logger/ur_logger.hpp"
#include <atomic>
#include <loader/ze_loader.h>
#include <mutex>
#include <optional>
#include <ur/ur.hpp>
#include <ze_api.h>

using PlatformVec = std::vector<std::unique_ptr<ur_platform_handle_t_>>;

class ur_legacy_sink;

struct ur_adapter_handle_t_ {
  ur_adapter_handle_t_();
  std::atomic<uint32_t> RefCount = 0;
  std::mutex Mutex;

  std::optional<ze_result_t> ZeResult;
  ZeCache<Result<PlatformVec>> PlatformCache;
  logger::Logger &logger;
};

extern ur_adapter_handle_t_ *GlobalAdapter;
