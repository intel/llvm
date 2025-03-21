//===--------- adapter.hpp - CUDA Adapter ---------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "logger/ur_logger.hpp"
#include "platform.hpp"
#include "tracing.hpp"
#include <ur_api.h>

#include <atomic>
#include <memory>
#include <mutex>

// should maybe be an ifdef
#pragma once

struct ur_platform_handle_t_;

struct ur_adapter_handle_t_ {
  std::atomic<uint32_t> RefCount = 0;
  std::mutex Mutex;
  struct cuda_tracing_context_t_ *TracingCtx = nullptr;
  logger::Logger &logger;
  std::unique_ptr<ur_platform_handle_t_> Platform;
  ur_adapter_handle_t_();
};

// Keep the global namespace'd
namespace ur::cuda {
extern std::shared_ptr<ur_adapter_handle_t_> adapter;
} // namespace ur::cuda
