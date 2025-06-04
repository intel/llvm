//===--------- adapter.hpp - CUDA Adapter ---------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef UR_CUDA_ADAPTER_HPP_INCLUDED
#define UR_CUDA_ADAPTER_HPP_INCLUDED

#include "logger/ur_logger.hpp"
#include "platform.hpp"
#include "tracing.hpp"
#include <ur_api.h>

#include <atomic>
#include <memory>

struct ur_adapter_handle_t_ : ur::cuda::handle_base {
  std::atomic<uint32_t> RefCount = 0;
  struct cuda_tracing_context_t_ *TracingCtx = nullptr;
  logger::Logger &logger;
  std::unique_ptr<ur_platform_handle_t_> Platform;
  ur_adapter_handle_t_();
  ~ur_adapter_handle_t_();
  ur_adapter_handle_t_(const ur_adapter_handle_t_ &) = delete;
};

// Keep the global namespace'd
namespace ur::cuda {
extern ur_adapter_handle_t adapter;
} // namespace ur::cuda

#endif // UR_CUDA_ADAPTER_HPP_INCLUDED
