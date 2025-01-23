//===--------- adapter.hpp - HIP Adapter ----------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef UR_HIP_ADAPTER_HPP_INCLUDED
#define UR_HIP_ADAPTER_HPP_INCLUDED

#include "logger/ur_logger.hpp"
#include "platform.hpp"

#include <atomic>
#include <memory>

struct ur_adapter_handle_t_ : ur::hip::handle_base {
  std::atomic<uint32_t> RefCount = 0;
  logger::Logger &logger;
  std::unique_ptr<ur_platform_handle_t_> Platform;
  ur_adapter_handle_t_();
};

namespace ur::hip {
extern ur_adapter_handle_t adapter;
}

#endif // UR_HIP_ADAPTER_HPP_INCLUDED
