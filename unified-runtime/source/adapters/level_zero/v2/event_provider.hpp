//===--------- command_list_cache.hpp - Level Zero Adapter ---------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <memory>
#include <mutex>
#include <stack>

#include <unordered_map>
#include <ur/ur.hpp>
#include <ur_api.h>
#include <vector>
#include <ze_api.h>

namespace v2 {

using event_flags_t = uint32_t;
enum event_flag_t {
  EVENT_FLAGS_COUNTER = UR_BIT(0),
  EVENT_FLAGS_PROFILING_ENABLED = UR_BIT(1),
};
static constexpr size_t EVENT_FLAGS_USED_BITS = 2;

class event_provider;

namespace raii {
using cache_borrowed_event =
    std::unique_ptr<_ze_event_handle_t,
                    std::function<void(::ze_event_handle_t)>>;
} // namespace raii

class event_provider {
public:
  virtual ~event_provider() = default;
  virtual raii::cache_borrowed_event allocate() = 0;
  virtual event_flags_t eventFlags() const = 0;
};

} // namespace v2
