//===--------- command_list_cache.hpp - Level Zero Adapter ---------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <memory>
#include <mutex>
#include <stack>

#include <unified-runtime/ur_api.h>
#include <unordered_map>
#include <ur/ur.hpp>
#include <vector>
#include <ze_api.h>

namespace v2 {

using event_flags_t = uint32_t;
enum event_flag_t {
  EVENT_FLAGS_COUNTER = UR_BIT(0),
  EVENT_FLAGS_PROFILING_ENABLED = UR_BIT(1),
  // IPC-shareable producer event.
  EVENT_FLAGS_IPC = UR_BIT(2),
  // Event opened from an IPC handle.
  EVENT_FLAGS_IPC_IMPORTED = UR_BIT(3),
};
// Bits used for indexing in the event_pool_cache. Imported IPC events are opened and not created from via the provider, so are excluded from this count.
static constexpr size_t EVENT_FLAGS_USED_BITS = 3;

enum queue_type {
  QUEUE_REGULAR,
  QUEUE_IMMEDIATE,
};

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
