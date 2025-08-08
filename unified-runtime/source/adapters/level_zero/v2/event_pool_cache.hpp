//===--------- event_pool_cache.hpp - Level Zero Adapter ------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <stack>

#include <unordered_map>
#include <ur/ur.hpp>
#include <ur_api.h>
#include <ze_api.h>

#include "../device.hpp"
#include "event_pool.hpp"
#include "event_provider.hpp"

namespace v2 {

namespace raii {
using cache_borrowed_event_pool =
    std::unique_ptr<event_pool, std::function<void(event_pool *)>>;
} // namespace raii

class event_pool_cache {
public:
  using ProviderCreateFunc = std::function<std::unique_ptr<event_provider>(
      DeviceId, event_flags_t flags)>;

  event_pool_cache(ur_context_handle_t hContext, size_t max_devices,
                   ProviderCreateFunc);

  raii::cache_borrowed_event_pool borrow(DeviceId, event_flags_t flags);

private:
  ur_context_handle_t hContext;
  ur_mutex mutex;
  ProviderCreateFunc providerCreate;

  struct event_descriptor {
    DeviceId device;
    event_flags_t flags;

    uint64_t index() {
      return uint64_t(flags) | (uint64_t(device) << EVENT_FLAGS_USED_BITS);
    }
  };

  // Indexed by event_descriptor::index()
  std::vector<std::vector<std::unique_ptr<event_pool>>> pools;
};

} // namespace v2
