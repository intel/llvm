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

using event_pool_borrowed =
    std::unique_ptr<event_pool, std::function<void(event_pool *)>>;

class event_pool_cache {
public:
  using ProviderCreateFunc =
      std::function<std::unique_ptr<event_provider>(DeviceId)>;

  event_pool_cache(size_t max_devices, ProviderCreateFunc);
  ~event_pool_cache();

  event_pool_borrowed borrow(DeviceId);

private:
  ur_mutex mutex;
  ProviderCreateFunc providerCreate;
  std::vector<std::vector<std::unique_ptr<event_pool>>> pools;
};

} // namespace v2
