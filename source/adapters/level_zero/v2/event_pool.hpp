//===--------- event_pool.hpp - Level Zero Adapter ------------------------===//
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

#include "../common.hpp"
#include "../device.hpp"
#include "event.hpp"
#include "event_provider.hpp"

namespace v2 {

class event_pool {
public:
  event_pool(std::unique_ptr<event_provider> Provider)
      : provider(std::move(Provider)){};

  event_pool(event_pool &&other) = default;
  event_pool &operator=(event_pool &&other) = default;

  event_pool(const event_pool &) = delete;
  event_pool &operator=(const event_pool &) = delete;

  DeviceId Id() { return provider->device()->Id.value(); };

  ur_event_handle_t_ *allocate();
  void free(ur_event_handle_t_ *event);

  event_provider *getProvider();

private:
  std::unique_ptr<event_provider> provider;

  std::deque<ur_event_handle_t_> events;
  std::vector<ur_event_handle_t_ *> freelist;
};

} // namespace v2
