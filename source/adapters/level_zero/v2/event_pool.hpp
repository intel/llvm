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
  // store weak reference to the queue as event_pool is part of the queue
  event_pool(ur_context_handle_t hContext,
             std::unique_ptr<event_provider> Provider)
      : hContext(hContext), provider(std::move(Provider)),
        mutex(std::make_unique<std::mutex>()){};

  event_pool(event_pool &&other) = default;
  event_pool &operator=(event_pool &&other) = default;

  event_pool(const event_pool &) = delete;
  event_pool &operator=(const event_pool &) = delete;

  // Allocate an event from the pool. Thread safe.
  ur_pooled_event_t *allocate();

  // Free an event back to the pool. Thread safe.
  void free(ur_pooled_event_t *event);

  event_provider *getProvider() const;
  event_flags_t getFlags() const;

private:
  ur_context_handle_t hContext;
  std::unique_ptr<event_provider> provider;

  std::deque<ur_pooled_event_t> events;
  std::vector<ur_pooled_event_t *> freelist;

  std::unique_ptr<std::mutex> mutex;
};

} // namespace v2
