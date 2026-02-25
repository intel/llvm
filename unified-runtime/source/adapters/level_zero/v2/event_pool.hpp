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
      : hContext(hContext), provider(std::move(Provider)) {};

  event_pool(event_pool &&other) = delete;
  event_pool &operator=(event_pool &&other) = delete;

  event_pool(const event_pool &) = delete;
  event_pool &operator=(const event_pool &) = delete;

  ~event_pool() = default;

  // Allocate an event from the pool. Thread safe.
  ur_event_handle_t allocate();

  // Free an event back to the pool. Thread safe.
  void free(ur_event_handle_t event);

  event_provider *getProvider() const;
  event_flags_t getFlags() const;

private:
  ur_context_handle_t hContext;
  std::unique_ptr<event_provider> provider;

  std::deque<ur_event_handle_t_> events;
  std::vector<ur_event_handle_t> freelist;

  ur_mutex mutex;
};

// Only create an event when requested by the user.
static inline ur_event_handle_t
createEventIfRequested(event_pool *eventPool, ur_event_handle_t *phEvent,
                       ur_queue_t_ *queue) {
  if (phEvent == nullptr) {
    return nullptr;
  }

  (*phEvent) = eventPool->allocate();
  (*phEvent)->setQueue(queue);
  return (*phEvent);
}

// Always creates an event (used in functions that need to store the event
// internally).
static inline ur_event_handle_t createEvent(event_pool *eventPool,
                                            ur_event_handle_t *phEvent,
                                            ur_queue_t_ *queue) {
  auto hEvent = eventPool->allocate();
  hEvent->setQueue(queue);

  if (phEvent) {
    (*phEvent) = hEvent;
  }

  return hEvent;
}

// Always creates an event (used in functions that need to store the event
// internally). If event was requested by the user, also increase ref count of
// that event to avoid pre-mature release.
static inline ur_event_handle_t createEventAndRetain(event_pool *eventPool,
                                                     ur_event_handle_t *phEvent,
                                                     ur_queue_t_ *queue) {
  auto *event = createEvent(eventPool, phEvent, queue);
  if (phEvent) {
    (*phEvent)->retain();
  }
  return event;
}

} // namespace v2
