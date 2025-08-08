//===--------- event_pool.cpp - Level Zero Adapter ------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "event_pool.hpp"
#include "common/latency_tracker.hpp"
#include "event.hpp"
#include "queue_api.hpp"
#include "ur_api.h"

namespace v2 {

static constexpr size_t EVENTS_BURST = 64;

ur_event_handle_t event_pool::allocate() {
  TRACK_SCOPE_LATENCY("event_pool::allocate");

  std::unique_lock<ur_mutex> lock(mutex);

  if (freelist.empty()) {
    auto start = events.size();
    auto end = start + EVENTS_BURST;
    for (; start < end; ++start) {
      events.emplace_back(hContext, provider->allocate(), this);
      freelist.push_back(&events.at(start));
    }
  }

  auto event = freelist.back();
  freelist.pop_back();

#ifndef NDEBUG
  // Set the command type to an invalid value to catch any misuses in tests
  event->setQueue(nullptr);
  event->setCommandType(UR_COMMAND_FORCE_UINT32);
#endif

  return event;
}

void event_pool::free(ur_event_handle_t event) {
  TRACK_SCOPE_LATENCY("event_pool::free");

  std::unique_lock<ur_mutex> lock(mutex);

  event->reset();
  freelist.push_back(event);

  // The event is still in the pool, so we need to increment the refcount
  assert(event->RefCount.getCount() == 0);
  event->RefCount.retain();
}

event_provider *event_pool::getProvider() const { return provider.get(); }

event_flags_t event_pool::getFlags() const {
  return getProvider()->eventFlags();
}

} // namespace v2
