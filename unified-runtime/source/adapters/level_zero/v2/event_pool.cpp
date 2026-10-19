//===--------- event_pool.cpp - Level Zero Adapter ------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "event_pool.hpp"
#include "common/latency_tracker.hpp"
#include "event.hpp"
#include "queue_api.hpp"
#include "unified-runtime/ur_api.h"

#include <algorithm>

namespace ur::level_zero::v2 {

ur_event_handle_t event_pool::allocate() {
  TRACK_SCOPE_LATENCY("event_pool::allocate");

  std::unique_lock<ur_mutex> lock(mutex);

  auto it = std::find_if(
      freelist.rbegin(), freelist.rend(), [](ur_event_handle_t event) {
        return !event->isCounter() ||
               ZE_CALL_NOCHECK(zeEventQueryStatus, (event->getZeEvent())) ==
                   ZE_RESULT_SUCCESS;
      });

  ur_event_handle_t event;
  if (it == freelist.rend()) {
    events.emplace_back(hContext, provider->allocate(), this);
    event = &events.back();
  } else {
    event = *it;
    *it = freelist.back();
    freelist.pop_back();
  }

#ifndef NDEBUG
  // Set the command type to an invalid value to catch any misuses in tests
  event->setQueue(nullptr);
  event->setCommandType(UR_COMMAND_FORCE_UINT32);
#endif

  return event;
}

ur_event_handle_t event_pool::allocateDetached() {
  TRACK_SCOPE_LATENCY("event_pool::allocateDetached");
  raii::ze_event_handle_t ownedEvent(provider->allocate().release(),
                                     /*ownZeHandle=*/true);
  return new ur_event_handle_t_(hContext, std::move(ownedEvent),
                                provider->eventFlags());
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

} // namespace ur::level_zero::v2
