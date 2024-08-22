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
#include "ur_api.h"

namespace v2 {

static constexpr size_t EVENTS_BURST = 64;

ur_event_handle_t_ *event_pool::allocate() {
  if (freelist.empty()) {
    auto start = events.size();
    auto end = start + EVENTS_BURST;
    for (; start < end; ++start) {
      events.emplace_back(provider->allocate(), this);
      freelist.push_back(&events.at(start));
    }
  }

  auto event = freelist.back();
  freelist.pop_back();

  return event;
}

void event_pool::free(ur_event_handle_t_ *event) {
  event->reset();
  freelist.push_back(event);
}

event_provider *event_pool::getProvider() { return provider.get(); }

} // namespace v2
