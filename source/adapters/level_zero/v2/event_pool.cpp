//===--------- event_pool.cpp - Level Zero Adapter ------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "ur_api.h"
#include <event_pool.hpp>

namespace v2 {

static constexpr size_t EVENTS_BURST = 64;

ur_event *event_pool::allocate() {
  if (freelist.empty()) {
    auto start = events.size();
    auto end = start + EVENTS_BURST;
    events.resize(end);
    for (; start < end; ++start) {
      freelist.push_back(&events.at(start));
    }
  }

  auto event = freelist.back();

  auto ZeEvent = provider->allocate();
  event->attachZeHandle(std::move(ZeEvent));

  freelist.pop_back();

  return event;
}

void event_pool::free(ur_event *event) {
  auto _ = event->detachZeHandle();

  freelist.push_back(event);
}

} // namespace v2
