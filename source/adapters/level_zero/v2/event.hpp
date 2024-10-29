//===--------- event.hpp - Level Zero Adapter -----------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <stack>

#include <ur/ur.hpp>
#include <ur_api.h>
#include <ze_api.h>

#include "common.hpp"
#include "event_provider.hpp"

namespace v2 {
class event_pool;
}

struct ur_event_handle_t_ : _ur_object {
public:
  ur_event_handle_t_(v2::raii::cache_borrowed_event eventAllocation,
                     v2::event_pool *pool);

  void reset();
  ze_event_handle_t getZeEvent() const;

  ur_result_t retain();
  ur_result_t release();

  // Tells if this event was created as a timestamp event, allowing profiling
  // info even if profiling is not enabled.
  bool isTimestamped() const;

  // Tells if this event comes from a pool that has profiling enabled.
  bool isProfilingEnabled() const;

  // Device associated with this event
  ur_device_handle_t getDevice() const;

  void recordStartTimestamp();
  uint64_t *getEventEndTimestampPtr();

  uint64_t getEventStartTimestmap() const;
  uint64_t getEventEndTimestamp();

private:
  v2::raii::cache_borrowed_event zeEvent;
  v2::event_pool *pool;

  uint64_t adjustedEventStartTimestamp;
  uint64_t recordEventEndTimestamp;
  uint64_t adjustedEventEndTimestamp;

  const uint64_t zeTimerResolution;
  const uint64_t timestampMaxValue;
};
