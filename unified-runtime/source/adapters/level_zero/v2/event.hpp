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

#include "adapters/level_zero/v2/queue_api.hpp"
#include "common.hpp"
#include "event_provider.hpp"

namespace v2 {
class event_pool;
}

struct event_profiling_data_t {
  event_profiling_data_t(ze_event_handle_t hZeEvent) : hZeEvent(hZeEvent) {}

  void recordStartTimestamp(ur_device_handle_t hDevice);
  uint64_t getEventStartTimestmap() const;

  uint64_t getEventEndTimestamp();
  uint64_t *eventEndTimestampAddr();

  bool recordingStarted() const;
  bool recordingEnded() const;

  // clear the profiling data, allowing the event to be reused
  // for a new command
  void reset();

private:
  ze_event_handle_t hZeEvent;

  uint64_t adjustedEventStartTimestamp = 0;
  uint64_t recordEventEndTimestamp = 0;
  uint64_t adjustedEventEndTimestamp = 0;

  uint64_t zeTimerResolution = 0;
  uint64_t timestampMaxValue = 0;
};

struct ur_event_handle_t_ : ur_object {
public:
  // cache_borrowed_event is used for pooled events, whilst ze_event_handle_t is
  // used for native events
  using event_variant =
      std::variant<v2::raii::cache_borrowed_event, v2::raii::ze_event_handle_t>;

  ur_event_handle_t_(ur_context_handle_t hContext,
                     v2::raii::cache_borrowed_event eventAllocation,
                     v2::event_pool *pool);

  ur_event_handle_t_(ur_context_handle_t hContext,
                     ur_native_handle_t hNativeEvent,
                     const ur_event_native_properties_t *pProperties);

  // Set the queue and command that this event is associated with
  void setQueue(ur_queue_t_ *hQueue);
  void setCommandType(ur_command_t commandType);

  void reset();
  ze_event_handle_t getZeEvent() const;

  ur_result_t retain();

  // releases event immediately, or adds to a list for deffered deletion
  ur_result_t release();

  // releases a signaled and no longer in-use event, that's on the
  // deffered events list in the queue
  ur_result_t releaseDeferred();

  // Tells if this event was created as a timestamp event, allowing profiling
  // info even if profiling is not enabled.
  bool isTimestamped() const;

  // Tells if this event comes from a pool that has profiling enabled.
  bool isProfilingEnabled() const;

  // Queue associated with this event. Can be nullptr (for native events)
  ur_queue_t_ *getQueue() const;

  // Context associated with this event
  ur_context_handle_t getContext() const;

  // Get the type of the command that this event is associated with
  ur_command_t getCommandType() const;

  // Get the device associated with this event
  ur_device_handle_t getDevice() const;

  // Record the start timestamp of the event, to be obtained by
  // urEventGetProfilingInfo. setQueue should be
  // called before this.
  void recordStartTimestamp();

  // Get pointer to the end timestamp, and ze event handle.
  // Caller is responsible for signaling the event once the timestamp is ready.
  std::pair<uint64_t *, ze_event_handle_t> getEventEndTimestampAndHandle();

  uint64_t getEventStartTimestmap() const;
  uint64_t getEventEndTimestamp();

private:
  ur_event_handle_t_(ur_context_handle_t hContext, event_variant hZeEvent,
                     v2::event_flags_t flags, v2::event_pool *pool);

protected:
  ur_context_handle_t hContext;

  // Pool is used if and only if this is a pooled event
  v2::event_pool *event_pool = nullptr;
  event_variant hZeEvent;

  // queue and commandType that this event is associated with, set by enqueue
  // commands
  ur_queue_t_ *hQueue = nullptr;
  ur_command_t commandType = UR_COMMAND_FORCE_UINT32;
  ur_device_handle_t hDevice = nullptr;

  v2::event_flags_t flags;
  event_profiling_data_t profilingData;
};
