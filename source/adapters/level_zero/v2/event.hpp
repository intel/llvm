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

struct event_profiling_data_t {
  event_profiling_data_t(ze_event_handle_t hZeEvent) : hZeEvent(hZeEvent) {}

  void recordStartTimestamp(ur_device_handle_t hDevice);
  uint64_t getEventStartTimestmap() const;

  uint64_t getEventEndTimestamp();
  uint64_t *eventEndTimestampAddr();

  bool recordingStarted() const;
  bool recordingEnded() const;

private:
  ze_event_handle_t hZeEvent;

  uint64_t adjustedEventStartTimestamp = 0;
  uint64_t recordEventEndTimestamp = 0;
  uint64_t adjustedEventEndTimestamp = 0;

  uint64_t zeTimerResolution = 0;
  uint64_t timestampMaxValue = 0;
};

struct ur_event_handle_t_ : _ur_object {
public:
  ur_event_handle_t_(ur_context_handle_t hContext, ze_event_handle_t hZeEvent,
                     v2::event_flags_t flags);

  // Set the queue and command that this event is associated with
  void resetQueueAndCommand(ur_queue_handle_t hQueue, ur_command_t commandType);

  // releases event immediately
  virtual ur_result_t forceRelease() = 0;
  virtual ~ur_event_handle_t_() = default;

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
  ur_queue_handle_t getQueue() const;

  // Context associated with this event
  ur_context_handle_t getContext() const;

  // Get the type of the command that this event is associated with
  ur_command_t getCommandType() const;

  // Record the start timestamp of the event, to be obtained by
  // urEventGetProfilingInfo. resetQueueAndCommand should be
  // called before this.
  void recordStartTimestamp();

  // Get pointer to the end timestamp, and ze event handle.
  // Caller is responsible for signaling the event once the timestamp is ready.
  std::pair<uint64_t *, ze_event_handle_t> getEventEndTimestampAndHandle();

  uint64_t getEventStartTimestmap() const;
  uint64_t getEventEndTimestamp();

protected:
  ur_context_handle_t hContext;

  // non-owning handle to the L0 event
  const ze_event_handle_t hZeEvent;

  // queue and commandType that this event is associated with, set by enqueue
  // commands
  ur_queue_handle_t hQueue = nullptr;
  ur_command_t commandType = UR_COMMAND_FORCE_UINT32;

  v2::event_flags_t flags;
  event_profiling_data_t profilingData;
};

struct ur_pooled_event_t : ur_event_handle_t_ {
  ur_pooled_event_t(ur_context_handle_t hContext,
                    v2::raii::cache_borrowed_event eventAllocation,
                    v2::event_pool *pool);

  ur_result_t forceRelease() override;

private:
  v2::raii::cache_borrowed_event zeEvent;
  v2::event_pool *pool;
};

struct ur_native_event_t : ur_event_handle_t_ {
  ur_native_event_t(ur_native_handle_t hNativeEvent,
                    ur_context_handle_t hContext,
                    const ur_event_native_properties_t *pProperties);

  ur_result_t forceRelease() override;

private:
  v2::raii::ze_event_handle_t zeEvent;
};
