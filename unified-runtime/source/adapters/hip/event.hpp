//===--------- event.hpp - HIP Adapter ------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "common.hpp"
#include "queue.hpp"

/// UR Event mapping to hipEvent_t
struct ur_event_handle_t_ : ur::hip::handle_base {
  using native_type = hipEvent_t;

  ur_event_handle_t_(
      ur_command_t Type, ur_queue_handle_t Queue, hipStream_t Stream,
      uint32_t StreamToken = std::numeric_limits<uint32_t>::max());
  ur_event_handle_t_(ur_context_handle_t Context, hipEvent_t EventNative);
  ~ur_event_handle_t_();
  ur_result_t release();

  ur_result_t start();
  ur_result_t record();
  ur_result_t wait();

  bool isRecorded() const noexcept { return IsRecorded; }
  bool isStarted() const noexcept { return IsStarted; }
  bool isCompleted() const;

  uint64_t getQueuedTime() const;
  uint64_t getStartTime() const;
  uint64_t getEndTime() const;
  uint32_t getExecutionStatus() const {
    if (!isRecorded()) {
      return UR_EVENT_STATUS_SUBMITTED;
    }

    if (!isCompleted()) {
      return UR_EVENT_STATUS_RUNNING;
    }
    return UR_EVENT_STATUS_COMPLETE;
  }

  bool isInterop() const noexcept { return IsInterop; };
  bool hasProfiling() const noexcept { return HasProfiling; }

  // Basic getters.
  native_type get() const noexcept { return EvEnd; };
  ur_queue_handle_t getQueue() const noexcept { return Queue; }
  hipStream_t getStream() const noexcept { return Stream; }
  uint32_t getComputeStreamToken() const noexcept { return StreamToken; }
  ur_command_t getCommandType() const noexcept { return CommandType; }
  ur_context_handle_t getContext() const noexcept { return Context; };
  uint32_t getEventId() const noexcept { return EventId; }

  // Reference counting.
  uint32_t getReferenceCount() const noexcept { return RefCount; }
  uint32_t incrementReferenceCount() { return ++RefCount; }
  uint32_t decrementReferenceCount() { return --RefCount; }

private:
  ur_command_t CommandType; // The type of command associated with event.

  std::atomic_uint32_t RefCount{1}; // Event reference count.

  bool HasOwnership{true};  // Signifies if event owns the native type.
  bool HasProfiling{false}; // Signifies if event has profiling information.

  bool HasBeenWaitedOn{false}; // Signifies whether the event has been waited
                               // on through a call to wait(), which implies
                               // that it has completed.

  bool IsRecorded{false}; // Signifies wether a native HIP event has been
                          // recorded yet.
  bool IsStarted{false};  // Signifies wether the operation associated with the
                          // UR event has started or not

  const bool IsInterop{false}; // Made with urEventCreateWithNativeHandle

  uint32_t StreamToken;
  uint32_t EventId{0}; // Queue identifier of the event.

  native_type EvEnd{nullptr};    // Native event if IsInterop.
  native_type EvStart{nullptr};  // Profiling event for command start.
  native_type EvQueued{nullptr}; // Profiling even for command enqueue.

  ur_queue_handle_t Queue; // ur_queue_handle_t associated with the event. If
                           // this is a user event, this will be nullptr.

  hipStream_t Stream; // hipStream_t associated with the event. If this is a
                      // user event, this will be uninitialized.

  ur_context_handle_t Context; // ur_context_handle_t associated with the event.
                               // If this is a native event, this will be the
                               // same context associated with the Queue member.
};

// Iterate over `EventWaitList` and apply the given callback `F` to the
// latest event on each queue therein. The callback must take a single
// ur_event_handle_t argument and return a ur_result_t. If the callback returns
// an error, the iteration terminates and the error is returned.
template <typename Func>
ur_result_t forLatestEvents(const ur_event_handle_t *EventWaitList,
                            size_t NumEventsInWaitList, Func &&F) {

  if (EventWaitList == nullptr || NumEventsInWaitList == 0) {
    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
  }

  // Fast path if we only have a single event
  if (NumEventsInWaitList == 1) {
    return F(EventWaitList[0]);
  }

  std::vector<ur_event_handle_t> Events{EventWaitList,
                                        EventWaitList + NumEventsInWaitList};
  std::sort(Events.begin(), Events.end(),
            [](ur_event_handle_t E0, ur_event_handle_t E1) {
              // Tiered sort creating sublists of streams (smallest value first)
              // in which the corresponding events are sorted into a sequence of
              // newest first.
              return E0->getStream() < E1->getStream() ||
                     (E0->getStream() == E1->getStream() &&
                      E0->getEventId() > E1->getEventId());
            });

  hipStream_t LastSeenStream = 0;
  for (size_t i = 0; i < Events.size(); i++) {
    auto Event = Events[i];
    if (!Event || (i != 0 && !Event->isInterop() &&
                   Event->getStream() == LastSeenStream)) {
      continue;
    }

    LastSeenStream = Event->getStream();

    auto Result = F(Event);
    if (Result != UR_RESULT_SUCCESS) {
      return Result;
    }
  }

  return UR_RESULT_SUCCESS;
}
