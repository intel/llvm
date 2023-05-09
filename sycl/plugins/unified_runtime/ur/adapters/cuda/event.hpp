//===--------- event.hpp - CUDA Adapter ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include <cuda.h>
#include <ur/ur.hpp>

#include "queue.hpp"

/// UR Event mapping to CUevent
///
struct ur_event_handle_t_ {
public:
  using native_type = CUevent;

  ur_result_t record();

  ur_result_t wait();

  ur_result_t start();

  native_type get() const noexcept { return evEnd_; };

  ur_queue_handle_t get_queue() const noexcept { return queue_; }

  CUstream get_stream() const noexcept { return stream_; }

  uint32_t get_compute_stream_token() const noexcept { return streamToken_; }

  ur_command_t get_command_type() const noexcept { return commandType_; }

  uint32_t get_reference_count() const noexcept { return refCount_; }

  bool is_recorded() const noexcept { return isRecorded_; }

  bool is_started() const noexcept { return isStarted_; }

  bool is_completed() const noexcept;

  uint32_t get_execution_status() const noexcept {

    if (!is_recorded()) {
      return UR_EVENT_STATUS_SUBMITTED;
    }

    if (!is_completed()) {
      return UR_EVENT_STATUS_RUNNING;
    }
    return UR_EVENT_STATUS_COMPLETE;
  }

  ur_context_handle_t get_context() const noexcept { return context_; };

  uint32_t increment_reference_count() { return ++refCount_; }

  uint32_t decrement_reference_count() { return --refCount_; }

  uint32_t get_event_id() const noexcept { return eventId_; }

  bool backend_has_ownership() const noexcept { return has_ownership_; }

  // Returns the counter time when the associated command(s) were enqueued
  //
  uint64_t get_queued_time() const;

  // Returns the counter time when the associated command(s) started execution
  //
  uint64_t get_start_time() const;

  // Returns the counter time when the associated command(s) completed
  //
  uint64_t get_end_time() const;

  // construct a native CUDA. This maps closely to the underlying CUDA event.
  static ur_event_handle_t
  make_native(ur_command_t type, ur_queue_handle_t queue, CUstream stream,
              uint32_t stream_token = std::numeric_limits<uint32_t>::max()) {
    return new ur_event_handle_t_(type, queue->get_context(), queue, stream,
                                  stream_token);
  }

  static ur_event_handle_t make_with_native(ur_context_handle_t context,
                                            CUevent eventNative) {
    return new ur_event_handle_t_(context, eventNative);
  }

  ur_result_t release();

  ~ur_event_handle_t_();

private:
  // This constructor is private to force programmers to use the make_native /
  // make_user static members in order to create a pi_event for CUDA.
  ur_event_handle_t_(ur_command_t type, ur_context_handle_t context,
                     ur_queue_handle_t queue, CUstream stream,
                     uint32_t stream_token);

  // This constructor is private to force programmers to use the
  // make_with_native for event introp
  ur_event_handle_t_(ur_context_handle_t context, CUevent eventNative);

  ur_command_t commandType_; // The type of command associated with event.

  std::atomic_uint32_t refCount_; // Event reference count.

  bool has_ownership_; // Signifies if event owns the native type.

  bool hasBeenWaitedOn_; // Signifies whether the event has been waited
                         // on through a call to wait(), which implies
                         // that it has completed.

  bool isRecorded_; // Signifies wether a native CUDA event has been recorded
                    // yet.
  bool isStarted_;  // Signifies wether the operation associated with the
                    // PI event has started or not
                    //

  uint32_t streamToken_;
  uint32_t eventId_; // Queue identifier of the event.

  native_type evEnd_; // CUDA event handle. If this _pi_event represents a user
                      // event, this will be nullptr.

  native_type evStart_; // CUDA event handle associated with the start

  native_type evQueued_; // CUDA event handle associated with the time
                         // the command was enqueued

  ur_queue_handle_t queue_; // pi_queue associated with the event. If this is a
                            // user event, this will be nullptr.

  CUstream stream_; // CUstream associated with the event. If this is a user
                    // event, this will be uninitialized.

  ur_context_handle_t context_; // pi_context associated with the event. If this
                                // is a native event, this will be the same
                                // context associated with the queue_ member.
};

// Iterates over the event wait list, returns correct ur_result_t error codes.
// Invokes the callback for the latest event of each queue in the wait list.
// The callback must take a single pi_event argument and return a ur_result_t.
template <typename Func>
ur_result_t forLatestEvents(const ur_event_handle_t *event_wait_list,
                            std::size_t num_events_in_wait_list, Func &&f) {

  if (event_wait_list == nullptr || num_events_in_wait_list == 0) {
    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
  }

  // Fast path if we only have a single event
  if (num_events_in_wait_list == 1) {
    return f(event_wait_list[0]);
  }

  std::vector<ur_event_handle_t> events{
      event_wait_list, event_wait_list + num_events_in_wait_list};
  std::sort(events.begin(), events.end(),
            [](ur_event_handle_t e0, ur_event_handle_t e1) {
              // Tiered sort creating sublists of streams (smallest value first)
              // in which the corresponding events are sorted into a sequence of
              // newest first.
              return e0->get_stream() < e1->get_stream() ||
                     (e0->get_stream() == e1->get_stream() &&
                      e0->get_event_id() > e1->get_event_id());
            });

  bool first = true;
  CUstream lastSeenStream = 0;
  for (ur_event_handle_t event : events) {
    if (!event || (!first && event->get_stream() == lastSeenStream)) {
      continue;
    }

    first = false;
    lastSeenStream = event->get_stream();

    auto result = f(event);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  return UR_RESULT_SUCCESS;
}
