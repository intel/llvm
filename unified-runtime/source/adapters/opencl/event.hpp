//===--------- queue.hpp - OpenCL Adapter ---------------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "common.hpp"
#include "queue.hpp"

#include <vector>

struct ur_event_handle_t_ : ur::opencl::handle_base {
  using native_type = cl_event;
  native_type CLEvent;
  ur_context_handle_t Context;
  ur_queue_handle_t Queue;
  std::atomic<uint32_t> RefCount = 0;
  bool IsNativeHandleOwned = true;

  ur_event_handle_t_(native_type Event, ur_context_handle_t Ctx,
                     ur_queue_handle_t Queue)
      : handle_base(), CLEvent(Event), Context(Ctx), Queue(Queue) {
    RefCount = 1;
    urContextRetain(Context);
    if (Queue) {
      urQueueRetain(Queue);
    }
  }

  ~ur_event_handle_t_() {
    urContextRelease(Context);
    if (Queue) {
      urQueueRelease(Queue);
    }
    if (IsNativeHandleOwned) {
      clReleaseEvent(CLEvent);
    }
  }

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }

  uint32_t decrementReferenceCount() noexcept { return --RefCount; }

  uint32_t getReferenceCount() const noexcept { return RefCount; }

  ur_result_t ensureQueue() {
    if (!Queue) {
      cl_command_queue native_queue;
      CL_RETURN_ON_FAILURE(clGetEventInfo(CLEvent, CL_EVENT_COMMAND_QUEUE,
                                          sizeof(native_queue), &native_queue,
                                          nullptr));
      UR_RETURN_ON_FAILURE(ur_queue_handle_t_::makeWithNative(
          native_queue, Context, nullptr, Queue));
    }

    return UR_RESULT_SUCCESS;
  }
};

inline cl_event *ifUrEvent(ur_event_handle_t *ReturnedEvent, cl_event &Event) {
  return ReturnedEvent ? &Event : nullptr;
}

inline ur_result_t createUREvent(cl_event Event, ur_context_handle_t Context,
                                 ur_queue_handle_t Queue,
                                 ur_event_handle_t *ReturnedEvent) {
  assert(Queue);
  if (ReturnedEvent) {
    try {
      auto UREvent =
          std::make_unique<ur_event_handle_t_>(Event, Context, Queue);
      *ReturnedEvent = UREvent.release();
      UR_RETURN_ON_FAILURE(Queue->storeLastEvent(*ReturnedEvent));
    } catch (std::bad_alloc &) {
      return UR_RESULT_ERROR_OUT_OF_RESOURCES;
    } catch (...) {
      return UR_RESULT_ERROR_UNKNOWN;
    }
  } else {
    UR_RETURN_ON_FAILURE(Queue->storeLastEvent(nullptr));
  }
  return UR_RESULT_SUCCESS;
}
