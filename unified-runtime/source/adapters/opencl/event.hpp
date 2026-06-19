//===--------- queue.hpp - OpenCL Adapter ---------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "common.hpp"
#include "common/ur_ref_count.hpp"
#include "queue.hpp"

#include <vector>

namespace ur::opencl {

struct ur_event_handle_t_ : handle_base {
  using native_type = cl_event;
  native_type CLEvent;
  ur_context_handle_t_ *Context;
  ur_queue_handle_t_ *Queue;
  bool IsNativeHandleOwned = true;
  ur::RefCount RefCount;

  ur_event_handle_t_(native_type Event, ur_context_handle_t_ *Ctx,
                     ur_queue_handle_t_ *Queue)
      : handle_base(), CLEvent(Event), Context(Ctx), Queue(Queue) {
    ur::opencl::urContextRetain(ur_cast<ur_context_handle_t>(Context));
    if (Queue) {
      ur::opencl::urQueueRetain(ur_cast<ur_queue_handle_t>(Queue));
    }
  }

  ~ur_event_handle_t_() {
    ur::opencl::urContextRelease(ur_cast<ur_context_handle_t>(Context));
    if (Queue) {
      ur::opencl::urQueueRelease(ur_cast<ur_queue_handle_t>(Queue));
    }
    if (IsNativeHandleOwned) {
      clReleaseEvent(CLEvent);
    }
  }

  ur_result_t ensureQueue() {
    if (!Queue) {
      cl_command_queue native_queue;
      CL_RETURN_ON_FAILURE(clGetEventInfo(CLEvent, CL_EVENT_COMMAND_QUEUE,
                                          sizeof(native_queue), &native_queue,
                                          nullptr));
      ur_queue_handle_t OpaqueQueue = nullptr;
      UR_RETURN_ON_FAILURE(ur_queue_handle_t_::makeWithNative(
          native_queue, ur_cast<ur_context_handle_t>(Context), nullptr,
          OpaqueQueue));
      Queue = ur_cast<ur_queue_handle_t_ *>(OpaqueQueue);
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
  auto UrQueue = ur_cast<ur_queue_handle_t_ *>(Queue);
  if (ReturnedEvent) {
    try {
      auto UREvent = std::make_unique<ur_event_handle_t_>(
          Event, ur_cast<ur_context_handle_t_ *>(Context), UrQueue);
      *ReturnedEvent = ur_cast<ur_event_handle_t>(UREvent.release());
      UR_RETURN_ON_FAILURE(UrQueue->storeLastEvent(*ReturnedEvent));
    } catch (std::bad_alloc &) {
      return UR_RESULT_ERROR_OUT_OF_RESOURCES;
    } catch (...) {
      return UR_RESULT_ERROR_UNKNOWN;
    }
  } else {
    UR_RETURN_ON_FAILURE(UrQueue->storeLastEvent(nullptr));
  }
  return UR_RESULT_SUCCESS;
}

} // namespace ur::opencl
