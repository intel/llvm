//===--------- queue.hpp - OpenCL Adapter ---------------------------===//
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

#include <vector>

struct ur_event_handle_t_ : cl_adapter::ur_handle_t_ {
  using native_type = cl_event;
  native_type CLEvent;
  ur_context_handle_t Context;
  ur_queue_handle_t Queue;
  std::atomic<uint32_t> RefCount = 0;
  bool IsNativeHandleOwned = true;

  ur_event_handle_t_(native_type Event, ur_context_handle_t Ctx,
                     ur_queue_handle_t Queue)
      : cl_adapter::ur_handle_t_(), CLEvent(Event), Context(Ctx), Queue(Queue) {
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
};
