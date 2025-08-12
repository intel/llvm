//===----------- event.hpp - LLVM Offload Adapter  ------------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <OffloadAPI.h>
#include <ur_api.h>

#include "common.hpp"

struct ur_event_handle_t_ : RefCounted {
  ol_event_handle_t OffloadEvent;
  ur_command_t Type;
  ur_queue_handle_t UrQueue;

  ur_event_handle_t_(ur_command_t Type, ur_queue_handle_t Queue)
      : Type(Type), UrQueue(Queue) {}

  static ur_event_handle_t createEmptyEvent(ur_command_t Type,
                                            ur_queue_handle_t Queue) {
    auto *Event = new ur_event_handle_t_(Type, Queue);
    // Null event represents an empty event. Waiting on it is a no-op.
    Event->OffloadEvent = nullptr;

    return Event;
  }
};
