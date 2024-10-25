//===----------- queue.hpp - Native CPU Adapter ---------------------------===//
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
#include "event.hpp"
#include "ur_api.h"
#include <set>

struct ur_queue_handle_t_ : RefCounted {
  ur_queue_handle_t_(ur_device_handle_t device, ur_context_handle_t context,
                     const ur_queue_properties_t *pProps)
      : device(device), context(context),
        inOrder(pProps ? !(pProps->flags &
                           UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE)
                       : true),
        profilingEnabled(pProps ? pProps->flags & UR_QUEUE_FLAG_PROFILING_ENABLE
                                : false) {}

  ur_device_handle_t getDevice() const { return device; }

  ur_context_handle_t getContext() const { return context; }

  void addEvent(ur_event_handle_t event) { events.insert(event); }

  void removeEvent(ur_event_handle_t event) { events.erase(event); }

  void finish() {
    while (!events.empty()) {
      auto ev = *events.begin();
      // ur_event_handle_t_::wait removes itself from the events set in the
      // queue
      ev->wait();
    }
    events.clear();
  }

  ~ur_queue_handle_t_() { finish(); }

  bool isInOrder() const { return inOrder; }

  bool isProfiling() const { return profilingEnabled; }

private:
  ur_device_handle_t device;
  ur_context_handle_t context;
  std::set<ur_event_handle_t> events;
  const bool inOrder;
  const bool profilingEnabled;
};
