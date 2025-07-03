//===--------- enqueued_pool.hpp - Level Zero Adapter ---------------------===//
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

#include "ur_api.h"
#include "ur_pool_manager.hpp"
#include <set>
#include <umf_helpers.hpp>

class EnqueuedPool {
public:
  struct Allocation {
    void *Ptr;
    size_t Size;
    ur_event_handle_t Event;
    // Queue handle, used as an identifier for the associated queue.
    // This can either be a `ur_queue_handle_t` or a pointer to a v2 queue
    // object.
    void *Queue;
    size_t Alignment;
  };

  using event_release_callback_t = ur_result_t (*)(ur_event_handle_t);

  EnqueuedPool(event_release_callback_t eventRelease)
      : eventRelease(eventRelease) {}

  ~EnqueuedPool();
  std::optional<Allocation> getBestFit(size_t Size, size_t Alignment,
                                       void *Queue);
  void insert(void *Ptr, size_t Size, ur_event_handle_t Event, void *Queue);
  bool cleanup();
  bool cleanupForQueue(void *Queue);

private:
  struct Comparator {
    bool operator()(const Allocation &lhs, const Allocation &rhs) const {
      if (lhs.Queue != rhs.Queue) {
        return lhs.Queue < rhs.Queue; // Compare by queue handle first
      }
      if (lhs.Alignment != rhs.Alignment) {
        return lhs.Alignment < rhs.Alignment; // Then by alignment
      }
      if (lhs.Size != rhs.Size) {
        return lhs.Size < rhs.Size; // Then by size
      }
      return lhs.Ptr < rhs.Ptr; // Finally by pointer address
    }
  };

  using AllocationSet = std::set<Allocation, Comparator>;
  ur_mutex Mutex;
  AllocationSet Freelist;
  event_release_callback_t eventRelease;
};
