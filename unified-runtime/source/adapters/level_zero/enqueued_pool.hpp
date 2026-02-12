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
#include <map>
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
  using memory_free_callback_t = std::function<ur_result_t(void *)>;

  EnqueuedPool(event_release_callback_t EventReleaseFn,
               memory_free_callback_t MemFreeFn)
      : EventReleaseFn(std::move(EventReleaseFn)),
        MemFreeFn(std::move(MemFreeFn)) {}

  ~EnqueuedPool();
  std::optional<Allocation> getBestFit(size_t Size, size_t Alignment,
                                       void *Queue);
  void insert(void *Ptr, size_t Size, ur_event_handle_t Event, void *Queue);
  bool cleanup();
  bool cleanupForQueue(void *Queue);

  // Allocations are grouped by queue and alignment.
  struct AllocationGroupKey {
    void *Queue;
    size_t Alignment;
  };

  struct GroupComparator {
    bool operator()(const AllocationGroupKey &lhs,
                    const AllocationGroupKey &rhs) const {
      if (lhs.Queue != rhs.Queue) {
        return lhs.Queue < rhs.Queue;
      }
      return lhs.Alignment < rhs.Alignment;
    }
  };

  // Then, the allocations are sorted by size.
  struct SizeComparator {
    bool operator()(const Allocation &lhs, const Allocation &rhs) const {
      if (lhs.Size != rhs.Size) {
        return lhs.Size < rhs.Size;
      }
      return lhs.Ptr < rhs.Ptr;
    }
  };

  using AllocationGroup = std::set<Allocation, SizeComparator>;
  using AllocationGroupMap =
      std::map<AllocationGroupKey, AllocationGroup, GroupComparator>;

private:
  ur_mutex Mutex;

  // Freelist grouped by queue and alignment.
  AllocationGroupMap FreelistByQueue;
  // Freelist grouped by alignment only.
  AllocationGroupMap FreelistGlobal;

  event_release_callback_t EventReleaseFn;
  memory_free_callback_t MemFreeFn;
};
