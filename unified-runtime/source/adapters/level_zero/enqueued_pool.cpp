//===--------- enqueued_pool.cpp - Level Zero Adapter ---------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "enqueued_pool.hpp"

#include <ur_api.h>

EnqueuedPool::~EnqueuedPool() { cleanup(); }

std::optional<EnqueuedPool::Allocation> EnqueuedPool::getBestFit(size_t Size,
                                                                 void *Queue) {
  auto Lock = std::lock_guard(Mutex);

  Allocation Alloc = {nullptr, Size, nullptr, Queue};

  auto It = Freelist.lower_bound(Alloc);
  if (It != Freelist.end() && It->Size >= Size && It->Queue == Queue) {
    Allocation BestFit = *It;
    Freelist.erase(It);

    return BestFit;
  }

  return std::nullopt;
}

void EnqueuedPool::insert(void *Ptr, size_t Size, ur_event_handle_t Event,
                          void *Queue) {
  auto Lock = std::lock_guard(Mutex);

  Freelist.emplace(Allocation{Ptr, Size, Event, Queue});
}

bool EnqueuedPool::cleanup() {
  auto Lock = std::lock_guard(Mutex);
  auto FreedAllocations = !Freelist.empty();

  auto Ret [[maybe_unused]] = UR_RESULT_SUCCESS;
  for (auto It : Freelist) {
    Ret = MemFreeFn(It.Ptr);
    assert(Ret == UR_RESULT_SUCCESS);

    if (It.Event)
      EventReleaseFn(It.Event);
  }
  Freelist.clear();

  return FreedAllocations;
}

bool EnqueuedPool::cleanupForQueue(void *Queue) {
  auto Lock = std::lock_guard(Mutex);

  Allocation Alloc = {nullptr, 0, nullptr, Queue};
  // first allocation on the freelist with the specific queue
  auto It = Freelist.lower_bound(Alloc);

  bool FreedAllocations = false;

  auto Ret [[maybe_unused]] = UR_RESULT_SUCCESS;
  while (It != Freelist.end() && It->Queue == Queue) {
    Ret = MemFreeFn(It->Ptr);
    assert(Ret == UR_RESULT_SUCCESS);

    if (It->Event)
      EventReleaseFn(It->Event);

    // Erase the current allocation and move to the next one
    It = Freelist.erase(It);
    FreedAllocations = true;
  }

  return FreedAllocations;
}
