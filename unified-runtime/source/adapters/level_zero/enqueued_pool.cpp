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

std::optional<EnqueuedPool::Allocation>
EnqueuedPool::getBestFit(size_t Size, size_t Alignment, void *Queue) {
  auto Lock = std::lock_guard(Mutex);

  Allocation Alloc = {nullptr, Size, nullptr, Queue, Alignment};

  auto It = Freelist.lower_bound(Alloc);
  if (It != Freelist.end() && It->Size >= Size && It->Queue == Queue &&
      It->Alignment >= Alignment) {
    Allocation BestFit = *It;
    Freelist.erase(It);

    return BestFit;
  }

  // To make sure there's no match on other queues, we need to reset it to
  // nullptr and try again.
  Alloc.Queue = nullptr;
  It = Freelist.lower_bound(Alloc);

  if (It != Freelist.end() && It->Size >= Size && It->Alignment >= Alignment) {
    Allocation BestFit = *It;
    Freelist.erase(It);

    return BestFit;
  }

  return std::nullopt;
}

void EnqueuedPool::insert(void *Ptr, size_t Size, ur_event_handle_t Event,
                          void *Queue) {
  auto Lock = std::lock_guard(Mutex);

  uintptr_t Address = (uintptr_t)Ptr;
  size_t Alignment = Address & (~Address + 1);

  Freelist.emplace(Allocation{Ptr, Size, Event, Queue, Alignment});
}

bool EnqueuedPool::cleanup() {
  auto Lock = std::lock_guard(Mutex);
  auto FreedAllocations = !Freelist.empty();

  auto umfRet [[maybe_unused]] = UMF_RESULT_SUCCESS;
  for (auto It : Freelist) {
    umf_memory_pool_handle_t hPool = nullptr;
    umfRet = umfPoolByPtr(It.Ptr, &hPool);
    assert(hPool != nullptr);

    umfRet = umfPoolFree(hPool, It.Ptr);
    assert(umfRet == UMF_RESULT_SUCCESS);

    if (It.Event)
      eventRelease(It.Event);
  }
  Freelist.clear();

  return FreedAllocations;
}

bool EnqueuedPool::cleanupForQueue(void *Queue) {
  auto Lock = std::lock_guard(Mutex);

  Allocation Alloc = {nullptr, 0, nullptr, Queue, 0};
  // first allocation on the freelist with the specific queue
  auto It = Freelist.lower_bound(Alloc);

  bool FreedAllocations = false;

  auto umfRet [[maybe_unused]] = UMF_RESULT_SUCCESS;
  while (It != Freelist.end() && It->Queue == Queue) {
    umf_memory_pool_handle_t hPool = nullptr;
    umfRet = umfPoolByPtr(It->Ptr, &hPool);
    assert(hPool != nullptr);

    umfRet = umfPoolFree(hPool, It->Ptr);
    assert(umfRet == UMF_RESULT_SUCCESS);

    if (It->Event)
      eventRelease(It->Event);

    // Erase the current allocation and move to the next one
    It = Freelist.erase(It);
    FreedAllocations = true;
  }

  return FreedAllocations;
}
