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
#include "usm.hpp"

#include <ur_api.h>

namespace {

std::optional<EnqueuedPool::Allocation>
getBestFitHelper(size_t Size, size_t Alignment, void *Queue,
                 EnqueuedPool::AllocationGroupMap &Freelist) {
  // Iterate over the alignments for a given queue.
  auto GroupIt = Freelist.lower_bound({Queue, Alignment});
  for (; GroupIt != Freelist.end() && GroupIt->first.Queue == Queue;
       ++GroupIt) {
    auto &AllocSet = GroupIt->second;
    // Find the first allocation that is large enough.
    auto AllocIt = AllocSet.lower_bound({nullptr, Size, nullptr, nullptr, 0});
    if (AllocIt != AllocSet.end()) {
      auto BestFit = *AllocIt;
      AllocSet.erase(AllocIt);
      if (AllocSet.empty()) {
        Freelist.erase(GroupIt);
      }
      return BestFit;
    }
  }
  return std::nullopt;
}

void removeFromFreelist(const EnqueuedPool::Allocation &Alloc,
                        EnqueuedPool::AllocationGroupMap &Freelist,
                        bool IsGlobal) {
  const EnqueuedPool::AllocationGroupKey Key = {
      IsGlobal ? nullptr : Alloc.Queue, Alloc.Alignment};

  auto GroupIt = Freelist.find(Key);
  assert(GroupIt != Freelist.end() && "Allocation group not found in freelist");

  auto &AllocSet = GroupIt->second;
  auto AllocIt = AllocSet.find(Alloc);
  assert(AllocIt != AllocSet.end() && "Allocation not found in group");

  AllocSet.erase(AllocIt);
  if (AllocSet.empty()) {
    Freelist.erase(GroupIt);
  }
}

} // namespace

EnqueuedPool::~EnqueuedPool() { cleanup(); }

std::optional<EnqueuedPool::Allocation>
EnqueuedPool::getBestFit(size_t Size, size_t Alignment, void *Queue) {
  auto Lock = std::lock_guard(Mutex);

  // First, try to find the best fit in the queue-specific freelist.
  auto BestFit = getBestFitHelper(Size, Alignment, Queue, FreelistByQueue);
  if (BestFit) {
    // Remove the allocation from the global freelist as well.
    removeFromFreelist(*BestFit, FreelistGlobal, true);
    return BestFit;
  }

  // If no fit was found in the queue-specific freelist, try the global
  // freelist.
  BestFit = getBestFitHelper(Size, Alignment, nullptr, FreelistGlobal);
  if (BestFit) {
    // Remove the allocation from the queue-specific freelist.
    removeFromFreelist(*BestFit, FreelistByQueue, false);
    return BestFit;
  }

  return std::nullopt;
}

void EnqueuedPool::insert(void *Ptr, size_t Size, ur_event_handle_t Event,
                          void *Queue) {
  auto Lock = std::lock_guard(Mutex);

  uintptr_t Address = (uintptr_t)Ptr;
  size_t Alignment = Address & (~Address + 1);

  Allocation Alloc = {Ptr, Size, Event, Queue, Alignment};
  FreelistByQueue[{Queue, Alignment}].emplace(Alloc);
  FreelistGlobal[{nullptr, Alignment}].emplace(Alloc);
}

bool EnqueuedPool::cleanup() {
  auto Lock = std::lock_guard(Mutex);
  auto FreedAllocations = !FreelistGlobal.empty();

  auto Ret [[maybe_unused]] = UR_RESULT_SUCCESS;
  for (const auto &[GroupKey, AllocSet] : FreelistGlobal) {
    for (const auto &Alloc : AllocSet) {
      Ret = MemFreeFn(Alloc.Ptr);
      assert(Ret == UR_RESULT_SUCCESS);

      if (Alloc.Event) {
        EventReleaseFn(Alloc.Event);
      }
    }
  }

  FreelistGlobal.clear();
  FreelistByQueue.clear();

  return FreedAllocations;
}

bool EnqueuedPool::cleanupForQueue(void *Queue) {
  auto Lock = std::lock_guard(Mutex);
  bool FreedAllocations = false;

  auto Ret [[maybe_unused]] = UR_RESULT_SUCCESS;
  auto GroupIt = FreelistByQueue.lower_bound({Queue, 0});
  while (GroupIt != FreelistByQueue.end() && GroupIt->first.Queue == Queue) {
    auto &AllocSet = GroupIt->second;
    for (const auto &Alloc : AllocSet) {
      Ret = MemFreeFn(Alloc.Ptr);
      assert(Ret == UR_RESULT_SUCCESS);

      if (Alloc.Event) {
        EventReleaseFn(Alloc.Event);
      }

      removeFromFreelist(Alloc, FreelistGlobal, true);
    }

    // Move to the next group.
    GroupIt = FreelistByQueue.erase(GroupIt);
    FreedAllocations = true;
  }

  return FreedAllocations;
}
