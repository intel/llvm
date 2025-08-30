//===----------- queue.hpp - LLVM Offload Adapter  ------------------------===//
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
#include "event.hpp"

constexpr size_t OOO_QUEUE_POOL_SIZE = 32;

struct ur_queue_handle_t_ : RefCounted {
  ur_queue_handle_t_(ol_device_handle_t Device, ur_context_handle_t UrContext,
                     ur_queue_flags_t Flags)
      : OffloadQueues((Flags & UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE)
                          ? 1
                          : OOO_QUEUE_POOL_SIZE),
        QueueOffset(0), Barrier(nullptr), OffloadDevice(Device),
        UrContext(UrContext), Flags(Flags) {}

  // In-order queues only have one element here, while out of order queues have
  // a bank of queues to use. We rotate through them round robin instead of
  // constantly creating new ones in case there is a long-running program that
  // never destroys the ur queue. Out-of-order queues create ol queues when
  // needed; any queues that are not yet created are nullptr.
  // This is a simpler implementation of the HIP/Cuda queue pooling logic in
  // `stream_queue_t`. In the future, if we want more performance or it
  // simplifies the implementation of a feature, we can consider using it.
  std::vector<ol_queue_handle_t> OffloadQueues;
  // Mutex guarding the offset and barrier for out of order queues
  std::mutex OooMutex;
  size_t QueueOffset;
  ur_event_handle_t Barrier;
  ol_device_handle_t OffloadDevice;
  ur_context_handle_t UrContext;
  ur_queue_flags_t Flags;

  bool isInOrder() const { return OffloadQueues.size() == 1; }

  // This queue is empty if and only if all queues are empty
  ol_result_t isEmpty(bool &Empty) const {
    Empty = true;

    for (auto *Q : OffloadQueues) {
      if (!Q) {
        continue;
      }
      if (auto Err =
              olGetQueueInfo(Q, OL_QUEUE_INFO_EMPTY, sizeof(Empty), &Empty)) {
        return Err;
      }
      if (!Empty) {
        return OL_SUCCESS;
      }
    }

    return OL_SUCCESS;
  }

  ol_result_t nextQueueNoLock(ol_queue_handle_t &Handle) {
    auto &Slot = OffloadQueues[(QueueOffset++) % OffloadQueues.size()];

    if (!Slot) {
      if (auto Res = olCreateQueue(OffloadDevice, &Slot)) {
        return Res;
      }

      if (auto Event = Barrier) {
        if (auto Res = olWaitEvents(Slot, &Event->OffloadEvent, 1)) {
          return Res;
        }
      }
    }

    Handle = Slot;
    return nullptr;
  }

  ol_result_t nextQueue(ol_queue_handle_t &Handle) {
    std::lock_guard<std::mutex> Lock(OooMutex);
    return nextQueueNoLock(Handle);
  }
};
