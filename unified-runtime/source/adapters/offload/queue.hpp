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

constexpr size_t OOO_QUEUE_POOL_SIZE = 32;

struct ur_queue_handle_t_ : RefCounted {
  ur_queue_handle_t_(ol_device_handle_t Device, ur_context_handle_t UrContext,
                     ur_queue_flags_t Flags)
      : OffloadQueues((Flags & UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE)
                          ? 1
                          : OOO_QUEUE_POOL_SIZE),
        QueueOffset(0), OffloadDevice(Device), UrContext(UrContext),
        Flags(Flags) {}

  // In-order queues only have one element here, while out of order queues have
  // a bank of queues to use. We rotate through them round robin instead of
  // constantly creating new ones in case there is a long-running program that
  // never destroys the ur queue. Out-of-order queues create ol queues when
  // needed; any queues that are not yet created are nullptr.
  // This is a simpler implementation of the HIP/Cuda queue pooling logic in
  // `stream_queue_t`. In the future, if we want more performance or it
  // simplifies the implementation of a feature, we can consider using it.
  std::vector<ol_queue_handle_t> OffloadQueues;
  size_t QueueOffset;
  ol_device_handle_t OffloadDevice;
  ur_context_handle_t UrContext;
  ur_queue_flags_t Flags;

  ol_result_t nextQueue(ol_queue_handle_t &Handle) {
    auto &Slot = OffloadQueues[QueueOffset++];
    QueueOffset %= OffloadQueues.size();

    if (!Slot) {
      if (auto Res = olCreateQueue(OffloadDevice, &Slot)) {
        return Res;
      }
    }

    Handle = Slot;
    return nullptr;
  }
};
