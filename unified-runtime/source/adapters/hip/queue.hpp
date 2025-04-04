//===--------- queue.hpp - HIP Adapter ------------------------------------===//
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
#include <hip/hip_runtime.h>
#include <mutex>
#include <vector>

#include <common/cuda-hip/stream_queue.hpp>

/// UR queue mapping on to hipStream_t objects.
///
struct ur_queue_handle_t_ : stream_queue_t<hipStream_t, 64, 16> {
  using stream_queue_t<hipStream_t, DefaultNumComputeStreams,
                       DefaultNumTransferStreams>::stream_queue_t;

  hipEvent_t BarrierEvent = nullptr;
  hipEvent_t BarrierTmpEvent = nullptr;

  void computeStreamWaitForBarrierIfNeeded(hipStream_t Strean,
                                           uint32_t StreamI) override;
  void transferStreamWaitForBarrierIfNeeded(hipStream_t Stream,
                                            uint32_t StreamI) override;
  ur_queue_handle_t getEventQueue(const ur_event_handle_t) override;
  uint32_t getEventComputeStreamToken(const ur_event_handle_t) override;
  hipStream_t getEventStream(const ur_event_handle_t) override;

  // Function which creates the profiling stream. Called only from makeNative
  // event when profiling is required.
  void createHostSubmitTimeStream() {
    static std::once_flag HostSubmitTimeStreamFlag;
    std::call_once(HostSubmitTimeStreamFlag, [&]() {
      UR_CHECK_ERROR(hipStreamCreateWithFlags(&HostSubmitTimeStream,
                                              hipStreamNonBlocking));
    });
  }

  void createStreamWithPriority(hipStream_t *Stream, unsigned int Flags,
                                int Priority) override {
    UR_CHECK_ERROR(hipStreamCreateWithPriority(Stream, Flags, Priority));
  }
};

// RAII object to make hQueue stream getter methods all return the same stream
// within the lifetime of this object.
//
// This is useful for urEnqueueNativeCommandExp where we want guarantees that
// the user submitted native calls will be dispatched to a known stream, which
// must be "got" within the user submitted function.
//
// TODO: Add a test that this scoping works
class ScopedStream {
  ur_queue_handle_t hQueue;

public:
  ScopedStream(ur_queue_handle_t hQueue, uint32_t NumEventsInWaitList,
               const ur_event_handle_t *EventWaitList)
      : hQueue{hQueue} {
    ur_stream_guard Guard;
    hQueue->getThreadLocalStream() =
        hQueue->getNextComputeStream(NumEventsInWaitList, EventWaitList, Guard);
  }
  hipStream_t getStream() { return hQueue->getThreadLocalStream(); }
  ~ScopedStream() { hQueue->getThreadLocalStream() = hipStream_t{0}; }
};
