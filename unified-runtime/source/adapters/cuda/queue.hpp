//===--------- queue.hpp - CUDA Adapter -----------------------------------===//
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
#include <ur/ur.hpp>

#include <algorithm>
#include <cuda.h>
#include <mutex>
#include <vector>

#include <common/cuda-hip/stream_queue.hpp>

/// UR queue mapping on to CUstream objects.
///
struct ur_queue_handle_t_ : stream_queue_t<CUstream, 128, 64> {
  using stream_queue_t<CUstream, DefaultNumComputeStreams,
                       DefaultNumTransferStreams>::stream_queue_t;

  CUevent BarrierEvent = nullptr;
  CUevent BarrierTmpEvent = nullptr;

  void computeStreamWaitForBarrierIfNeeded(CUstream Strean,
                                           uint32_t StreamI) override;
  void transferStreamWaitForBarrierIfNeeded(CUstream Stream,
                                            uint32_t StreamI) override;
  ur_queue_handle_t getEventQueue(const ur_event_handle_t) override;
  uint32_t getEventComputeStreamToken(const ur_event_handle_t) override;
  CUstream getEventStream(const ur_event_handle_t) override;

  // Function which creates the profiling stream. Called only from makeNative
  // event when profiling is required.
  void createHostSubmitTimeStream() {
    static std::once_flag HostSubmitTimeStreamFlag;
    std::call_once(HostSubmitTimeStreamFlag, [&]() {
      UR_CHECK_ERROR(cuStreamCreateWithPriority(&HostSubmitTimeStream,
                                                CU_STREAM_NON_BLOCKING, 0));
    });
  }

  void createStreamWithPriority(CUstream *Stream, unsigned int Flags,
                                int Priority) override {
    UR_CHECK_ERROR(cuStreamCreateWithPriority(Stream, Flags, Priority));
  }
};

// RAII object to make hQueue stream getter methods all return the same stream
// within the lifetime of this object.
//
// This is useful for urEnqueueNativeCommandExp where we want guarantees that
// the user submitted native calls will be dispatched to a known stream, which
// must be "got" within the user submitted fuction.
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
  CUstream getStream() { return hQueue->getThreadLocalStream(); }
  ~ScopedStream() { hQueue->getThreadLocalStream() = CUstream{0}; }
};
