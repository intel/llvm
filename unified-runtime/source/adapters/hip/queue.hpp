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

using hip_stream_queue = stream_queue_t<hipStream_t, 64, 16, hipEvent_t>;
struct ur_queue_handle_t_ : public hip_stream_queue {};

template <>
inline void hip_stream_queue::createStreamWithPriority(hipStream_t *Stream,
                                                       unsigned int Flags,
                                                       int Priority) {
  UR_CHECK_ERROR(hipStreamCreateWithPriority(Stream, Flags, Priority));
}

// Function which creates the profiling stream. Called only from makeNative
// event when profiling is required.
template <> inline void hip_stream_queue::createHostSubmitTimeStream() {
  static std::once_flag HostSubmitTimeStreamFlag;
  std::call_once(HostSubmitTimeStreamFlag, [&]() {
    UR_CHECK_ERROR(
        hipStreamCreateWithFlags(&HostSubmitTimeStream, hipStreamNonBlocking));
  });
}

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
