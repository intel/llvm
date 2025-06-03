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
struct ur_queue_handle_t_ : ur::hip::handle_base, public hip_stream_queue {};

using InteropGuard = hip_stream_queue::interop_guard;

template <>
inline void hip_stream_queue::createStreamWithPriority(hipStream_t *Stream,
                                                       unsigned int Flags,
                                                       int Priority) {
  UR_CHECK_ERROR(hipStreamCreateWithPriority(Stream, Flags, Priority));
}

// Function which creates the profiling stream. Called only from makeNative
// event when profiling is required.
template <> inline void hip_stream_queue::createHostSubmitTimeStream() {
  std::call_once(HostSubmitTimeStreamFlag, [&]() {
    UR_CHECK_ERROR(
        hipStreamCreateWithFlags(&HostSubmitTimeStream, hipStreamNonBlocking));
  });
}
