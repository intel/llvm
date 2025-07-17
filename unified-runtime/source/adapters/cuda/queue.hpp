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

using cuda_stream_queue = stream_queue_t<CUstream, 128, 64, CUevent>;
struct ur_queue_handle_t_ : ur::cuda::handle_base, public cuda_stream_queue {};

using InteropGuard = cuda_stream_queue::interop_guard;

// Function which creates the profiling stream. Called only from makeNative
// event when profiling is required.
template <> inline void cuda_stream_queue::createHostSubmitTimeStream() {
  std::call_once(HostSubmitTimeStreamFlag, [&]() {
    UR_CHECK_ERROR(cuStreamCreateWithPriority(&HostSubmitTimeStream,
                                              CU_STREAM_NON_BLOCKING, 0));
  });
}

template <>
inline void cuda_stream_queue::createStreamWithPriority(CUstream *Stream,
                                                        unsigned int Flags,
                                                        int Priority) {
  UR_CHECK_ERROR(cuStreamCreateWithPriority(Stream, Flags, Priority));
}
