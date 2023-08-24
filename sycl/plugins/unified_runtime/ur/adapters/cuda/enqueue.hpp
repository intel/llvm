//===--------- enqueue.hpp - CUDA Adapter ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include <cassert>
#include <cuda.h>
#include <ur_api.h>

ur_result_t enqueueEventsWait(ur_queue_handle_t CommandQueue, CUstream Stream,
                              uint32_t NumEventsInWaitList,
                              const ur_event_handle_t *EventWaitList);

void guessLocalWorkSize(ur_device_handle_t Device, size_t *ThreadsPerBlock,
                        const size_t *GlobalWorkSize, const uint32_t WorkDim,
                        const size_t MaxThreadsPerBlock[3],
                        ur_kernel_handle_t Kernel, uint32_t LocalSize);

bool hasExceededMaxRegistersPerBlock(ur_device_handle_t Device,
                                     ur_kernel_handle_t Kernel,
                                     size_t BlockSize);

ur_result_t
setKernelParams(const ur_context_handle_t Context,
                const ur_device_handle_t Device, const uint32_t WorkDim,
                const size_t *GlobalWorkOffset, const size_t *GlobalWorkSize,
                const size_t *LocalWorkSize, ur_kernel_handle_t &Kernel,
                CUfunction &CuFunc, size_t (&ThreadsPerBlock)[3],
                size_t (&BlocksPerGrid)[3]);
