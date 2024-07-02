//===--------- enqueue.hpp - CUDA Adapter ---------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cassert>
#include <cuda.h>
#include <ur_api.h>

ur_result_t enqueueEventsWait(ur_queue_handle_t CommandQueue, CUstream Stream,
                              uint32_t NumEventsInWaitList,
                              const ur_event_handle_t *EventWaitList);

void guessLocalWorkSize(ur_device_handle_t Device, size_t *ThreadsPerBlock,
                        const size_t *GlobalWorkSize, const uint32_t WorkDim,
                        ur_kernel_handle_t Kernel);

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

void setCopyRectParams(ur_rect_region_t region, const void *SrcPtr,
                       const CUmemorytype_enum SrcType,
                       ur_rect_offset_t src_offset, size_t src_row_pitch,
                       size_t src_slice_pitch, void *DstPtr,
                       const CUmemorytype_enum DstType,
                       ur_rect_offset_t dst_offset, size_t dst_row_pitch,
                       size_t dst_slice_pitch, CUDA_MEMCPY3D &params);
