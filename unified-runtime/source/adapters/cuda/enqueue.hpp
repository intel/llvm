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

#include "common.hpp"
#include <cassert>
#include <cuda.h>
#include <ur_api.h>

ur_result_t enqueueEventsWait(ur_queue_handle_t CommandQueue, CUstream Stream,
                              uint32_t NumEventsInWaitList,
                              const ur_event_handle_t *EventWaitList);

template <typename PtrT>
void getUSMHostOrDevicePtr(PtrT USMPtr, CUmemorytype *OutMemType,
                           CUdeviceptr *OutDevPtr, PtrT *OutHostPtr) {
  // do not throw if cuPointerGetAttribute returns CUDA_ERROR_INVALID_VALUE
  // checks with PI_CHECK_ERROR are not suggested
  CUresult Ret = cuPointerGetAttribute(
      OutMemType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)USMPtr);
  // ARRAY, UNIFIED types are not supported!
  assert(*OutMemType != CU_MEMORYTYPE_ARRAY &&
         *OutMemType != CU_MEMORYTYPE_UNIFIED);

  // pointer not known to the CUDA subsystem (possibly a system allocated ptr)
  if (Ret == CUDA_ERROR_INVALID_VALUE) {
    *OutMemType = CU_MEMORYTYPE_HOST;
    *OutDevPtr = 0;
    *OutHostPtr = USMPtr;

    // todo: resets the above "non-stick" error
  } else if (Ret == CUDA_SUCCESS) {
    *OutDevPtr = (*OutMemType == CU_MEMORYTYPE_DEVICE)
                     ? reinterpret_cast<CUdeviceptr>(USMPtr)
                     : 0;
    *OutHostPtr = (*OutMemType == CU_MEMORYTYPE_HOST) ? USMPtr : nullptr;
  } else {
    UR_CHECK_ERROR(Ret);
  }
}

void guessLocalWorkSize(ur_device_handle_t Device, size_t *ThreadsPerBlock,
                        const size_t *GlobalWorkSize, const uint32_t WorkDim,
                        ur_kernel_handle_t Kernel);

bool hasExceededMaxRegistersPerBlock(ur_device_handle_t Device,
                                     ur_kernel_handle_t Kernel,
                                     size_t BlockSize);

ur_result_t
setKernelParams(const ur_device_handle_t Device, const uint32_t WorkDim,
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
