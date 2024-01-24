//===----------- enqueue.cpp - NATIVE CPU Adapter -------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <array>
#include <cstdint>

#include "ur_api.h"

#include "common.hpp"
#include "kernel.hpp"
#include "memory.hpp"

namespace native_cpu {
struct NDRDescT {
  using RangeT = std::array<size_t, 3>;
  uint32_t WorkDim;
  RangeT GlobalOffset;
  RangeT GlobalSize;
  RangeT LocalSize;
  NDRDescT(uint32_t WorkDim, const size_t *GlobalWorkOffset,
           const size_t *GlobalWorkSize, const size_t *LocalWorkSize)
      : WorkDim(WorkDim) {
    for (uint32_t I = 0; I < WorkDim; I++) {
      GlobalOffset[I] = GlobalWorkOffset[I];
      GlobalSize[I] = GlobalWorkSize[I];
      LocalSize[I] = LocalWorkSize[I];
    }
    for (uint32_t I = WorkDim; I < 3; I++) {
      GlobalSize[I] = 1;
      LocalSize[I] = LocalSize[0] ? 1 : 0;
      GlobalOffset[I] = 0;
    }
  }
};
} // namespace native_cpu

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueKernelLaunch(
    ur_queue_handle_t hQueue, ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;

  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pGlobalWorkOffset, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(workDim > 0, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);
  UR_ASSERT(workDim < 4, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);

  if (*pGlobalWorkSize == 0) {
    DIE_NO_IMPLEMENTATION;
  }

  // TODO: add proper error checking
  // TODO: add proper event dep management
  native_cpu::NDRDescT ndr(workDim, pGlobalWorkOffset, pGlobalWorkSize,
                           pLocalWorkSize);
  hKernel->handleLocalArgs();

  native_cpu::state state(ndr.GlobalSize[0], ndr.GlobalSize[1],
                          ndr.GlobalSize[2], ndr.LocalSize[0], ndr.LocalSize[1],
                          ndr.LocalSize[2], ndr.GlobalOffset[0],
                          ndr.GlobalOffset[1], ndr.GlobalOffset[2]);

  auto numWG0 = ndr.GlobalSize[0] / ndr.LocalSize[0];
  auto numWG1 = ndr.GlobalSize[1] / ndr.LocalSize[1];
  auto numWG2 = ndr.GlobalSize[2] / ndr.LocalSize[2];
  for (unsigned g2 = 0; g2 < numWG2; g2++) {
    for (unsigned g1 = 0; g1 < numWG1; g1++) {
      for (unsigned g0 = 0; g0 < numWG0; g0++) {
#ifdef NATIVECPU_USE_OCK
        state.update(g0, g1, g2);
        hKernel->_subhandler(hKernel->_args.data(), &state);
#else
        for (unsigned local2 = 0; local2 < ndr.LocalSize[2]; local2++) {
          for (unsigned local1 = 0; local1 < ndr.LocalSize[1]; local1++) {
            for (unsigned local0 = 0; local0 < ndr.LocalSize[0]; local0++) {
              state.update(g0, g1, g2, local0, local1, local2);
              hKernel->_subhandler(hKernel->_args.data(), &state);
            }
          }
        }
#endif
      }
    }
  }
  // TODO: we should avoid calling clear here by avoiding using push_back
  // in setKernelArgs.
  hKernel->_args.clear();
  hKernel->_localArgInfo.clear();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueEventsWait(
    ur_queue_handle_t hQueue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueEventsWaitWithBarrier(
    ur_queue_handle_t hQueue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;

  DIE_NO_IMPLEMENTATION;
}

template <bool IsRead>
static inline ur_result_t enqueueMemBufferReadWriteRect_impl(
    ur_queue_handle_t, ur_mem_handle_t Buff, bool,
    ur_rect_offset_t BufferOffset, ur_rect_offset_t HostOffset,
    ur_rect_region_t region, size_t BufferRowPitch, size_t BufferSlicePitch,
    size_t HostRowPitch, size_t HostSlicePitch,
    typename std::conditional<IsRead, void *, const void *>::type DstMem,
    uint32_t, const ur_event_handle_t *, ur_event_handle_t *) {
  // TODO: events, blocking, check other constraints, performance optimizations
  //       More sharing with level_zero where possible

  if (BufferRowPitch == 0)
    BufferRowPitch = region.width;
  if (BufferSlicePitch == 0)
    BufferSlicePitch = BufferRowPitch * region.height;
  if (HostRowPitch == 0)
    HostRowPitch = region.width;
  if (HostSlicePitch == 0)
    HostSlicePitch = HostRowPitch * region.height;
  for (size_t w = 0; w < region.width; w++)
    for (size_t h = 0; h < region.height; h++)
      for (size_t d = 0; d < region.depth; d++) {
        size_t buff_orign = (d + BufferOffset.z) * BufferSlicePitch +
                            (h + BufferOffset.y) * BufferRowPitch + w +
                            BufferOffset.x;
        size_t host_origin = (d + HostOffset.z) * HostSlicePitch +
                             (h + HostOffset.y) * HostRowPitch + w +
                             HostOffset.x;
        int8_t &buff_mem = ur_cast<int8_t *>(Buff->_mem)[buff_orign];
        if constexpr (IsRead)
          ur_cast<int8_t *>(DstMem)[host_origin] = buff_mem;
        else
          buff_mem = ur_cast<const int8_t *>(DstMem)[host_origin];
      }
  return UR_RESULT_SUCCESS;
}

static inline ur_result_t doCopy_impl(ur_queue_handle_t hQueue, void *DstPtr,
                                      const void *SrcPtr, size_t Size,
                                      uint32_t numEventsInWaitList,
                                      const ur_event_handle_t *EventWaitList,
                                      ur_event_handle_t *Event) {
  // todo: non-blocking, events, UR integration
  std::ignore = EventWaitList;
  std::ignore = Event;
  std::ignore = hQueue;
  std::ignore = numEventsInWaitList;
  if (SrcPtr != DstPtr && Size)
    memmove(DstPtr, SrcPtr, Size);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferRead(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingRead,
    size_t offset, size_t size, void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = blockingRead;

  void *FromPtr = /*Src*/ hBuffer->_mem + offset;
  return doCopy_impl(hQueue, pDst, FromPtr, size, numEventsInWaitList,
                     phEventWaitList, phEvent);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferWrite(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingWrite,
    size_t offset, size_t size, const void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = blockingWrite;

  void *ToPtr = hBuffer->_mem + offset;
  return doCopy_impl(hQueue, ToPtr, pSrc, size, numEventsInWaitList,
                     phEventWaitList, phEvent);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferReadRect(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingRead,
    ur_rect_offset_t bufferOrigin, ur_rect_offset_t hostOrigin,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pDst,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  return enqueueMemBufferReadWriteRect_impl<true /*read*/>(
      hQueue, hBuffer, blockingRead, bufferOrigin, hostOrigin, region,
      bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch, pDst,
      numEventsInWaitList, phEventWaitList, phEvent);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferWriteRect(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingWrite,
    ur_rect_offset_t bufferOrigin, ur_rect_offset_t hostOrigin,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pSrc,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  return enqueueMemBufferReadWriteRect_impl<false /*write*/>(
      hQueue, hBuffer, blockingWrite, bufferOrigin, hostOrigin, region,
      bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch, pSrc,
      numEventsInWaitList, phEventWaitList, phEvent);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferCopy(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBufferSrc,
    ur_mem_handle_t hBufferDst, size_t srcOffset, size_t dstOffset, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  const void *SrcPtr = hBufferSrc->_mem + srcOffset;
  void *DstPtr = hBufferDst->_mem + dstOffset;
  return doCopy_impl(hQueue, DstPtr, SrcPtr, size, numEventsInWaitList,
                     phEventWaitList, phEvent);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferCopyRect(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBufferSrc,
    ur_mem_handle_t hBufferDst, ur_rect_offset_t srcOrigin,
    ur_rect_offset_t dstOrigin, ur_rect_region_t region, size_t srcRowPitch,
    size_t srcSlicePitch, size_t dstRowPitch, size_t dstSlicePitch,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  return enqueueMemBufferReadWriteRect_impl<true /*read*/>(
      hQueue, hBufferSrc, false /*todo: check blocking*/, srcOrigin,
      /*HostOffset*/ dstOrigin, region, srcRowPitch, srcSlicePitch, dstRowPitch,
      dstSlicePitch, hBufferDst->_mem, numEventsInWaitList, phEventWaitList,
      phEvent);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferFill(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, const void *pPattern,
    size_t patternSize, size_t offset, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;

  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  // TODO: error checking
  // TODO: handle async
  void *startingPtr = hBuffer->_mem + offset;
  unsigned steps = size / patternSize;
  for (unsigned i = 0; i < steps; i++) {
    memcpy(static_cast<int8_t *>(startingPtr) + i * patternSize, pPattern,
           patternSize);
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemImageRead(
    ur_queue_handle_t hQueue, ur_mem_handle_t hImage, bool blockingRead,
    ur_rect_offset_t origin, ur_rect_region_t region, size_t rowPitch,
    size_t slicePitch, void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = hImage;
  std::ignore = blockingRead;
  std::ignore = origin;
  std::ignore = region;
  std::ignore = rowPitch;
  std::ignore = slicePitch;
  std::ignore = pDst;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemImageWrite(
    ur_queue_handle_t hQueue, ur_mem_handle_t hImage, bool blockingWrite,
    ur_rect_offset_t origin, ur_rect_region_t region, size_t rowPitch,
    size_t slicePitch, void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = hImage;
  std::ignore = blockingWrite;
  std::ignore = origin;
  std::ignore = region;
  std::ignore = rowPitch;
  std::ignore = slicePitch;
  std::ignore = pSrc;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemImageCopy(
    ur_queue_handle_t hQueue, ur_mem_handle_t hImageSrc,
    ur_mem_handle_t hImageDst, ur_rect_offset_t srcOrigin,
    ur_rect_offset_t dstOrigin, ur_rect_region_t region,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = hImageSrc;
  std::ignore = hImageDst;
  std::ignore = srcOrigin;
  std::ignore = dstOrigin;
  std::ignore = region;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferMap(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingMap,
    ur_map_flags_t mapFlags, size_t offset, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent, void **ppRetMap) {
  std::ignore = hQueue;
  std::ignore = blockingMap;
  std::ignore = mapFlags;
  std::ignore = size;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;

  *ppRetMap = hBuffer->_mem + offset;

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemUnmap(
    ur_queue_handle_t hQueue, ur_mem_handle_t hMem, void *pMappedPtr,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = hMem;
  std::ignore = pMappedPtr;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMFill(
    ur_queue_handle_t hQueue, void *ptr, size_t patternSize,
    const void *pPattern, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;

  UR_ASSERT(ptr, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(pPattern, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(size % patternSize == 0 || patternSize > size,
            UR_RESULT_ERROR_INVALID_SIZE);

  memset(ptr, *static_cast<const uint8_t *>(pPattern), size * patternSize);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMMemcpy(
    ur_queue_handle_t hQueue, bool blocking, void *pDst, const void *pSrc,
    size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = blocking;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;

  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_QUEUE);
  UR_ASSERT(pDst, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(pSrc, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  memcpy(pDst, pSrc, size);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMPrefetch(
    ur_queue_handle_t hQueue, const void *pMem, size_t size,
    ur_usm_migration_flags_t flags, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = pMem;
  std::ignore = size;
  std::ignore = flags;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueUSMAdvise(ur_queue_handle_t hQueue, const void *pMem, size_t size,
                   ur_usm_advice_flags_t advice, ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = pMem;
  std::ignore = size;
  std::ignore = advice;
  std::ignore = phEvent;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMFill2D(
    ur_queue_handle_t hQueue, void *pMem, size_t pitch, size_t patternSize,
    const void *pPattern, size_t width, size_t height,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = pMem;
  std::ignore = pitch;
  std::ignore = patternSize;
  std::ignore = pPattern;
  std::ignore = width;
  std::ignore = height;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMMemcpy2D(
    ur_queue_handle_t hQueue, bool blocking, void *pDst, size_t dstPitch,
    const void *pSrc, size_t srcPitch, size_t width, size_t height,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = blocking;
  std::ignore = pDst;
  std::ignore = dstPitch;
  std::ignore = pSrc;
  std::ignore = srcPitch;
  std::ignore = width;
  std::ignore = height;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueDeviceGlobalVariableWrite(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram, const char *name,
    bool blockingWrite, size_t count, size_t offset, const void *pSrc,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = hProgram;
  std::ignore = name;
  std::ignore = blockingWrite;
  std::ignore = count;
  std::ignore = offset;
  std::ignore = pSrc;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueDeviceGlobalVariableRead(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram, const char *name,
    bool blockingRead, size_t count, size_t offset, void *pDst,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = hProgram;
  std::ignore = name;
  std::ignore = blockingRead;
  std::ignore = count;
  std::ignore = offset;
  std::ignore = pDst;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueReadHostPipe(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram,
    const char *pipe_symbol, bool blocking, void *pDst, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = hProgram;
  std::ignore = pipe_symbol;
  std::ignore = blocking;
  std::ignore = pDst;
  std::ignore = size;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueWriteHostPipe(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram,
    const char *pipe_symbol, bool blocking, void *pSrc, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = hProgram;
  std::ignore = pipe_symbol;
  std::ignore = blocking;
  std::ignore = pSrc;
  std::ignore = size;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;

  DIE_NO_IMPLEMENTATION;
}
