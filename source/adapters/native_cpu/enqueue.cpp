//===----------- enqueue.cpp - NATIVE CPU Adapter -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "ur_api.h"

#include "common.hpp"
#include "event.hpp"
#include "kernel.hpp"
#include "memory.hpp"
#include "queue.hpp"
#include "threadpool.hpp"

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
      LocalSize[I] = LocalWorkSize ? LocalWorkSize[I] : 1;
    }
    for (uint32_t I = WorkDim; I < 3; I++) {
      GlobalSize[I] = 1;
      LocalSize[I] = LocalSize[0] ? 1 : 0;
      GlobalOffset[I] = 0;
    }
  }

  void dump(std::ostream &os) const {
    os << "GlobalSize: " << GlobalSize[0] << " " << GlobalSize[1] << " "
       << GlobalSize[2] << "\n";
    os << "LocalSize: " << LocalSize[0] << " " << LocalSize[1] << " "
       << LocalSize[2] << "\n";
    os << "GlobalOffset: " << GlobalOffset[0] << " " << GlobalOffset[1] << " "
       << GlobalOffset[2] << "\n";
  }
};
} // namespace native_cpu

#ifdef NATIVECPU_USE_OCK
static native_cpu::state getResizedState(const native_cpu::NDRDescT &ndr,
                                         size_t itemsPerThread) {
  native_cpu::state resized_state(
      ndr.GlobalSize[0], ndr.GlobalSize[1], ndr.GlobalSize[2], itemsPerThread,
      ndr.LocalSize[1], ndr.LocalSize[2], ndr.GlobalOffset[0],
      ndr.GlobalOffset[1], ndr.GlobalOffset[2]);
  return resized_state;
}
#endif

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueKernelLaunch(
    ur_queue_handle_t hQueue, ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {

  urEventWait(numEventsInWaitList, phEventWaitList);
  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pGlobalWorkOffset, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(workDim > 0, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);
  UR_ASSERT(workDim < 4, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);

  if (*pGlobalWorkSize == 0) {
    DIE_NO_IMPLEMENTATION;
  }

  // Check reqd_work_group_size and other kernel constraints
  if (pLocalWorkSize != nullptr) {
    uint64_t TotalNumWIs = 1;
    for (uint32_t Dim = 0; Dim < workDim; Dim++) {
      TotalNumWIs *= pLocalWorkSize[Dim];
      if (auto Reqd = hKernel->getReqdWGSize();
          Reqd && pLocalWorkSize[Dim] != Reqd.value()[Dim]) {
        return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
      }
      if (auto MaxWG = hKernel->getMaxWGSize();
          MaxWG && pLocalWorkSize[Dim] > MaxWG.value()[Dim]) {
        return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
      }
    }
    if (auto MaxLinearWG = hKernel->getMaxLinearWGSize()) {
      if (TotalNumWIs > MaxLinearWG) {
        return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
      }
    }
  }

  // TODO: add proper error checking
  native_cpu::NDRDescT ndr(workDim, pGlobalWorkOffset, pGlobalWorkSize,
                           pLocalWorkSize);
  auto &tp = hQueue->getDevice()->tp;
  const size_t numParallelThreads = tp.num_threads();
  hKernel->updateMemPool(numParallelThreads);
  std::vector<std::future<void>> futures;
  std::vector<std::function<void(size_t, ur_kernel_handle_t_)>> groups;
  auto numWG0 = ndr.GlobalSize[0] / ndr.LocalSize[0];
  auto numWG1 = ndr.GlobalSize[1] / ndr.LocalSize[1];
  auto numWG2 = ndr.GlobalSize[2] / ndr.LocalSize[2];
  native_cpu::state state(ndr.GlobalSize[0], ndr.GlobalSize[1],
                          ndr.GlobalSize[2], ndr.LocalSize[0], ndr.LocalSize[1],
                          ndr.LocalSize[2], ndr.GlobalOffset[0],
                          ndr.GlobalOffset[1], ndr.GlobalOffset[2]);
  auto event = new ur_event_handle_t_(hQueue, UR_COMMAND_KERNEL_LAUNCH);
  event->tick_start();

#ifndef NATIVECPU_USE_OCK
  hKernel->handleLocalArgs(1, 0);
  for (unsigned g2 = 0; g2 < numWG2; g2++) {
    for (unsigned g1 = 0; g1 < numWG1; g1++) {
      for (unsigned g0 = 0; g0 < numWG0; g0++) {
        for (unsigned local2 = 0; local2 < ndr.LocalSize[2]; local2++) {
          for (unsigned local1 = 0; local1 < ndr.LocalSize[1]; local1++) {
            for (unsigned local0 = 0; local0 < ndr.LocalSize[0]; local0++) {
              state.update(g0, g1, g2, local0, local1, local2);
              hKernel->_subhandler(hKernel->getArgs().data(), &state);
            }
          }
        }
      }
    }
  }
#else
  bool isLocalSizeOne =
      ndr.LocalSize[0] == 1 && ndr.LocalSize[1] == 1 && ndr.LocalSize[2] == 1;
  if (isLocalSizeOne && ndr.GlobalSize[0] > numParallelThreads) {
    // If the local size is one, we make the assumption that we are running a
    // parallel_for over a sycl::range.
    // Todo: we could add compiler checks and
    // kernel properties for this (e.g. check that no barriers are called, no
    // local memory args).

    // Todo: this assumes that dim 0 is the best dimension over which we want to
    // parallelize

    // Since we also vectorize the kernel, and vectorization happens within the
    // work group loop, it's better to have a large-ish local size. We can
    // divide the global range by the number of threads, set that as the local
    // size and peel everything else.

    size_t new_num_work_groups_0 = numParallelThreads;
    size_t itemsPerThread = ndr.GlobalSize[0] / numParallelThreads;

    for (unsigned g2 = 0; g2 < numWG2; g2++) {
      for (unsigned g1 = 0; g1 < numWG1; g1++) {
        for (unsigned g0 = 0; g0 < new_num_work_groups_0; g0 += 1) {
          futures.emplace_back(tp.schedule_task(
              [ndr, itemsPerThread, kernel = *hKernel, g0, g1, g2](size_t) {
                native_cpu::state resized_state =
                    getResizedState(ndr, itemsPerThread);
                resized_state.update(g0, g1, g2);
                kernel._subhandler(kernel.getArgs().data(), &resized_state);
              }));
        }
        // Peel the remaining work items. Since the local size is 1, we iterate
        // over the work groups.
        for (unsigned g0 = new_num_work_groups_0 * itemsPerThread; g0 < numWG0;
             g0++) {
          state.update(g0, g1, g2);
          hKernel->_subhandler(hKernel->getArgs().data(), &state);
        }
      }
    }

  } else {
    // We are running a parallel_for over an nd_range

    if (numWG1 * numWG2 >= numParallelThreads) {
      // Dimensions 1 and 2 have enough work, split them across the threadpool
      for (unsigned g2 = 0; g2 < numWG2; g2++) {
        for (unsigned g1 = 0; g1 < numWG1; g1++) {
          futures.emplace_back(
              tp.schedule_task([state, kernel = *hKernel, numWG0, g1, g2,
                                numParallelThreads](size_t threadId) mutable {
                for (unsigned g0 = 0; g0 < numWG0; g0++) {
                  kernel.handleLocalArgs(numParallelThreads, threadId);
                  state.update(g0, g1, g2);
                  kernel._subhandler(kernel.getArgs().data(), &state);
                }
              }));
        }
      }
    } else {
      // Split dimension 0 across the threadpool
      // Here we try to create groups of workgroups in order to reduce
      // synchronization overhead
      for (unsigned g2 = 0; g2 < numWG2; g2++) {
        for (unsigned g1 = 0; g1 < numWG1; g1++) {
          for (unsigned g0 = 0; g0 < numWG0; g0++) {
            groups.push_back(
                [state, g0, g1, g2, numParallelThreads](
                    size_t threadId, ur_kernel_handle_t_ kernel) mutable {
                  kernel.handleLocalArgs(numParallelThreads, threadId);
                  state.update(g0, g1, g2);
                  kernel._subhandler(kernel.getArgs().data(), &state);
                });
          }
        }
      }
      auto numGroups = groups.size();
      auto groupsPerThread = numGroups / numParallelThreads;
      auto remainder = numGroups % numParallelThreads;
      for (unsigned thread = 0; thread < numParallelThreads; thread++) {
        futures.emplace_back(
            tp.schedule_task([groups, thread, groupsPerThread,
                              kernel = *hKernel](size_t threadId) {
              for (unsigned i = 0; i < groupsPerThread; i++) {
                auto index = thread * groupsPerThread + i;
                groups[index](threadId, kernel);
              }
            }));
      }

      // schedule the remaining tasks
      if (remainder) {
        futures.emplace_back(
            tp.schedule_task([groups, remainder,
                              scheduled = numParallelThreads * groupsPerThread,
                              kernel = *hKernel](size_t threadId) {
              for (unsigned i = 0; i < remainder; i++) {
                auto index = scheduled + i;
                groups[index](threadId, kernel);
              }
            }));
      }
    }
  }

#endif // NATIVECPU_USE_OCK
  event->set_futures(futures);

  *phEvent = event;
  event->set_callback([hKernel, event]() {
    event->tick_end();
    // TODO: avoid calling clear() here.
    hKernel->_localArgInfo.clear();
  });

  if (hQueue->isInOrder()) {
    urEventWait(1, phEvent);
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t withTimingEvent(ur_command_t command_type, ur_queue_handle_t hQueue,
                            uint32_t numEventsInWaitList,
                            const ur_event_handle_t *phEventWaitList,
                            ur_event_handle_t *phEvent,
                            const std::function<ur_result_t()> &f) {
  urEventWait(numEventsInWaitList, phEventWaitList);
  ur_event_handle_t event;
  if (phEvent) {
    event = new ur_event_handle_t_(hQueue, command_type);
    event->tick_start();
  }

  ur_result_t result = f();

  if (phEvent) {
    event->tick_end();
    *phEvent = event;
  }
  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueEventsWait(
    ur_queue_handle_t hQueue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {

  // TODO: the wait here should be async
  return withTimingEvent(UR_COMMAND_EVENTS_WAIT, hQueue, numEventsInWaitList,
                         phEventWaitList, phEvent,
                         [&]() { return UR_RESULT_SUCCESS; });
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueEventsWaitWithBarrier(
    ur_queue_handle_t hQueue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  return withTimingEvent(UR_COMMAND_EVENTS_WAIT_WITH_BARRIER, hQueue,
                         numEventsInWaitList, phEventWaitList, phEvent,
                         [&]() { return UR_RESULT_SUCCESS; });
}

UR_APIEXPORT ur_result_t urEnqueueEventsWaitWithBarrierExt(
    ur_queue_handle_t hQueue, const ur_exp_enqueue_ext_properties_t *,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  return urEnqueueEventsWaitWithBarrier(hQueue, numEventsInWaitList,
                                        phEventWaitList, phEvent);
}

template <bool IsRead>
static inline ur_result_t enqueueMemBufferReadWriteRect_impl(
    ur_queue_handle_t hQueue, ur_mem_handle_t Buff, bool,
    ur_rect_offset_t BufferOffset, ur_rect_offset_t HostOffset,
    ur_rect_region_t region, size_t BufferRowPitch, size_t BufferSlicePitch,
    size_t HostRowPitch, size_t HostSlicePitch,
    typename std::conditional<IsRead, void *, const void *>::type DstMem,
    uint32_t NumEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  ur_command_t command_t;
  if constexpr (IsRead)
    command_t = UR_COMMAND_MEM_BUFFER_READ_RECT;
  else
    command_t = UR_COMMAND_MEM_BUFFER_WRITE_RECT;
  return withTimingEvent(
      command_t, hQueue, NumEventsInWaitList, phEventWaitList, phEvent, [&]() {
        // TODO: blocking, check other constraints, performance optimizations
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
      });
}

static inline ur_result_t doCopy_impl(ur_queue_handle_t hQueue, void *DstPtr,
                                      const void *SrcPtr, size_t Size,
                                      uint32_t numEventsInWaitList,
                                      const ur_event_handle_t *phEventWaitList,
                                      ur_event_handle_t *phEvent,
                                      ur_command_t command_type) {
  return withTimingEvent(command_type, hQueue, numEventsInWaitList,
                         phEventWaitList, phEvent, [&]() {
                           if (SrcPtr != DstPtr && Size)
                             memmove(DstPtr, SrcPtr, Size);
                           return UR_RESULT_SUCCESS;
                         });
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferRead(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingRead,
    size_t offset, size_t size, void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = blockingRead;

  void *FromPtr = /*Src*/ hBuffer->_mem + offset;
  auto res = doCopy_impl(hQueue, pDst, FromPtr, size, numEventsInWaitList,
                         phEventWaitList, phEvent, UR_COMMAND_MEM_BUFFER_READ);
  return res;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferWrite(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingWrite,
    size_t offset, size_t size, const void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = blockingWrite;

  void *ToPtr = hBuffer->_mem + offset;
  auto res = doCopy_impl(hQueue, ToPtr, pSrc, size, numEventsInWaitList,
                         phEventWaitList, phEvent, UR_COMMAND_MEM_BUFFER_WRITE);
  return res;
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
  urEventWait(numEventsInWaitList, phEventWaitList);
  const void *SrcPtr = hBufferSrc->_mem + srcOffset;
  void *DstPtr = hBufferDst->_mem + dstOffset;
  return doCopy_impl(hQueue, DstPtr, SrcPtr, size, numEventsInWaitList,
                     phEventWaitList, phEvent, UR_COMMAND_MEM_BUFFER_COPY);
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

  return withTimingEvent(
      UR_COMMAND_MEM_BUFFER_FILL, hQueue, numEventsInWaitList, phEventWaitList,
      phEvent, [&]() {
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
      });
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
  std::ignore = blockingMap;
  std::ignore = mapFlags;
  std::ignore = size;

  return withTimingEvent(UR_COMMAND_MEM_BUFFER_MAP, hQueue, numEventsInWaitList,
                         phEventWaitList, phEvent, [&]() {
                           *ppRetMap = hBuffer->_mem + offset;
                           return UR_RESULT_SUCCESS;
                         });
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemUnmap(
    ur_queue_handle_t hQueue, ur_mem_handle_t hMem, void *pMappedPtr,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  std::ignore = hMem;
  std::ignore = pMappedPtr;
  return withTimingEvent(UR_COMMAND_MEM_UNMAP, hQueue, numEventsInWaitList,
                         phEventWaitList, phEvent,
                         [&]() { return UR_RESULT_SUCCESS; });
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMFill(
    ur_queue_handle_t hQueue, void *ptr, size_t patternSize,
    const void *pPattern, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  return withTimingEvent(
      UR_COMMAND_USM_FILL, hQueue, numEventsInWaitList, phEventWaitList,
      phEvent, [&]() {
        UR_ASSERT(ptr, UR_RESULT_ERROR_INVALID_NULL_POINTER);
        UR_ASSERT(pPattern, UR_RESULT_ERROR_INVALID_NULL_POINTER);
        UR_ASSERT(patternSize != 0, UR_RESULT_ERROR_INVALID_SIZE)
        UR_ASSERT(size != 0, UR_RESULT_ERROR_INVALID_SIZE)
        UR_ASSERT(patternSize < size, UR_RESULT_ERROR_INVALID_SIZE)
        UR_ASSERT(size % patternSize == 0, UR_RESULT_ERROR_INVALID_SIZE)
        // TODO: add check for allocation size once the query is supported

        switch (patternSize) {
        case 1:
          memset(ptr, *static_cast<const uint8_t *>(pPattern),
                 size * patternSize);
          break;
        case 2: {
          const auto pattern = *static_cast<const uint16_t *>(pPattern);
          auto *start = reinterpret_cast<uint16_t *>(ptr);
          auto *end = reinterpret_cast<uint16_t *>(
              reinterpret_cast<uint8_t *>(ptr) + size);
          std::fill(start, end, pattern);
          break;
        }
        case 4: {
          const auto pattern = *static_cast<const uint32_t *>(pPattern);
          auto *start = reinterpret_cast<uint32_t *>(ptr);
          auto *end = reinterpret_cast<uint32_t *>(
              reinterpret_cast<uint8_t *>(ptr) + size);
          std::fill(start, end, pattern);
          break;
        }
        case 8: {
          const auto pattern = *static_cast<const uint64_t *>(pPattern);
          auto *start = reinterpret_cast<uint64_t *>(ptr);
          auto *end = reinterpret_cast<uint64_t *>(
              reinterpret_cast<uint8_t *>(ptr) + size);
          std::fill(start, end, pattern);
          break;
        }
        default: {
          for (unsigned int step{0}; step < size; step += patternSize) {
            auto *dest = reinterpret_cast<void *>(
                reinterpret_cast<uint8_t *>(ptr) + step);
            memcpy(dest, pPattern, patternSize);
          }
        }
        }
        return UR_RESULT_SUCCESS;
      });
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMMemcpy(
    ur_queue_handle_t hQueue, bool blocking, void *pDst, const void *pSrc,
    size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = blocking;
  return withTimingEvent(
      UR_COMMAND_USM_MEMCPY, hQueue, numEventsInWaitList, phEventWaitList,
      phEvent, [&]() {
        UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_QUEUE);
        UR_ASSERT(pDst, UR_RESULT_ERROR_INVALID_NULL_POINTER);
        UR_ASSERT(pSrc, UR_RESULT_ERROR_INVALID_NULL_POINTER);

        memcpy(pDst, pSrc, size);

        return UR_RESULT_SUCCESS;
      });
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

  // TODO: properly implement USM prefetch
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueUSMAdvise(ur_queue_handle_t hQueue, const void *pMem, size_t size,
                   ur_usm_advice_flags_t advice, ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = pMem;
  std::ignore = size;
  std::ignore = advice;
  std::ignore = phEvent;

  // TODO: properly implement USM advise
  return UR_RESULT_SUCCESS;
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

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueNativeCommandExp(
    ur_queue_handle_t, ur_exp_enqueue_native_command_function_t, void *,
    uint32_t, const ur_mem_handle_t *,
    const ur_exp_enqueue_native_command_properties_t *, uint32_t,
    const ur_event_handle_t *, ur_event_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
