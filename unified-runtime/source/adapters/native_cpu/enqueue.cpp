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
      GlobalOffset[I] = GlobalWorkOffset ? GlobalWorkOffset[I] : 0;
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

namespace {
class WaitInfo {
  std::vector<ur_event_handle_t> *const events;
  static_assert(std::is_pointer_v<ur_event_handle_t>);

public:
  WaitInfo(uint32_t numEvents, const ur_event_handle_t *WaitList)
      : events(numEvents ? new std::vector<ur_event_handle_t>(
                               WaitList, WaitList + numEvents)
                         : nullptr) {}
  void wait() const {
    if (events)
      urEventWait(events->size(), events->data());
  }
  std::unique_ptr<std::vector<ur_event_handle_t>> getUniquePtr() {
    return std::unique_ptr<std::vector<ur_event_handle_t>>(events);
  }
};

inline static WaitInfo getWaitInfo(uint32_t numEventsInWaitList,
                                   const ur_event_handle_t *phEventWaitList) {
  return native_cpu::WaitInfo(numEventsInWaitList, phEventWaitList);
}

} // namespace
} // namespace native_cpu

static inline native_cpu::state getResizedState(const native_cpu::NDRDescT &ndr,
                                                size_t itemsPerThread) {
  native_cpu::state resized_state(
      ndr.GlobalSize[0], ndr.GlobalSize[1], ndr.GlobalSize[2], itemsPerThread,
      ndr.LocalSize[1], ndr.LocalSize[2], ndr.GlobalOffset[0],
      ndr.GlobalOffset[1], ndr.GlobalOffset[2]);
  return resized_state;
}

static inline native_cpu::state getState(const native_cpu::NDRDescT &ndr) {
  return getResizedState(ndr, ndr.LocalSize[0]);
}

using IndexT = std::array<size_t, 3>;
using RangeT = native_cpu::NDRDescT::RangeT;

static inline void execute_range(native_cpu::state &state,
                                 const ur_kernel_handle_t_ &hKernel,
                                 const std::vector<void *> &args, IndexT first,
                                 IndexT lastPlusOne) {
  for (size_t g2 = first[2]; g2 < lastPlusOne[2]; g2++) {
    for (size_t g1 = first[1]; g1 < lastPlusOne[1]; g1++) {
      for (size_t g0 = first[0]; g0 < lastPlusOne[0]; g0 += 1) {
        state.update(g0, g1, g2);
        hKernel._subhandler(args.data(), &state);
      }
    }
  }
}

static inline void execute_range(native_cpu::state &state,
                                 const ur_kernel_handle_t_ &hKernel,
                                 IndexT first, IndexT lastPlusOne) {
  execute_range(state, hKernel, hKernel.getArgs(), first, lastPlusOne);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueKernelLaunch(
    ur_queue_handle_t hQueue, ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numPropsInLaunchPropList,
    const ur_kernel_launch_property_t *launchPropList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  // We don't support any launch properties.
  for (uint32_t propIndex = 0; propIndex < numPropsInLaunchPropList;
       propIndex++) {
    if (launchPropList[propIndex].id != UR_KERNEL_LAUNCH_PROPERTY_ID_IGNORE) {
      return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }
  }

  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
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
  auto Tasks = native_cpu::getScheduler(tp);
  auto numWG0 = ndr.GlobalSize[0] / ndr.LocalSize[0];
  auto numWG1 = ndr.GlobalSize[1] / ndr.LocalSize[1];
  auto numWG2 = ndr.GlobalSize[2] / ndr.LocalSize[2];
  auto event = new ur_event_handle_t_(hQueue, UR_COMMAND_KERNEL_LAUNCH);
  event->tick_start();

  // Create a copy of the kernel and its arguments.
  auto kernel = std::make_unique<ur_kernel_handle_t_>(*hKernel);
  kernel->updateMemPool(numParallelThreads);

  auto InEvents = native_cpu::getWaitInfo(numEventsInWaitList, phEventWaitList);

#ifndef NATIVECPU_USE_OCK
  native_cpu::state state = getState(ndr);
  urEventWait(numEventsInWaitList, phEventWaitList);
  for (unsigned g2 = 0; g2 < numWG2; g2++) {
    for (unsigned g1 = 0; g1 < numWG1; g1++) {
      for (unsigned g0 = 0; g0 < numWG0; g0++) {
        for (unsigned local2 = 0; local2 < ndr.LocalSize[2]; local2++) {
          for (unsigned local1 = 0; local1 < ndr.LocalSize[1]; local1++) {
            for (unsigned local0 = 0; local0 < ndr.LocalSize[0]; local0++) {
              state.update(g0, g1, g2, local0, local1, local2);
              kernel->_subhandler(kernel->getArgs(1, 0).data(), &state);
            }
          }
        }
      }
    }
  }
#else
  bool isLocalSizeOne =
      ndr.LocalSize[0] == 1 && ndr.LocalSize[1] == 1 && ndr.LocalSize[2] == 1;
  if (isLocalSizeOne && !kernel->hasLocalArgs()) {
    // If the local size is one, we make the assumption that we are running a
    // parallel_for over a sycl::range.
    // Todo: we could add more compiler checks and
    // kernel properties for this (e.g. check that no barriers are called).

    // Todo: this assumes that dim 0 is the best dimension over which we want to
    // parallelize

    // Since we also vectorize the kernel, and vectorization happens within the
    // work group loop, it's better to have a large-ish local size. We can
    // divide the global range by the number of threads, set that as the local
    // size and peel everything else.

    // The number of items per kernel invocation should ideally be at least a
    // multiple of the applied vector width, which we currently assume to be 8.
    // TODO: Encode this and other kernel capabilities in the binary so we can
    // use actual values to efficiently enqueue kernels instead of relying on
    // assumptions.
    const size_t itemsPerKernelInvocation = 8;

    size_t itemsPerThread = ndr.GlobalSize[0] / numParallelThreads;
    if (itemsPerThread < itemsPerKernelInvocation) {
      if (itemsPerKernelInvocation <= numWG0)
        itemsPerThread = itemsPerKernelInvocation;
      else if (itemsPerThread == 0)
        itemsPerThread = numWG0;
    } else if (itemsPerThread > itemsPerKernelInvocation) {
      // Launch kernel with number of items that is the next multiple of the
      // vector width.
      const size_t nextMult = (itemsPerThread + itemsPerKernelInvocation - 1) /
                              itemsPerKernelInvocation *
                              itemsPerKernelInvocation;
      if (nextMult < numWG0)
        itemsPerThread = nextMult;
    }

    size_t wg0_index = 0;
    for (size_t t = 0; (wg0_index + itemsPerThread) <= numWG0;
         wg0_index += itemsPerThread) {
      IndexT first = {t, 0, 0};
      IndexT last = {++t, numWG1, numWG2};
      Tasks.schedule([ndr, itemsPerThread, &kernel = *kernel, first, last,
                      InEvents](size_t) {
        native_cpu::state resized_state = getResizedState(ndr, itemsPerThread);
        InEvents.wait();
        execute_range(resized_state, kernel, first, last);
      });
    }

    if (wg0_index < numWG0) {
      // Peel the remaining work items. Since the local size is 1, we iterate
      // over the work groups.
      Tasks.schedule([ndr, &kernel = *kernel, wg0_index, numWG0, numWG1, numWG2,
                      InEvents](size_t) {
        IndexT first = {wg0_index, 0, 0};
        IndexT last = {numWG0, numWG1, numWG2};
        InEvents.wait();
        native_cpu::state state = getState(ndr);
        execute_range(state, kernel, first, last);
      });
    }
  } else {
    // We are running a parallel_for over an nd_range

    const IndexT numWG = {numWG0, numWG1, numWG2};
    IndexT groupsPerThread;
    for (size_t t = 0; t < 3; t++)
      groupsPerThread[t] = numWG[t] / numParallelThreads;
    size_t dim = 0;
    if (groupsPerThread[0] == 0) {
      if (groupsPerThread[1])
        dim = 1;
      else if (groupsPerThread[2])
        dim = 2;
    }
    IndexT first = {0, 0, 0}, last = numWG;
    size_t wg_start = 0;
    if (groupsPerThread[dim]) {
      for (size_t t = 0; t < numParallelThreads; t++) {
        first[dim] = wg_start;
        wg_start += groupsPerThread[dim];
        last[dim] = wg_start;
        Tasks.schedule([ndr, numParallelThreads, &kernel = *kernel, first, last,
                        InEvents](size_t threadId) {
          InEvents.wait();
          native_cpu::state state = getState(ndr);
          execute_range(state, kernel,
                        kernel.getArgs(numParallelThreads, threadId), first,
                        last);
        });
      }
    }
    if (wg_start < numWG[dim]) {
      first[dim] = wg_start;
      last[dim] = numWG[dim];
      Tasks.schedule([ndr, numParallelThreads, &kernel = *kernel, first, last,
                      InEvents](size_t threadId) {
        InEvents.wait();
        native_cpu::state state = getState(ndr);
        execute_range(state, kernel,
                      kernel.getArgs(numParallelThreads, threadId), first,
                      last);
      });
    }
  }

#endif // NATIVECPU_USE_OCK
  event->set_futures(Tasks.getTaskInfo());

  if (phEvent) {
    *phEvent = event;
  }
  event->set_callback([kernel = std::move(kernel), hKernel, event,
                       InEvents = InEvents.getUniquePtr()]() {
    event->tick_end();
    // TODO: avoid calling clear() here.
    hKernel->_localArgInfo.clear();
  });

  if (hQueue->isInOrder()) {
    urEventWait(1, &event);
  }

  return UR_RESULT_SUCCESS;
}

template <class T>
static inline ur_result_t
withTimingEvent(ur_command_t command_type, ur_queue_handle_t hQueue,
                uint32_t numEventsInWaitList,
                const ur_event_handle_t *phEventWaitList,
                ur_event_handle_t *phEvent, T &&f, bool blocking = true) {
  if (phEvent) {
    ur_event_handle_t event = new ur_event_handle_t_(hQueue, command_type);
    *phEvent = event;
    event->tick_start();
    if (blocking || hQueue->isInOrder()) {
      urEventWait(numEventsInWaitList, phEventWaitList);
      ur_result_t result = f();
      event->tick_end();
      return result;
    }
    auto &tp = hQueue->getDevice()->tp;
    auto Tasks = native_cpu::getScheduler(tp);
    auto InEvents =
        native_cpu::getWaitInfo(numEventsInWaitList, phEventWaitList);
    Tasks.schedule([f, InEvents](size_t) {
      InEvents.wait();
      f();
    });
    event->set_futures(Tasks.getTaskInfo());
    event->set_callback(
        [event, InEvents = InEvents.getUniquePtr()]() { event->tick_end(); });
    return UR_RESULT_SUCCESS;
  }
  urEventWait(numEventsInWaitList, phEventWaitList);
  ur_result_t result = f();
  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueEventsWait(
    ur_queue_handle_t hQueue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {

  // TODO: the wait here should be async
  return withTimingEvent(UR_COMMAND_EVENTS_WAIT, hQueue, numEventsInWaitList,
                         phEventWaitList, phEvent,
                         []() { return UR_RESULT_SUCCESS; });
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueEventsWaitWithBarrier(
    ur_queue_handle_t hQueue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  return withTimingEvent(UR_COMMAND_EVENTS_WAIT_WITH_BARRIER, hQueue,
                         numEventsInWaitList, phEventWaitList, phEvent,
                         []() { return UR_RESULT_SUCCESS; });
}

UR_APIEXPORT ur_result_t urEnqueueEventsWaitWithBarrierExt(
    ur_queue_handle_t hQueue, const ur_exp_enqueue_ext_properties_t *,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  return urEnqueueEventsWaitWithBarrier(hQueue, numEventsInWaitList,
                                        phEventWaitList, phEvent);
}

template <bool IsRead>
static inline void MemBufferReadWriteRect_impl(
    ur_mem_handle_t Buff, ur_rect_offset_t BufferOffset,
    ur_rect_offset_t HostOffset, ur_rect_region_t region, size_t BufferRowPitch,
    size_t BufferSlicePitch, size_t HostRowPitch, size_t HostSlicePitch,
    typename std::conditional<IsRead, void *, const void *>::type DstMem) {
  // TODO: check other constraints, performance optimizations
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
}

template <bool IsRead>
static inline ur_result_t enqueueMemBufferReadWriteRect_impl(
    ur_queue_handle_t hQueue, ur_mem_handle_t Buff, bool blocking,
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
      command_t, hQueue, NumEventsInWaitList, phEventWaitList, phEvent,
      [BufferRowPitch, region, BufferSlicePitch, HostRowPitch, HostSlicePitch,
       BufferOffset, HostOffset, Buff, DstMem]() {
        MemBufferReadWriteRect_impl<IsRead>(
            Buff, BufferOffset, HostOffset, region, BufferRowPitch,
            BufferSlicePitch, HostRowPitch, HostSlicePitch, DstMem);
        return UR_RESULT_SUCCESS;
      },
      blocking);
}

template <bool AllowPartialOverlap = true>
static inline ur_result_t doCopy_impl(
    ur_queue_handle_t hQueue, void *DstPtr, const void *SrcPtr, size_t Size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent, ur_command_t command_type, bool blocking) {
  if (SrcPtr == DstPtr || Size == 0) {
    bool hasInEvents = numEventsInWaitList && phEventWaitList;
    return withTimingEvent(
        command_type, hQueue, numEventsInWaitList, phEventWaitList, phEvent,
        []() { return UR_RESULT_SUCCESS; }, blocking || !hasInEvents);
  }

  return withTimingEvent(
      command_type, hQueue, numEventsInWaitList, phEventWaitList, phEvent,
      [DstPtr, SrcPtr, Size]() {
        if constexpr (AllowPartialOverlap) {
          memmove(DstPtr, SrcPtr, Size);
        } else {
          memcpy(DstPtr, SrcPtr, Size);
        }
        return UR_RESULT_SUCCESS;
      },
      blocking);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferRead(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingRead,
    size_t offset, size_t size, void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {

  void *FromPtr = /*Src*/ hBuffer->_mem + offset;
  auto res = doCopy_impl(hQueue, pDst, FromPtr, size, numEventsInWaitList,
                         phEventWaitList, phEvent, UR_COMMAND_MEM_BUFFER_READ,
                         blockingRead);
  return res;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferWrite(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingWrite,
    size_t offset, size_t size, const void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {

  void *ToPtr = hBuffer->_mem + offset;
  auto res = doCopy_impl(hQueue, ToPtr, pSrc, size, numEventsInWaitList,
                         phEventWaitList, phEvent, UR_COMMAND_MEM_BUFFER_WRITE,
                         blockingWrite);
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
                     phEventWaitList, phEvent, UR_COMMAND_MEM_BUFFER_COPY,
                     true /*TODO: check false for non-blocking*/);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferCopyRect(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBufferSrc,
    ur_mem_handle_t hBufferDst, ur_rect_offset_t srcOrigin,
    ur_rect_offset_t dstOrigin, ur_rect_region_t region, size_t srcRowPitch,
    size_t srcSlicePitch, size_t dstRowPitch, size_t dstSlicePitch,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  return enqueueMemBufferReadWriteRect_impl<true /*read*/>(
      hQueue, hBufferSrc, true /*todo: check false for non-blocking*/,
      srcOrigin,
      /*HostOffset*/ dstOrigin, region, srcRowPitch, srcSlicePitch, dstRowPitch,
      dstSlicePitch, hBufferDst->_mem, numEventsInWaitList, phEventWaitList,
      phEvent);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferFill(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, const void *pPattern,
    size_t patternSize, size_t offset, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  return withTimingEvent(
      UR_COMMAND_MEM_BUFFER_FILL, hQueue, numEventsInWaitList, phEventWaitList,
      phEvent, [hBuffer, offset, size, patternSize, pPattern]() {
        // TODO: error checking
        // TODO: handle async
        void *startingPtr = hBuffer->_mem + offset;
        size_t steps = size / patternSize;
        for (unsigned i = 0; i < steps; i++) {
          memcpy(static_cast<int8_t *>(startingPtr) + i * patternSize, pPattern,
                 patternSize);
        }

        return UR_RESULT_SUCCESS;
      });
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemImageRead(
    ur_queue_handle_t /*hQueue*/, ur_mem_handle_t /*hImage*/,
    bool /*blockingRead*/, ur_rect_offset_t /*origin*/,
    ur_rect_region_t /*region*/, size_t /*rowPitch*/, size_t /*slicePitch*/,
    void * /*pDst*/, uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_event_handle_t * /*phEvent*/) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemImageWrite(
    ur_queue_handle_t /*hQueue*/, ur_mem_handle_t /*hImage*/,
    bool /*blockingWrite*/, ur_rect_offset_t /*origin*/,
    ur_rect_region_t /*region*/, size_t /*rowPitch*/, size_t /*slicePitch*/,
    void * /*pSrc*/, uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_event_handle_t * /*phEvent*/) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemImageCopy(
    ur_queue_handle_t /*hQueue*/, ur_mem_handle_t /*hImageSrc*/,
    ur_mem_handle_t /*hImageDst*/, ur_rect_offset_t /*srcOrigin*/,
    ur_rect_offset_t /*dstOrigin*/, ur_rect_region_t /*region*/,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_event_handle_t * /*phEvent*/) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferMap(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool /*blockingMap*/,
    ur_map_flags_t /*mapFlags*/, size_t offset, size_t /*size*/,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent, void **ppRetMap) {

  return withTimingEvent(UR_COMMAND_MEM_BUFFER_MAP, hQueue, numEventsInWaitList,
                         phEventWaitList, phEvent,
                         [ppRetMap, hBuffer, offset]() {
                           *ppRetMap = hBuffer->_mem + offset;
                           return UR_RESULT_SUCCESS;
                         });
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemUnmap(
    ur_queue_handle_t hQueue, ur_mem_handle_t /*hMem*/, void * /*pMappedPtr*/,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  return withTimingEvent(UR_COMMAND_MEM_UNMAP, hQueue, numEventsInWaitList,
                         phEventWaitList, phEvent,
                         []() { return UR_RESULT_SUCCESS; });
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMFill(
    ur_queue_handle_t hQueue, void *ptr, size_t patternSize,
    const void *pPattern, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  return withTimingEvent(
      UR_COMMAND_USM_FILL, hQueue, numEventsInWaitList, phEventWaitList,
      phEvent, [ptr, pPattern, patternSize, size]() {
        UR_ASSERT(ptr, UR_RESULT_ERROR_INVALID_NULL_POINTER);
        UR_ASSERT(pPattern, UR_RESULT_ERROR_INVALID_NULL_POINTER);
        UR_ASSERT(patternSize != 0, UR_RESULT_ERROR_INVALID_SIZE)
        UR_ASSERT(size != 0, UR_RESULT_ERROR_INVALID_SIZE)
        UR_ASSERT(patternSize <= size, UR_RESULT_ERROR_INVALID_SIZE)
        UR_ASSERT(size % patternSize == 0, UR_RESULT_ERROR_INVALID_SIZE)
        // TODO: add check for allocation size once the query is supported

        switch (patternSize) {
        case 1:
          memset(ptr, *static_cast<const uint8_t *>(pPattern), size);
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
          for (size_t step{0}; step < size; step += patternSize) {
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
  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_QUEUE);
  UR_ASSERT(pDst, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(pSrc, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  return doCopy_impl<false /*use memcpy*/>(
      hQueue, pDst, pSrc, size, numEventsInWaitList, phEventWaitList, phEvent,
      UR_COMMAND_USM_MEMCPY, blocking);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMPrefetch(
    ur_queue_handle_t /*hQueue*/, const void * /*pMem*/, size_t /*size*/,
    ur_usm_migration_flags_t /*flags*/, uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_event_handle_t * /*phEvent*/) {

  // TODO: properly implement USM prefetch
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMAdvise(
    ur_queue_handle_t /*hQueue*/, const void * /*pMem*/, size_t /*size*/,
    ur_usm_advice_flags_t /*advice*/, ur_event_handle_t * /*phEvent*/) {

  // TODO: properly implement USM advise
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMFill2D(
    ur_queue_handle_t /*hQueue*/, void * /*pMem*/, size_t /*pitch*/,
    size_t /*patternSize*/, const void * /*pPattern*/, size_t /*width*/,
    size_t /*height*/, uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_event_handle_t * /*phEvent*/) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMMemcpy2D(
    ur_queue_handle_t /*hQueue*/, bool /*blocking*/, void * /*pDst*/,
    size_t /*dstPitch*/, const void * /*pSrc*/, size_t /*srcPitch*/,
    size_t /*width*/, size_t /*height*/, uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_event_handle_t * /*phEvent*/) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueDeviceGlobalVariableWrite(
    ur_queue_handle_t /*hQueue*/, ur_program_handle_t /*hProgram*/,
    const char * /*name*/, bool /*blockingWrite*/, size_t /*count*/,
    size_t /*offset*/, const void * /*pSrc*/, uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_event_handle_t * /*phEvent*/) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueDeviceGlobalVariableRead(
    ur_queue_handle_t /*hQueue*/, ur_program_handle_t /*hProgram*/,
    const char * /*name*/, bool /*blockingRead*/, size_t /*count*/,
    size_t /*offset*/, void * /*pDst*/, uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_event_handle_t * /*phEvent*/) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueReadHostPipe(
    ur_queue_handle_t /*hQueue*/, ur_program_handle_t /*hProgram*/,
    const char * /*pipe_symbol*/, bool /*blocking*/, void * /*pDst*/,
    size_t /*size*/, uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_event_handle_t * /*phEvent*/) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueWriteHostPipe(
    ur_queue_handle_t /*hQueue*/, ur_program_handle_t /*hProgram*/,
    const char * /*pipe_symbol*/, bool /*blocking*/, void * /*pSrc*/,
    size_t /*size*/, uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_event_handle_t * /*phEvent*/) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueNativeCommandExp(
    ur_queue_handle_t, ur_exp_enqueue_native_command_function_t, void *,
    uint32_t, const ur_mem_handle_t *,
    const ur_exp_enqueue_native_command_properties_t *, uint32_t,
    const ur_event_handle_t *, ur_event_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
