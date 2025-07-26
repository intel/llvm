//===----------- enqueue.cpp - LLVM Offload Adapter  ----------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>
#include <assert.h>
#include <ur_api.h>

#include "context.hpp"
#include "event.hpp"
#include "kernel.hpp"
#include "memory.hpp"
#include "queue.hpp"
#include "ur2offload.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueKernelLaunch(
    ur_queue_handle_t hQueue, ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t, const ur_kernel_launch_property_t *,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  // Ignore wait list for now
  (void)numEventsInWaitList;
  (void)phEventWaitList;
  //

  (void)pGlobalWorkOffset;

  size_t GlobalSize[3] = {1, 1, 1};
  for (uint32_t I = 0; I < workDim; I++) {
    GlobalSize[I] = pGlobalWorkSize[I];
  }

  // TODO: We default to 1, 1, 1 here. In future if pLocalWorkSize is not
  // specified, we should pick the "best" one
  size_t GroupSize[3] = {1, 1, 1};
  if (pLocalWorkSize) {
    for (uint32_t I = 0; I < workDim; I++) {
      GroupSize[I] = pLocalWorkSize[I];
    }
  }

  if (GroupSize[0] > GlobalSize[0] || GroupSize[1] > GlobalSize[1] ||
      GroupSize[2] > GlobalSize[2] ||
      GroupSize[0] > std::numeric_limits<uint32_t>::max() ||
      GroupSize[1] > std::numeric_limits<uint32_t>::max() ||
      GroupSize[2] > std::numeric_limits<uint32_t>::max() ||
      GlobalSize[0] / GroupSize[0] > std::numeric_limits<uint32_t>::max() ||
      GlobalSize[1] / GroupSize[1] > std::numeric_limits<uint32_t>::max() ||
      GlobalSize[2] / GroupSize[2] > std::numeric_limits<uint32_t>::max()) {
    return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
  }

  ol_kernel_launch_size_args_t LaunchArgs;
  LaunchArgs.Dimensions = workDim;
  LaunchArgs.NumGroups.x = GlobalSize[0] / GroupSize[0];
  LaunchArgs.NumGroups.y = GlobalSize[1] / GroupSize[1];
  LaunchArgs.NumGroups.z = GlobalSize[2] / GroupSize[2];
  LaunchArgs.GroupSize.x = GroupSize[0];
  LaunchArgs.GroupSize.y = GroupSize[1];
  LaunchArgs.GroupSize.z = GroupSize[2];
  LaunchArgs.DynSharedMemory = 0;

  ol_queue_handle_t Queue;
  OL_RETURN_ON_ERR(hQueue->nextQueue(Queue));
  OL_RETURN_ON_ERR(olLaunchKernel(
      Queue, hQueue->OffloadDevice, hKernel->OffloadKernel,
      hKernel->Args.getStorage(), hKernel->Args.getStorageSize(), &LaunchArgs));

  if (phEvent) {
    auto *Event = new ur_event_handle_t_(UR_COMMAND_KERNEL_LAUNCH, hQueue);
    if (auto Res = olCreateEvent(Queue, &Event->OffloadEvent)) {
      delete Event;
      return offloadResultToUR(Res);
    };
    *phEvent = Event;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMFill2D(
    ur_queue_handle_t, void *, size_t, size_t, const void *, size_t, size_t,
    uint32_t, const ur_event_handle_t *, ur_event_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMMemcpy2D(
    ur_queue_handle_t, bool, void *, size_t, const void *, size_t, size_t,
    size_t, uint32_t, const ur_event_handle_t *, ur_event_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

namespace {
ur_result_t doMemcpy(ur_command_t Command, ur_queue_handle_t hQueue,
                     void *DestPtr, ol_device_handle_t DestDevice,
                     const void *SrcPtr, ol_device_handle_t SrcDevice,
                     size_t size, bool blocking, uint32_t numEventsInWaitList,
                     const ur_event_handle_t *phEventWaitList,
                     ur_event_handle_t *phEvent) {
  // Ignore wait list for now
  (void)numEventsInWaitList;
  (void)phEventWaitList;
  //

  if (blocking) {
    OL_RETURN_ON_ERR(
        olMemcpy(nullptr, DestPtr, DestDevice, SrcPtr, SrcDevice, size));
    if (phEvent) {
      *phEvent = ur_event_handle_t_::createEmptyEvent(Command, hQueue);
    }
    return UR_RESULT_SUCCESS;
  }

  ol_queue_handle_t Queue;
  OL_RETURN_ON_ERR(hQueue->nextQueue(Queue));
  OL_RETURN_ON_ERR(
      olMemcpy(Queue, DestPtr, DestDevice, SrcPtr, SrcDevice, size));
  if (phEvent) {
    auto *Event = new ur_event_handle_t_(Command, hQueue);
    if (auto Res = olCreateEvent(Queue, &Event->OffloadEvent)) {
      delete Event;
      return offloadResultToUR(Res);
    };
    *phEvent = Event;
  }

  return UR_RESULT_SUCCESS;
}
} // namespace

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferRead(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingRead,
    size_t offset, size_t size, void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  char *DevPtr =
      reinterpret_cast<char *>(std::get<BufferMem>(hBuffer->Mem).Ptr);

  return doMemcpy(UR_COMMAND_MEM_BUFFER_READ, hQueue, pDst, Adapter->HostDevice,
                  DevPtr + offset, hQueue->OffloadDevice, size, blockingRead,
                  numEventsInWaitList, phEventWaitList, phEvent);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferWrite(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingWrite,
    size_t offset, size_t size, const void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  char *DevPtr =
      reinterpret_cast<char *>(std::get<BufferMem>(hBuffer->Mem).Ptr);

  return doMemcpy(UR_COMMAND_MEM_BUFFER_WRITE, hQueue, DevPtr + offset,
                  hQueue->OffloadDevice, pSrc, Adapter->HostDevice, size,
                  blockingWrite, numEventsInWaitList, phEventWaitList, phEvent);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueDeviceGlobalVariableRead(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram, const char *name,
    bool blockingRead, size_t count, size_t offset, void *pDst,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  void *Ptr;
  if (auto Err = urProgramGetGlobalVariablePointer(nullptr, hProgram, name,
                                                   nullptr, &Ptr)) {
    return Err;
  }

  return doMemcpy(
      UR_COMMAND_DEVICE_GLOBAL_VARIABLE_READ, hQueue, pDst, Adapter->HostDevice,
      reinterpret_cast<const char *>(Ptr) + offset, hQueue->OffloadDevice,
      count, blockingRead, numEventsInWaitList, phEventWaitList, phEvent);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueDeviceGlobalVariableWrite(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram, const char *name,
    bool blockingWrite, size_t count, size_t offset, const void *pSrc,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  void *Ptr;
  if (auto Err = urProgramGetGlobalVariablePointer(nullptr, hProgram, name,
                                                   nullptr, &Ptr)) {
    return Err;
  }

  return doMemcpy(UR_COMMAND_DEVICE_GLOBAL_VARIABLE_WRITE, hQueue,
                  reinterpret_cast<char *>(Ptr) + offset, hQueue->OffloadDevice,
                  pSrc, Adapter->HostDevice, count, blockingWrite,
                  numEventsInWaitList, phEventWaitList, phEvent);
}

ur_result_t enqueueNoOp(ur_command_t Type, ur_queue_handle_t hQueue,
                        ur_event_handle_t *phEvent) {
  // This path is a no-op, but we can't output a real event because
  // Offload doesn't currently support creating arbitrary events, and we
  // don't know the last real event in the queue. Instead we just have to
  // wait on the whole queue and then return an empty (implicitly
  // finished) event.
  *phEvent = ur_event_handle_t_::createEmptyEvent(Type, hQueue);
  return urQueueFinish(hQueue);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferMap(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingMap,
    ur_map_flags_t mapFlags, size_t offset, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent, void **ppRetMap) {

  auto &BufferImpl = std::get<BufferMem>(hBuffer->Mem);
  auto MapPtr = BufferImpl.mapToPtr(size, offset, mapFlags);

  if (!MapPtr) {
    return UR_RESULT_ERROR_INVALID_MEM_OBJECT;
  }

  const bool IsPinned =
      BufferImpl.MemAllocMode == BufferMem::AllocMode::AllocHostPtr;

  ur_result_t Result = UR_RESULT_SUCCESS;
  if (!IsPinned &&
      ((mapFlags & UR_MAP_FLAG_READ) || (mapFlags & UR_MAP_FLAG_WRITE))) {
    // Pinned host memory is already on host so it doesn't need to be read.
    Result = urEnqueueMemBufferRead(hQueue, hBuffer, blockingMap, offset, size,
                                    MapPtr, numEventsInWaitList,
                                    phEventWaitList, phEvent);
  } else {
    if (IsPinned) {
      // TODO: Ignore the event waits list for now. When urEnqueueEventsWait is
      // implemented we can call it on the wait list.
    }

    if (phEvent) {
      enqueueNoOp(UR_COMMAND_MEM_BUFFER_MAP, hQueue, phEvent);
    }
  }
  *ppRetMap = MapPtr;

  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemUnmap(
    ur_queue_handle_t hQueue, ur_mem_handle_t hMem, void *pMappedPtr,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  auto &BufferImpl = std::get<BufferMem>(hMem->Mem);

  auto *Map = BufferImpl.getMapDetails(pMappedPtr);
  UR_ASSERT(Map != nullptr, UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  const bool IsPinned =
      BufferImpl.MemAllocMode == BufferMem::AllocMode::AllocHostPtr;

  ur_result_t Result = UR_RESULT_SUCCESS;
  if (!IsPinned && ((Map->MapFlags & UR_MAP_FLAG_WRITE) ||
                    (Map->MapFlags & UR_MAP_FLAG_WRITE_INVALIDATE_REGION))) {
    // Pinned host memory is only on host so it doesn't need to be written to.
    Result = urEnqueueMemBufferWrite(
        hQueue, hMem, true, Map->MapOffset, Map->MapSize, pMappedPtr,
        numEventsInWaitList, phEventWaitList, phEvent);
  } else {
    if (IsPinned) {
      // TODO: Ignore the event waits list for now. When urEnqueueEventsWait is
      // implemented we can call it on the wait list.
    }

    if (phEvent) {
      enqueueNoOp(UR_COMMAND_MEM_UNMAP, hQueue, phEvent);
    }
  }
  BufferImpl.unmap(pMappedPtr);

  return Result;
}
