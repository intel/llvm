//===--------- async_alloc.cpp - Level Zero Adapter -----------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "context.hpp"
#include "enqueued_pool.hpp"
#include "event.hpp"

#include "logger/ur_logger.hpp"

#include <umf_helpers.hpp>
#include <ur_api.h>

namespace ur::level_zero {

static ur_result_t enqueueUSMAllocHelper(
    ur_queue_handle_t Queue, ur_usm_pool_handle_t Pool, const size_t Size,
    const ur_exp_async_usm_alloc_properties_t *, uint32_t NumEventsInWaitList,
    const ur_event_handle_t *EventWaitList, void **RetMem,
    ur_event_handle_t *OutEvent, ur_usm_type_t Type) {

  std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

  // Allocate USM memory
  ur_usm_pool_handle_t USMPool = nullptr;
  if (Pool) {
    USMPool = Pool;
  } else {
    USMPool = &Queue->Context->AsyncPool;
  }

  auto Device = (Type == UR_USM_TYPE_HOST) ? nullptr : Queue->Device;

  std::vector<ur_event_handle_t> ExtEventWaitList;
  ur_event_handle_t OriginAllocEvent = nullptr;
  auto AsyncAlloc =
      USMPool->allocateEnqueued(Queue, Device, nullptr, Type, Size);
  if (!AsyncAlloc) {
    auto Ret =
        USMPool->allocate(Queue->Context, Device, nullptr, Type, Size, RetMem);
    if (Ret) {
      return Ret;
    }
  } else {
    *RetMem = std::get<0>(*AsyncAlloc);
    OriginAllocEvent = std::get<1>(*AsyncAlloc);
    if (OriginAllocEvent) {
      for (size_t I = 0; I < NumEventsInWaitList; ++I) {
        ExtEventWaitList.push_back(EventWaitList[I]);
      }
      ExtEventWaitList.push_back(OriginAllocEvent);
    }
  }

  if (!ExtEventWaitList.empty()) {
    NumEventsInWaitList = ExtEventWaitList.size();
    EventWaitList = ExtEventWaitList.data();
  }

  bool UseCopyEngine = false;
  _ur_ze_event_list_t TmpWaitList;
  UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
      NumEventsInWaitList, EventWaitList, Queue, UseCopyEngine));

  bool OkToBatch = true;
  // Get a new command list to be used on this call
  ur_command_list_ptr_t CommandList{};
  UR_CALL(Queue->Context->getAvailableCommandList(
      Queue, CommandList, UseCopyEngine, NumEventsInWaitList, EventWaitList,
      OkToBatch, nullptr /*ForcedCmdQueue*/));

  ze_event_handle_t ZeEvent = nullptr;
  ur_event_handle_t InternalEvent{};
  bool IsInternal = OutEvent == nullptr;
  ur_event_handle_t *Event = OutEvent ? OutEvent : &InternalEvent;

  ur_command_t CommandType = UR_COMMAND_FORCE_UINT32;
  switch (Type) {
  case UR_USM_TYPE_HOST:
    CommandType = UR_COMMAND_ENQUEUE_USM_HOST_ALLOC_EXP;
    break;
  case UR_USM_TYPE_DEVICE:
    CommandType = UR_COMMAND_ENQUEUE_USM_DEVICE_ALLOC_EXP;
    break;
  case UR_USM_TYPE_SHARED:
    CommandType = UR_COMMAND_ENQUEUE_USM_SHARED_ALLOC_EXP;
    break;
  default:
    logger::error("enqueueUSMAllocHelper: unsupported USM type");
    throw UR_RESULT_ERROR_UNKNOWN;
  }
  UR_CALL(createEventAndAssociateQueue(Queue, Event, CommandType, CommandList,
                                       IsInternal, false));
  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;
  (*Event)->OriginAllocEvent = OriginAllocEvent;

  const auto &ZeCommandList = CommandList->first;
  const auto &WaitList = (*Event)->WaitList;
  if (WaitList.Length) {
    ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
               (ZeCommandList, WaitList.Length, WaitList.ZeEventList));
  }

  // Signal that USM allocation event was finished
  ZE2UR_CALL(zeCommandListAppendSignalEvent, (CommandList->first, ZeEvent));

  UR_CALL(Queue->executeCommandList(CommandList, false, OkToBatch));

  return UR_RESULT_SUCCESS;
}

ur_result_t urEnqueueUSMDeviceAllocExp(
    ur_queue_handle_t Queue,   ///< [in] handle of the queue object
    ur_usm_pool_handle_t Pool, ///< [in][optional] USM pool descriptor
    const size_t Size, ///< [in] minimum size in bytes of the USM memory object
                       ///< to be allocated
    const ur_exp_async_usm_alloc_properties_t
        *Properties, ///< [in][optional] pointer to the enqueue asynchronous
                     ///< USM allocation properties
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< pointer to a list of events that must be complete
                        ///< before the kernel execution. If nullptr, the
                        ///< numEventsInWaitList must be 0, indicating no wait
                        ///< events.
    void **Mem,         ///< [out] pointer to USM memory object
    ur_event_handle_t *OutEvent ///< [out][optional] return an event object that
                                ///< identifies the async alloc
) {
  return enqueueUSMAllocHelper(Queue, Pool, Size, Properties,
                               NumEventsInWaitList, EventWaitList, Mem,
                               OutEvent, UR_USM_TYPE_DEVICE);
}

ur_result_t urEnqueueUSMSharedAllocExp(
    ur_queue_handle_t Queue,   ///< [in] handle of the queue object
    ur_usm_pool_handle_t Pool, ///< [in][optional] USM pool descriptor
    const size_t Size, ///< [in] minimum size in bytes of the USM memory object
                       ///< to be allocated
    const ur_exp_async_usm_alloc_properties_t
        *Properties, ///< [in][optional] pointer to the enqueue asynchronous
                     ///< USM allocation properties
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< pointer to a list of events that must be complete
                        ///< before the kernel execution. If nullptr, the
                        ///< numEventsInWaitList must be 0, indicating no wait
                        ///< events.
    void **Mem,         ///< [out] pointer to USM memory object
    ur_event_handle_t *OutEvent ///< [out][optional] return an event object that
                                ///< identifies the async alloc
) {
  return enqueueUSMAllocHelper(Queue, Pool, Size, Properties,
                               NumEventsInWaitList, EventWaitList, Mem,
                               OutEvent, UR_USM_TYPE_SHARED);
}

ur_result_t urEnqueueUSMHostAllocExp(
    ur_queue_handle_t Queue,   ///< [in] handle of the queue object
    ur_usm_pool_handle_t Pool, ///< [in][optional] handle of the USM memory pool
    const size_t Size, ///< [in] minimum size in bytes of the USM memory object
                       ///< to be allocated
    const ur_exp_async_usm_alloc_properties_t
        *Properties, ///< [in][optional] pointer to the enqueue asynchronous
                     ///< USM allocation properties
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< pointer to a list of events that must be complete
                        ///< before the kernel execution. If nullptr, the
                        ///< numEventsInWaitList must be 0, indicating no wait
                        ///< events.
    void **Mem,         ///< [out] pointer to USM memory object
    ur_event_handle_t
        *OutEvent ///< [out][optional] return an event object that identifies
                  ///< the asynchronous USM device allocation
) {
  return enqueueUSMAllocHelper(Queue, Pool, Size, Properties,
                               NumEventsInWaitList, EventWaitList, Mem,
                               OutEvent, UR_USM_TYPE_HOST);
}

ur_result_t urEnqueueUSMFreeExp(
    ur_queue_handle_t Queue,      ///< [in] handle of the queue object
    ur_usm_pool_handle_t,         ///< [in][optional] USM pool descriptor
    void *Mem,                    ///< [in] pointer to USM memory object
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< pointer to a list of events that must be complete
                        ///< before the kernel execution. If nullptr, the
                        ///< numEventsInWaitList must be 0, indicating no wait
                        ///< events.
    ur_event_handle_t *OutEvent ///< [out][optional] return an event object that
                                ///< identifies the async alloc
) {
  std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

  bool UseCopyEngine = false;
  _ur_ze_event_list_t TmpWaitList;
  UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
      NumEventsInWaitList, EventWaitList, Queue, UseCopyEngine));

  bool OkToBatch = false;
  // Get a new command list to be used on this call
  ur_command_list_ptr_t CommandList{};
  UR_CALL(Queue->Context->getAvailableCommandList(
      Queue, CommandList, UseCopyEngine, NumEventsInWaitList, EventWaitList,
      OkToBatch, nullptr /*ForcedCmdQueue*/));

  ze_event_handle_t ZeEvent = nullptr;
  ur_event_handle_t InternalEvent{};
  bool IsInternal = OutEvent == nullptr;
  ur_event_handle_t *Event = OutEvent ? OutEvent : &InternalEvent;

  UR_CALL(createEventAndAssociateQueue(Queue, Event,
                                       UR_COMMAND_ENQUEUE_USM_FREE_EXP,
                                       CommandList, IsInternal, false));
  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

  const auto &ZeCommandList = CommandList->first;
  const auto &WaitList = (*Event)->WaitList;
  if (WaitList.Length) {
    ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
               (ZeCommandList, WaitList.Length, WaitList.ZeEventList));
  }

  auto hPool = umfPoolByPtr(Mem);
  if (!hPool) {
    return USMFreeHelper(Queue->Context, Mem);
  }

  UsmPool *usmPool = nullptr;
  auto ret = umfPoolGetTag(hPool, (void **)&usmPool);
  if (ret != UMF_RESULT_SUCCESS || usmPool == nullptr) {
    return USMFreeHelper(Queue->Context, Mem);
  }

  size_t size = umfPoolMallocUsableSize(hPool, Mem);
  usmPool->AsyncPool.insert(Mem, size, *Event, Queue);

  // Signal that USM free event was finished
  ZE2UR_CALL(zeCommandListAppendSignalEvent, (ZeCommandList, ZeEvent));

  UR_CALL(Queue->executeCommandList(CommandList, false, OkToBatch));

  return UR_RESULT_SUCCESS;
}
} // namespace ur::level_zero
