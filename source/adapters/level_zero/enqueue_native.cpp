//===--------- enqueue_native.cpp - LevelZero Adapter ---------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <ur_api.h>

#include "logger/ur_logger.hpp"
#include "queue.hpp"
#include "ur_level_zero.hpp"

ur_result_t ur_queue_handle_legacy_t_::enqueueNativeCommandExp(
    ur_exp_enqueue_native_command_function_t pfnNativeEnqueue, void *data,
    uint32_t, const ur_mem_handle_t *,
    const ur_exp_enqueue_native_command_properties_t *,
    uint32_t NumEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  auto Queue = this;

  // TODO: Do I need this lock?
  std::scoped_lock<ur_shared_mutex> Lock(Queue->Mutex);

  // TODO: What do I need to do with phMemList? Will a ur_mem_handle_t always
  // be usable as a native arg from within pfnNativeEnqueue, or should some
  // mem migration happen?

  bool UseCopyEngine = false;
  _ur_ze_event_list_t TmpWaitList;
  UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
      NumEventsInWaitList, phEventWaitList, Queue, UseCopyEngine));

  // Get a new command list to be used on this call
  ur_command_list_ptr_t CommandList{};
  UR_CALL(Queue->Context->getAvailableCommandList(
      Queue, CommandList, UseCopyEngine, NumEventsInWaitList, phEventWaitList,
      true /* AllowBatching */));

  ze_event_handle_t ZeEvent = nullptr;
  ur_event_handle_t InternalEvent{};
  bool IsInternal = phEvent == nullptr;
  ur_event_handle_t *Event = phEvent ? phEvent : &InternalEvent;

  UR_CALL(createEventAndAssociateQueue(Queue, Event,
                                       UR_COMMAND_ENQUEUE_NATIVE_EXP,
                                       CommandList, IsInternal, false));
  UR_CALL(setSignalEvent(Queue, UseCopyEngine, &ZeEvent, Event,
                         NumEventsInWaitList, phEventWaitList,
                         CommandList->second.ZeQueue));
  (*Event)->WaitList = TmpWaitList;

  // FIXME: blocking synchronization. Make this faster
  Queue->queueFinish();

  // Execute interop func
  pfnNativeEnqueue(Queue, data);

  // FIXME: blocking synchronization. Make this faster
  Queue->queueFinish();

  return UR_RESULT_SUCCESS;
}
