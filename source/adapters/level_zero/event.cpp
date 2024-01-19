//===--------- event.cpp - Level Zero Adapter -----------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <climits>
#include <mutex>
#include <string.h>

#include "command_buffer.hpp"
#include "common.hpp"
#include "event.hpp"
#include "ur_level_zero.hpp"

void printZeEventList(const _ur_ze_event_list_t &UrZeEventList) {
  if (UrL0Debug & UR_L0_DEBUG_BASIC) {
    urPrint("  NumEventsInWaitList %d:", UrZeEventList.Length);

    for (uint32_t I = 0; I < UrZeEventList.Length; I++) {
      urPrint(" %#llx", ur_cast<std::uintptr_t>(UrZeEventList.ZeEventList[I]));
    }

    urPrint("\n");
  }
}

// This is an experimental option that allows the use of multiple command lists
// when submitting barriers. The default is 0.
static const bool UseMultipleCmdlistBarriers = [] {
  const char *UrRet = std::getenv("UR_L0_USE_MULTIPLE_COMMANDLIST_BARRIERS");
  const char *PiRet =
      std::getenv("SYCL_PI_LEVEL_ZERO_USE_MULTIPLE_COMMANDLIST_BARRIERS");
  const char *UseMultipleCmdlistBarriersFlag =
      UrRet ? UrRet : (PiRet ? PiRet : nullptr);
  if (!UseMultipleCmdlistBarriersFlag)
    return true;
  return std::atoi(UseMultipleCmdlistBarriersFlag) > 0;
}();

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueEventsWait(
    ur_queue_handle_t Queue,      ///< [in] handle of the queue object
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< pointer to a list of events that must be complete
                        ///< before this command can be executed. If nullptr,
                        ///< the numEventsInWaitList must be 0, indicating that
                        ///< all previously enqueued commands must be complete.
    ur_event_handle_t
        *OutEvent ///< [in,out][optional] return an event object that identifies
                  ///< this particular command instance.
) {
  if (EventWaitList) {
    bool UseCopyEngine = false;

    // Lock automatically releases when this goes out of scope.
    std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

    _ur_ze_event_list_t TmpWaitList = {};
    UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
        NumEventsInWaitList, EventWaitList, Queue, UseCopyEngine));

    // Get a new command list to be used on this call
    ur_command_list_ptr_t CommandList{};
    UR_CALL(Queue->Context->getAvailableCommandList(Queue, CommandList,
                                                    UseCopyEngine));

    ze_event_handle_t ZeEvent = nullptr;
    ur_event_handle_t InternalEvent;
    bool IsInternal = OutEvent == nullptr;
    ur_event_handle_t *Event = OutEvent ? OutEvent : &InternalEvent;
    UR_CALL(createEventAndAssociateQueue(Queue, Event, UR_COMMAND_EVENTS_WAIT,
                                         CommandList, IsInternal));

    ZeEvent = (*Event)->ZeEvent;
    (*Event)->WaitList = TmpWaitList;

    const auto &WaitList = (*Event)->WaitList;
    auto ZeCommandList = CommandList->first;
    ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
               (ZeCommandList, WaitList.Length, WaitList.ZeEventList));

    ZE2UR_CALL(zeCommandListAppendSignalEvent, (ZeCommandList, ZeEvent));

    // Execute command list asynchronously as the event will be used
    // to track down its completion.
    return Queue->executeCommandList(CommandList);
  }

  {
    // If wait-list is empty, then this particular command should wait until
    // all previous enqueued commands to the command-queue have completed.
    //
    // TODO: find a way to do that without blocking the host.

    // Lock automatically releases when this goes out of scope.
    std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

    if (OutEvent) {
      UR_CALL(createEventAndAssociateQueue(
          Queue, OutEvent, UR_COMMAND_EVENTS_WAIT, Queue->CommandListMap.end(),
          /* IsInternal */ false));
    }

    Queue->synchronize();

    if (OutEvent) {
      Queue->LastCommandEvent = reinterpret_cast<ur_event_handle_t>(*OutEvent);

      ZE2UR_CALL(zeEventHostSignal, ((*OutEvent)->ZeEvent));
      (*OutEvent)->Completed = true;
    }
  }

  if (!Queue->UsingImmCmdLists) {
    std::unique_lock<ur_shared_mutex> Lock(Queue->Mutex);
    resetCommandLists(Queue);
  }

  return UR_RESULT_SUCCESS;
}

// Control if wait with barrier is implemented by signal of an event
// as opposed by true barrier command for in-order queue.
static const bool InOrderBarrierBySignal = [] {
  const char *UrRet = std::getenv("UR_L0_IN_ORDER_BARRIER_BY_SIGNAL");
  return (UrRet ? std::atoi(UrRet) : true);
}();

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueEventsWaitWithBarrier(
    ur_queue_handle_t Queue,      ///< [in] handle of the queue object
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< pointer to a list of events that must be complete
                        ///< before this command can be executed. If nullptr,
                        ///< the numEventsInWaitList must be 0, indicating that
                        ///< all previously enqueued commands must be complete.
    ur_event_handle_t
        *OutEvent ///< [in,out][optional] return an event object that identifies
                  ///< this particular command instance.
) {

  // Lock automatically releases when this goes out of scope.
  std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

  // Helper function for appending a barrier to a command list.
  auto insertBarrierIntoCmdList =
      [&Queue](ur_command_list_ptr_t CmdList,
               const _ur_ze_event_list_t &EventWaitList,
               ur_event_handle_t &Event, bool IsInternal) {
        UR_CALL(createEventAndAssociateQueue(
            Queue, &Event, UR_COMMAND_EVENTS_WAIT_WITH_BARRIER, CmdList,
            IsInternal));

        Event->WaitList = EventWaitList;

        // For in-order queue we don't need a real barrier, just wait for
        // requested events in potentially different queues and add a "barrier"
        // event signal because it is already guaranteed that previous commands
        // in this queue are completed when the signal is started.
        //
        // Only consideration here is that when profiling is used, signalEvent
        // cannot be used if EventWaitList.Lenght == 0. In those cases, we need
        // to fallback directly to barrier to have correct timestamps. See here:
        // https://spec.oneapi.io/level-zero/latest/core/api.html?highlight=appendsignalevent#_CPPv430zeCommandListAppendSignalEvent24ze_command_list_handle_t17ze_event_handle_t
        //
        // TODO: this and other special handling of in-order queues to be
        // updated when/if Level Zero adds native support for in-order queues.
        //
        if (Queue->isInOrderQueue() && InOrderBarrierBySignal &&
            !Queue->isProfilingEnabled()) {
          if (EventWaitList.Length) {
            ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
                       (CmdList->first, EventWaitList.Length,
                        EventWaitList.ZeEventList));
          }
          ZE2UR_CALL(zeCommandListAppendSignalEvent,
                     (CmdList->first, Event->ZeEvent));
        } else {
          ZE2UR_CALL(zeCommandListAppendBarrier,
                     (CmdList->first, Event->ZeEvent, EventWaitList.Length,
                      EventWaitList.ZeEventList));
        }

        return UR_RESULT_SUCCESS;
      };

  // If the queue is in-order then each command in it effectively acts as a
  // barrier, so we don't need to do anything except if we were requested
  // a "barrier" event to be created. Or if we need to wait for events in
  // potentially different queues.
  //
  if (Queue->isInOrderQueue() && NumEventsInWaitList == 0 && !OutEvent) {
    return UR_RESULT_SUCCESS;
  }

  ur_event_handle_t InternalEvent;
  bool IsInternal = OutEvent == nullptr;
  ur_event_handle_t *Event = OutEvent ? OutEvent : &InternalEvent;

  auto WaitListEmptyOrAllEventsFromSameQueue = [Queue, NumEventsInWaitList,
                                                EventWaitList]() {
    if (!NumEventsInWaitList)
      return true;

    for (uint32_t I = 0; I < NumEventsInWaitList; ++I)
      if (Queue != EventWaitList[I]->UrQueue)
        return false;

    return true;
  };

  // For in-order queue and wait-list which is empty or has events from
  // the same queue just use the last command event as the barrier event.
  if (Queue->isInOrderQueue() && WaitListEmptyOrAllEventsFromSameQueue() &&
      Queue->LastCommandEvent && !Queue->LastCommandEvent->IsDiscarded) {
    UR_CALL(urEventRetain(Queue->LastCommandEvent));
    *Event = Queue->LastCommandEvent;
    return UR_RESULT_SUCCESS;
  }

  // Indicator for whether batching is allowed. This may be changed later in
  // this function, but allow it by default.
  bool OkToBatch = true;

  // If we have a list of events to make the barrier from, then we can create a
  // barrier on these and use the resulting event as our future barrier.
  // We use the same approach if
  // UR_L0_USE_MULTIPLE_COMMANDLIST_BARRIERS is not set to a
  // positive value.
  // We use the same approach if we have in-order queue because every command
  // depends on previous one, so we don't need to insert barrier to multiple
  // command lists.
  if (NumEventsInWaitList || !UseMultipleCmdlistBarriers ||
      Queue->isInOrderQueue()) {
    // Retain the events as they will be owned by the result event.
    _ur_ze_event_list_t TmpWaitList;
    UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
        NumEventsInWaitList, EventWaitList, Queue, false /*UseCopyEngine=*/));

    // Get an arbitrary command-list in the queue.
    ur_command_list_ptr_t CmdList;
    UR_CALL(Queue->Context->getAvailableCommandList(
        Queue, CmdList, false /*UseCopyEngine=*/, OkToBatch));

    // Insert the barrier into the command-list and execute.
    UR_CALL(insertBarrierIntoCmdList(CmdList, TmpWaitList, *Event, IsInternal));

    UR_CALL(Queue->executeCommandList(CmdList, false, OkToBatch));

    // Because of the dependency between commands in the in-order queue we don't
    // need to keep track of any active barriers if we have in-order queue.
    if (UseMultipleCmdlistBarriers && !Queue->isInOrderQueue()) {
      auto UREvent = reinterpret_cast<ur_event_handle_t>(*Event);
      Queue->ActiveBarriers.add(UREvent);
    }
    return UR_RESULT_SUCCESS;
  }

  // Since there are no events to explicitly create a barrier for, we are
  // inserting a queue-wide barrier.

  // Command list(s) for putting barriers.
  std::vector<ur_command_list_ptr_t> CmdLists;

  // There must be at least one L0 queue.
  auto &ComputeGroup = Queue->ComputeQueueGroupsByTID.get();
  auto &CopyGroup = Queue->CopyQueueGroupsByTID.get();
  UR_ASSERT(!ComputeGroup.ZeQueues.empty() || !CopyGroup.ZeQueues.empty(),
            UR_RESULT_ERROR_INVALID_QUEUE);

  size_t NumQueues = 0;
  for (auto &QueueMap :
       {Queue->ComputeQueueGroupsByTID, Queue->CopyQueueGroupsByTID})
    for (auto &QueueGroup : QueueMap)
      NumQueues += QueueGroup.second.ZeQueues.size();

  OkToBatch = true;
  // Get an available command list tied to each command queue. We need
  // these so a queue-wide barrier can be inserted into each command
  // queue.
  CmdLists.reserve(NumQueues);
  for (auto &QueueMap :
       {Queue->ComputeQueueGroupsByTID, Queue->CopyQueueGroupsByTID})
    for (auto &QueueGroup : QueueMap) {
      bool UseCopyEngine =
          QueueGroup.second.Type != ur_queue_handle_t_::queue_type::Compute;
      if (Queue->UsingImmCmdLists) {
        // If immediate command lists are being used, each will act as their own
        // queue, so we must insert a barrier into each.
        for (auto &ImmCmdList : QueueGroup.second.ImmCmdLists)
          if (ImmCmdList != Queue->CommandListMap.end())
            CmdLists.push_back(ImmCmdList);
      } else {
        for (auto ZeQueue : QueueGroup.second.ZeQueues) {
          if (ZeQueue) {
            ur_command_list_ptr_t CmdList;
            UR_CALL(Queue->Context->getAvailableCommandList(
                Queue, CmdList, UseCopyEngine, OkToBatch, &ZeQueue));
            CmdLists.push_back(CmdList);
          }
        }
      }
    }

  // If no activity has occurred on the queue then there will be no cmdlists.
  // We need one for generating an Event, so create one.
  if (CmdLists.size() == 0) {
    // Get any available command list.
    ur_command_list_ptr_t CmdList;
    UR_CALL(Queue->Context->getAvailableCommandList(
        Queue, CmdList, false /*UseCopyEngine=*/, OkToBatch));
    CmdLists.push_back(CmdList);
  }

  if (CmdLists.size() > 1) {
    // Insert a barrier into each unique command queue using the available
    // command-lists.
    std::vector<ur_event_handle_t> EventWaitVector(CmdLists.size());
    for (size_t I = 0; I < CmdLists.size(); ++I) {
      UR_CALL(insertBarrierIntoCmdList(CmdLists[I], _ur_ze_event_list_t{},
                                       EventWaitVector[I],
                                       true /*IsInternal*/));
    }
    // If there were multiple queues we need to create a "convergence" event to
    // be our active barrier. This convergence event is signalled by a barrier
    // on all the events from the barriers we have inserted into each queue.
    // Use the first command list as our convergence command list.
    ur_command_list_ptr_t &ConvergenceCmdList = CmdLists[0];

    // Create an event list. It will take ownership over all relevant events so
    // we relinquish ownership and let it keep all events it needs.
    _ur_ze_event_list_t BaseWaitList;
    UR_CALL(BaseWaitList.createAndRetainUrZeEventList(
        EventWaitVector.size(),
        reinterpret_cast<const ur_event_handle_t *>(EventWaitVector.data()),
        Queue, ConvergenceCmdList->second.isCopy(Queue)));

    // Insert a barrier with the events from each command-queue into the
    // convergence command list. The resulting event signals the convergence of
    // all barriers.
    UR_CALL(insertBarrierIntoCmdList(ConvergenceCmdList, BaseWaitList, *Event,
                                     IsInternal));
  } else {
    // If there is only a single queue then insert a barrier and the single
    // result event can be used as our active barrier and used as the return
    // event. Take into account whether output event is discarded or not.
    UR_CALL(insertBarrierIntoCmdList(CmdLists[0], _ur_ze_event_list_t{}, *Event,
                                     IsInternal));
  }

  // Execute each command list so the barriers can be encountered.
  for (ur_command_list_ptr_t &CmdList : CmdLists)
    UR_CALL(Queue->executeCommandList(CmdList, false, OkToBatch));

  UR_CALL(Queue->ActiveBarriers.clear());
  auto UREvent = reinterpret_cast<ur_event_handle_t>(*Event);
  Queue->ActiveBarriers.add(UREvent);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventGetInfo(
    ur_event_handle_t Event,  ///< [in] handle of the event object
    ur_event_info_t PropName, ///< [in] the name of the event property to query
    size_t PropValueSize, ///< [in] size in bytes of the event property value
    void *PropValue,      ///< [out][optional] value of the event property
    size_t
        *PropValueSizeRet ///< [out][optional] bytes returned in event property
) {
  UrReturnHelper ReturnValue(PropValueSize, PropValue, PropValueSizeRet);

  switch (PropName) {
  case UR_EVENT_INFO_COMMAND_QUEUE: {
    std::shared_lock<ur_shared_mutex> EventLock(Event->Mutex);
    return ReturnValue(ur_queue_handle_t{Event->UrQueue});
  }
  case UR_EVENT_INFO_CONTEXT: {
    std::shared_lock<ur_shared_mutex> EventLock(Event->Mutex);
    return ReturnValue(ur_context_handle_t{Event->Context});
  }
  case UR_EVENT_INFO_COMMAND_TYPE: {
    std::shared_lock<ur_shared_mutex> EventLock(Event->Mutex);
    return ReturnValue(ur_cast<uint64_t>(Event->CommandType));
  }
  case UR_EVENT_INFO_COMMAND_EXECUTION_STATUS: {
    // Check to see if the event's Queue has an open command list due to
    // batching. If so, go ahead and close and submit it, because it is
    // possible that this is trying to query some event's status that
    // is part of the batch.  This isn't strictly required, but it seems
    // like a reasonable thing to do.
    auto UrQueue = Event->UrQueue;
    if (UrQueue) {
      // Lock automatically releases when this goes out of scope.
      std::scoped_lock<ur_shared_mutex> lock(UrQueue->Mutex);
      const auto &OpenCommandList = UrQueue->eventOpenCommandList(Event);
      if (OpenCommandList != UrQueue->CommandListMap.end()) {
        UR_CALL(UrQueue->executeOpenCommandList(
            OpenCommandList->second.isCopy(UrQueue)));
      }
    }

    // Level Zero has a much more explicit notion of command submission than
    // OpenCL. It doesn't happen unless the user submits a command list. We've
    // done it just above so the status is at least PI_EVENT_SUBMITTED.
    //
    // NOTE: We currently cannot tell if command is currently running, so
    // it will always show up "submitted" before it is finally "completed".
    //
    uint32_t Result = ur_cast<uint32_t>(UR_EVENT_STATUS_SUBMITTED);

    // Make sure that we query a host-visible event only.
    // If one wasn't yet created then don't create it here as well, and
    // just conservatively return that event is not yet completed.
    std::shared_lock<ur_shared_mutex> EventLock(Event->Mutex);
    auto HostVisibleEvent = Event->HostVisibleEvent;
    if (Event->Completed) {
      Result = UR_EVENT_STATUS_COMPLETE;
    } else if (HostVisibleEvent) {
      ze_result_t ZeResult;
      ZeResult =
          ZE_CALL_NOCHECK(zeEventQueryStatus, (HostVisibleEvent->ZeEvent));
      if (ZeResult == ZE_RESULT_SUCCESS) {
        Result = UR_EVENT_STATUS_COMPLETE;
      }
    }
    return ReturnValue(Result);
  }
  case UR_EVENT_INFO_REFERENCE_COUNT: {
    return ReturnValue(Event->RefCount.load());
  }
  default:
    urPrint("Unsupported ParamName in urEventGetInfo: ParamName=%d(%x)\n",
            PropName, PropName);
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventGetProfilingInfo(
    ur_event_handle_t Event, ///< [in] handle of the event object
    ur_profiling_info_t
        PropName, ///< [in] the name of the profiling property to query
    size_t
        PropValueSize, ///< [in] size in bytes of the profiling property value
    void *PropValue,   ///< [out][optional] value of the profiling property
    size_t *PropValueSizeRet ///< [out][optional] pointer to the actual size in
                             ///< bytes returned in propValue
) {
  std::shared_lock<ur_shared_mutex> EventLock(Event->Mutex);

  if (Event->UrQueue &&
      (Event->UrQueue->Properties & UR_QUEUE_FLAG_PROFILING_ENABLE) == 0) {
    return UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE;
  }

  ur_device_handle_t Device =
      Event->UrQueue ? Event->UrQueue->Device : Event->Context->Devices[0];

  uint64_t ZeTimerResolution = Device->ZeDeviceProperties->timerResolution;
  const uint64_t TimestampMaxValue =
      ((1ULL << Device->ZeDeviceProperties->kernelTimestampValidBits) - 1ULL);

  UrReturnHelper ReturnValue(PropValueSize, PropValue, PropValueSizeRet);

  ze_kernel_timestamp_result_t tsResult;

  // A Command-buffer consists of three command-lists for which only a single
  // event is returned to users. The actual profiling information related to the
  // command-buffer should therefore be extrated from graph events themsleves.
  // The timestamps of these events are saved in a memory region attached to
  // event usning CommandData field. The timings must therefore be recovered
  // from this memory.
  if (Event->CommandType == UR_COMMAND_COMMAND_BUFFER_ENQUEUE_EXP) {
    if (Event->CommandData) {
      command_buffer_profiling_t *ProfilingsPtr;
      switch (PropName) {
      case UR_PROFILING_INFO_COMMAND_START: {
        ProfilingsPtr =
            static_cast<command_buffer_profiling_t *>(Event->CommandData);
        // Sync-point order does not necessarily match to the order of
        // execution. We therefore look for the first command executed.
        uint64_t MinStart = ProfilingsPtr->Timestamps[0].global.kernelStart;
        for (uint64_t i = 1; i < ProfilingsPtr->NumEvents; i++) {
          uint64_t Timestamp = ProfilingsPtr->Timestamps[i].global.kernelStart;
          if (Timestamp < MinStart) {
            MinStart = Timestamp;
          }
        }
        uint64_t ContextStartTime =
            (MinStart & TimestampMaxValue) * ZeTimerResolution;
        return ReturnValue(ContextStartTime);
      }
      case UR_PROFILING_INFO_COMMAND_END: {
        ProfilingsPtr =
            static_cast<command_buffer_profiling_t *>(Event->CommandData);
        // Sync-point order does not necessarily match to the order of
        // execution. We therefore look for the last command executed.
        uint64_t MaxEnd = ProfilingsPtr->Timestamps[0].global.kernelEnd;
        uint64_t LastStart = ProfilingsPtr->Timestamps[0].global.kernelStart;
        for (uint64_t i = 1; i < ProfilingsPtr->NumEvents; i++) {
          uint64_t Timestamp = ProfilingsPtr->Timestamps[i].global.kernelEnd;
          if (Timestamp > MaxEnd) {
            MaxEnd = Timestamp;
            LastStart = ProfilingsPtr->Timestamps[i].global.kernelStart;
          }
        }
        uint64_t ContextStartTime = (LastStart & TimestampMaxValue);
        uint64_t ContextEndTime = (MaxEnd & TimestampMaxValue);

        //
        // Handle a possible wrap-around (the underlying HW counter is <
        // 64-bit). Note, it will not report correct time if there were multiple
        // wrap arounds, and the longer term plan is to enlarge the capacity of
        // the HW timestamps.
        //
        if (ContextEndTime <= ContextStartTime) {
          ContextEndTime += TimestampMaxValue;
        }
        ContextEndTime *= ZeTimerResolution;
        return ReturnValue(ContextEndTime);
      }
      default:
        urPrint("urEventGetProfilingInfo: not supported ParamName\n");
        return UR_RESULT_ERROR_INVALID_VALUE;
      }
    } else {
      return UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE;
    }
  }

  switch (PropName) {
  case UR_PROFILING_INFO_COMMAND_START: {
    ZE2UR_CALL(zeEventQueryKernelTimestamp, (Event->ZeEvent, &tsResult));
    uint64_t ContextStartTime =
        (tsResult.global.kernelStart & TimestampMaxValue) * ZeTimerResolution;
    return ReturnValue(ContextStartTime);
  }
  case UR_PROFILING_INFO_COMMAND_END: {
    ZE2UR_CALL(zeEventQueryKernelTimestamp, (Event->ZeEvent, &tsResult));

    uint64_t ContextStartTime =
        (tsResult.global.kernelStart & TimestampMaxValue);
    uint64_t ContextEndTime = (tsResult.global.kernelEnd & TimestampMaxValue);

    //
    // Handle a possible wrap-around (the underlying HW counter is < 64-bit).
    // Note, it will not report correct time if there were multiple wrap
    // arounds, and the longer term plan is to enlarge the capacity of the
    // HW timestamps.
    //
    if (ContextEndTime <= ContextStartTime) {
      ContextEndTime += TimestampMaxValue;
    }
    ContextEndTime *= ZeTimerResolution;
    return ReturnValue(ContextEndTime);
  }
  case UR_PROFILING_INFO_COMMAND_QUEUED:
  case UR_PROFILING_INFO_COMMAND_SUBMIT:
    // Note: No users for this case
    // The "command_submit" time is implemented by recording submission
    // timestamp with a call to urDeviceGetGlobalTimestamps before command
    // enqueue.
    //
    return ReturnValue(uint64_t{0});
  default:
    urPrint("urEventGetProfilingInfo: not supported ParamName\n");
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_event_handle_t_::getOrCreateHostVisibleEvent(
    ze_event_handle_t &ZeHostVisibleEvent) {

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(UrQueue->Mutex,
                                                          this->Mutex);

  if (!HostVisibleEvent) {
    if (UrQueue->ZeEventsScope != OnDemandHostVisibleProxy)
      die("getOrCreateHostVisibleEvent: missing host-visible event");

    // Submit the command(s) signalling the proxy event to the queue.
    // We have to first submit a wait for the device-only event for which this
    // proxy is created.
    //
    // Get a new command list to be used on this call

    // We want to batch these commands to avoid extra submissions (costly)
    bool OkToBatch = true;

    ur_command_list_ptr_t CommandList{};
    UR_CALL(UrQueue->Context->getAvailableCommandList(
        UrQueue, CommandList, false /* UseCopyEngine */, OkToBatch))

    // Create a "proxy" host-visible event.
    UR_CALL(createEventAndAssociateQueue(
        UrQueue, &HostVisibleEvent, UR_EXT_COMMAND_TYPE_USER, CommandList,
        /* IsInternal */ false, /* HostVisible */ true));

    ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
               (CommandList->first, 1, &ZeEvent));
    ZE2UR_CALL(zeCommandListAppendSignalEvent,
               (CommandList->first, HostVisibleEvent->ZeEvent));

    UR_CALL(UrQueue->executeCommandList(CommandList, false, OkToBatch))
  }

  ZeHostVisibleEvent = HostVisibleEvent->ZeEvent;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventWait(
    uint32_t NumEvents, ///< [in] number of events in the event list
    const ur_event_handle_t
        *EventWaitList ///< [in][range(0, numEvents)] pointer to a list of
                       ///< events to wait for completion
) {
  for (uint32_t I = 0; I < NumEvents; I++) {
    auto e = EventWaitList[I];
    if (e->UrQueue && e->UrQueue->ZeEventsScope == OnDemandHostVisibleProxy) {
      // Make sure to add all host-visible "proxy" event signals if needed.
      // This ensures that all signalling commands are submitted below and
      // thus proxy events can be waited without a deadlock.
      //
      ur_event_handle_t_ *Event = ur_cast<ur_event_handle_t_ *>(e);
      if (!Event->hasExternalRefs())
        die("urEventsWait must not be called for an internal event");

      ze_event_handle_t ZeHostVisibleEvent;
      if (auto Res = Event->getOrCreateHostVisibleEvent(ZeHostVisibleEvent))
        return Res;
    }
  }
  // Submit dependent open command lists for execution, if any
  for (uint32_t I = 0; I < NumEvents; I++) {
    ur_event_handle_t_ *Event = ur_cast<ur_event_handle_t_ *>(EventWaitList[I]);
    auto UrQueue = Event->UrQueue;
    if (UrQueue) {
      // Lock automatically releases when this goes out of scope.
      std::scoped_lock<ur_shared_mutex> lock(UrQueue->Mutex);

      UR_CALL(UrQueue->executeAllOpenCommandLists());
    }
  }
  std::unordered_set<ur_queue_handle_t> Queues;
  for (uint32_t I = 0; I < NumEvents; I++) {
    {
      ur_event_handle_t_ *Event =
          ur_cast<ur_event_handle_t_ *>(EventWaitList[I]);
      {
        std::shared_lock<ur_shared_mutex> EventLock(Event->Mutex);
        if (!Event->hasExternalRefs())
          die("urEventWait must not be called for an internal event");

        if (!Event->Completed) {
          auto HostVisibleEvent = Event->HostVisibleEvent;
          if (!HostVisibleEvent)
            die("The host-visible proxy event missing");

          ze_event_handle_t ZeEvent = HostVisibleEvent->ZeEvent;
          urPrint("ZeEvent = %#llx\n", ur_cast<std::uintptr_t>(ZeEvent));
          ZE2UR_CALL(zeHostSynchronize, (ZeEvent));
          Event->Completed = true;
        }
      }
      if (auto Q = Event->UrQueue) {
        if (Q->UsingImmCmdLists && Q->isInOrderQueue())
          // Use information about waited event to cleanup completed events in
          // the in-order queue.
          CleanupEventsInImmCmdLists(
              Event->UrQueue, false /* QueueLocked */, false /* QueueSynced */,
              reinterpret_cast<ur_event_handle_t>(Event));
        else {
          // NOTE: we are cleaning up after the event here to free resources
          // sooner in case run-time is not calling urEventRelease soon enough.
          CleanupCompletedEvent(reinterpret_cast<ur_event_handle_t>(Event));
          // For the case when we have out-of-order queue or regular command
          // lists its more efficient to check fences so put the queue in the
          // set to cleanup later.
          Queues.insert(Q);
        }
      }
    }
  }

  // We waited some events above, check queue for signaled command lists and
  // reset them.
  for (auto &Q : Queues) {
    std::unique_lock<ur_shared_mutex> Lock(Q->Mutex);
    resetCommandLists(Q);
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventRetain(
    ur_event_handle_t Event ///< [in] handle of the event object
) {
  Event->RefCountExternal++;
  Event->RefCount.increment();

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventRelease(
    ur_event_handle_t Event ///< [in] handle of the event object
) {
  Event->RefCountExternal--;
  UR_CALL(urEventReleaseInternal(Event));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventGetNativeHandle(
    ur_event_handle_t Event, ///< [in] handle of the event.
    ur_native_handle_t
        *NativeEvent ///< [out] a pointer to the native handle of the event.
) {
  {
    std::shared_lock<ur_shared_mutex> Lock(Event->Mutex);
    auto *ZeEvent = ur_cast<ze_event_handle_t *>(NativeEvent);
    *ZeEvent = Event->ZeEvent;
  }
  // Event can potentially be in an open command-list, make sure that
  // it is submitted for execution to avoid potential deadlock if
  // interop app is going to wait for it.
  auto Queue = Event->UrQueue;
  if (Queue) {
    std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);
    const auto &OpenCommandList = Queue->eventOpenCommandList(Event);
    if (OpenCommandList != Queue->CommandListMap.end()) {
      UR_CALL(
          Queue->executeOpenCommandList(OpenCommandList->second.isCopy(Queue)));
    }
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urExtEventCreate(
    ur_context_handle_t Context, ///< [in] handle of the context object
    ur_event_handle_t
        *Event ///< [out] pointer to the handle of the event object created.
) {
  UR_CALL(EventCreate(Context, nullptr, true, Event));

  (*Event)->RefCountExternal++;
  ZE2UR_CALL(zeEventHostSignal, ((*Event)->ZeEvent));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventCreateWithNativeHandle(
    ur_native_handle_t NativeEvent, ///< [in] the native handle of the event.
    ur_context_handle_t Context,    ///< [in] handle of the context object
    const ur_event_native_properties_t *Properties,
    ur_event_handle_t
        *Event ///< [out] pointer to the handle of the event object created.
) {

  // we dont have urEventCreate, so use this check for now to know that
  // the call comes from urEventCreate()
  if (NativeEvent == nullptr) {
    UR_CALL(EventCreate(Context, nullptr, true, Event));

    (*Event)->RefCountExternal++;
    ZE2UR_CALL(zeEventHostSignal, ((*Event)->ZeEvent));
    return UR_RESULT_SUCCESS;
  }

  auto ZeEvent = ur_cast<ze_event_handle_t>(NativeEvent);
  ur_event_handle_t_ *UREvent{};
  try {
    UREvent = new ur_event_handle_t_(ZeEvent, nullptr /* ZeEventPool */,
                                     Context, UR_EXT_COMMAND_TYPE_USER,
                                     Properties->isNativeHandleOwned);

    UREvent->RefCountExternal++;

  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  // Assume native event is host-visible, or otherwise we'd
  // need to create a host-visible proxy for it.
  UREvent->HostVisibleEvent = reinterpret_cast<ur_event_handle_t>(UREvent);

  // Unlike regular events managed by SYCL RT we don't have to wait for interop
  // events completion, and not need to do the their `cleanup()`. This in
  // particular guarantees that the extra `urEventRelease` is not called on
  // them. That release is needed to match the `urEventRetain` of regular events
  // made for waiting for event completion, but not this interop event.
  UREvent->CleanedUp = true;

  *Event = reinterpret_cast<ur_event_handle_t>(UREvent);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventSetCallback(
    ur_event_handle_t Event,        ///< [in] handle of the event object
    ur_execution_info_t ExecStatus, ///< [in] execution status of the event
    ur_event_callback_t Notify,     ///< [in] execution status of the event
    void *UserData ///< [in][out][optional] pointer to data to be passed to
                   ///< callback.
) {
  std::ignore = Event;
  std::ignore = ExecStatus;
  std::ignore = Notify;
  std::ignore = UserData;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urEventReleaseInternal(ur_event_handle_t Event) {
  if (!Event->RefCount.decrementAndTest())
    return UR_RESULT_SUCCESS;

  if (Event->CommandType == UR_COMMAND_MEM_UNMAP && Event->CommandData) {
    // Free the memory allocated in the urEnqueueMemBufferMap.
    if (auto Res = ZeMemFreeHelper(Event->Context, Event->CommandData))
      return Res;
    Event->CommandData = nullptr;
  }
  if (Event->CommandType == UR_COMMAND_COMMAND_BUFFER_ENQUEUE_EXP &&
      Event->CommandData) {
    // Free the memory extra event allocated for profiling purposed.
    command_buffer_profiling_t *ProfilingPtr =
        static_cast<command_buffer_profiling_t *>(Event->CommandData);
    delete[] ProfilingPtr->Timestamps;
    delete ProfilingPtr;
    Event->CommandData = nullptr;
  }
  if (Event->OwnNativeHandle) {
    if (DisableEventsCaching) {
      auto ZeResult = ZE_CALL_NOCHECK(zeEventDestroy, (Event->ZeEvent));
      // Gracefully handle the case that L0 was already unloaded.
      if (ZeResult && ZeResult != ZE_RESULT_ERROR_UNINITIALIZED)
        return ze2urResult(ZeResult);
      auto Context = Event->Context;
      if (auto Res = Context->decrementUnreleasedEventsInPool(Event))
        return Res;
    }
  }
  // It is possible that host-visible event was never created.
  // In case it was check if that's different from this same event
  // and release a reference to it.
  if (Event->HostVisibleEvent && Event->HostVisibleEvent != Event) {
    // Decrement ref-count of the host-visible proxy event.
    UR_CALL(urEventReleaseInternal(Event->HostVisibleEvent));
  }

  // Save pointer to the queue before deleting/resetting event.
  // When we add an event to the cache we need to check whether profiling is
  // enabled or not, so we access properties of the queue and that's why queue
  // must released later.
  auto Queue = Event->UrQueue;
  if (DisableEventsCaching || !Event->OwnNativeHandle) {
    delete Event;
  } else {
    Event->Context->addEventToContextCache(Event);
  }

  // We intentionally incremented the reference counter when an event is
  // created so that we can avoid ur_queue_handle_t is released before the
  // associated ur_event_handle_t is released. Here we have to decrement it so
  // ur_queue_handle_t can be released successfully.
  if (Queue) {
    UR_CALL(urQueueReleaseInternal(Queue));
  }

  return UR_RESULT_SUCCESS;
}

// Helper function to implement zeHostSynchronize.
// The behavior is to avoid infinite wait during host sync under ZE_DEBUG.
// This allows for a much more responsive debugging of hangs.
//
template <typename T, typename Func>
ze_result_t zeHostSynchronizeImpl(Func Api, T Handle) {
  if (!UrL0Debug) {
    return Api(Handle, UINT64_MAX);
  }

  ze_result_t R;
  while ((R = Api(Handle, 1000)) == ZE_RESULT_NOT_READY)
    ;
  return R;
}

// Template function to do various types of host synchronizations.
// This is intended to be used instead of direct calls to specific
// Level-Zero synchronization APIs.
//
template <typename T> ze_result_t zeHostSynchronize(T Handle);
template <> ze_result_t zeHostSynchronize(ze_event_handle_t Handle) {
  return zeHostSynchronizeImpl(zeEventHostSynchronize, Handle);
}
template <> ze_result_t zeHostSynchronize(ze_command_queue_handle_t Handle) {
  return zeHostSynchronizeImpl(zeCommandQueueSynchronize, Handle);
}

// Perform any necessary cleanup after an event has been signalled.
// This currently makes sure to release any kernel that may have been used by
// the event, updates the last command event in the queue and cleans up all dep
// events of the event.
// If the caller locks queue mutex then it must pass 'true' to QueueLocked.
ur_result_t CleanupCompletedEvent(ur_event_handle_t Event, bool QueueLocked,
                                  bool SetEventCompleted) {
  ur_kernel_handle_t AssociatedKernel = nullptr;
  // List of dependent events.
  std::list<ur_event_handle_t> EventsToBeReleased;
  ur_queue_handle_t AssociatedQueue = nullptr;
  {
    std::scoped_lock<ur_shared_mutex> EventLock(Event->Mutex);
    if (SetEventCompleted)
      Event->Completed = true;
    // Exit early of event was already cleanedup.
    if (Event->CleanedUp)
      return UR_RESULT_SUCCESS;

    AssociatedQueue = Event->UrQueue;

    // Remember the kernel associated with this event if there is one. We are
    // going to release it later.
    if (Event->CommandType == UR_COMMAND_KERNEL_LAUNCH && Event->CommandData) {
      AssociatedKernel =
          reinterpret_cast<ur_kernel_handle_t>(Event->CommandData);
      Event->CommandData = nullptr;
    }

    // Make a list of all the dependent events that must have signalled
    // because this event was dependent on them.
    Event->WaitList.collectEventsForReleaseAndDestroyUrZeEventList(
        EventsToBeReleased);

    Event->CleanedUp = true;
  }

  auto ReleaseIndirectMem = [](ur_kernel_handle_t Kernel) {
    if (IndirectAccessTrackingEnabled) {
      // urKernelRelease is called by CleanupCompletedEvent(Event) as soon as
      // kernel execution has finished. This is the place where we need to
      // release memory allocations. If kernel is not in use (not submitted by
      // some other thread) then release referenced memory allocations. As a
      // result, memory can be deallocated and context can be removed from
      // container in the platform. That's why we need to lock a mutex here.
      ur_platform_handle_t Plt = Kernel->Program->Context->getPlatform();
      std::scoped_lock<ur_shared_mutex> ContextsLock(Plt->ContextsMutex);

      if (--Kernel->SubmissionsCount == 0) {
        // Kernel is not submitted for execution, release referenced memory
        // allocations.
        for (auto &MemAlloc : Kernel->MemAllocs) {
          // std::pair<void *const, MemAllocRecord> *, Hash
          USMFreeHelper(MemAlloc->second.Context, MemAlloc->first,
                        MemAlloc->second.OwnNativeHandle);
        }
        Kernel->MemAllocs.clear();
      }
    }
  };

  // We've reset event data members above, now cleanup resources.
  if (AssociatedKernel) {
    ReleaseIndirectMem(AssociatedKernel);
    UR_CALL(urKernelRelease(AssociatedKernel));
  }

  if (AssociatedQueue) {
    {
      // Lock automatically releases when this goes out of scope.
      std::unique_lock<ur_shared_mutex> QueueLock(AssociatedQueue->Mutex,
                                                  std::defer_lock);
      if (!QueueLocked)
        QueueLock.lock();

      // If this event was the LastCommandEvent in the queue, being used
      // to make sure that commands were executed in-order, remove this.
      // If we don't do this, the event can get released and freed leaving
      // a dangling pointer to this event.  It could also cause unneeded
      // already finished events to show up in the wait list.
      if (AssociatedQueue->LastCommandEvent == Event) {
        AssociatedQueue->LastCommandEvent = nullptr;
      }
    }

    // Release this event since we explicitly retained it on creation and
    // association with queue. Events which don't have associated queue doesn't
    // require this release because it means that they are not created using
    // createEventAndAssociateQueue, i.e. additional retain was not made.
    UR_CALL(urEventReleaseInternal(Event));
  }

  // The list of dependent events will be appended to as we walk it so that this
  // algorithm doesn't go recursive due to dependent events themselves being
  // dependent on other events forming a potentially very deep tree, and deep
  // recursion.  That turned out to be a significant problem with the recursive
  // code that preceded this implementation.
  while (!EventsToBeReleased.empty()) {
    ur_event_handle_t DepEvent = EventsToBeReleased.front();
    DepEvent->Completed = true;
    EventsToBeReleased.pop_front();

    ur_kernel_handle_t DepEventKernel = nullptr;
    {
      std::scoped_lock<ur_shared_mutex> DepEventLock(DepEvent->Mutex);
      DepEvent->WaitList.collectEventsForReleaseAndDestroyUrZeEventList(
          EventsToBeReleased);
      if (IndirectAccessTrackingEnabled) {
        // DepEvent has finished, we can release the associated kernel if there
        // is one. This is the earliest place we can do this and it can't be
        // done twice, so it is safe. Lock automatically releases when this goes
        // out of scope.
        // TODO: this code needs to be moved out of the guard.
        if (DepEvent->CommandType == UR_COMMAND_KERNEL_LAUNCH &&
            DepEvent->CommandData) {
          DepEventKernel =
              reinterpret_cast<ur_kernel_handle_t>(DepEvent->CommandData);
          DepEvent->CommandData = nullptr;
        }
      }
    }
    if (DepEventKernel) {
      ReleaseIndirectMem(DepEventKernel);
      UR_CALL(urKernelRelease(DepEventKernel));
    }
    UR_CALL(urEventReleaseInternal(DepEvent));
  }

  return UR_RESULT_SUCCESS;
}

// Helper function for creating a PI event.
// The "Queue" argument specifies the PI queue where a command is submitted.
// The "HostVisible" argument specifies if event needs to be allocated from
// a host-visible pool.
//
ur_result_t EventCreate(ur_context_handle_t Context, ur_queue_handle_t Queue,
                        bool HostVisible, ur_event_handle_t *RetEvent) {

  bool ProfilingEnabled = !Queue || Queue->isProfilingEnabled();

  if (auto CachedEvent =
          Context->getEventFromContextCache(HostVisible, ProfilingEnabled)) {
    *RetEvent = CachedEvent;
    return UR_RESULT_SUCCESS;
  }

  ze_event_handle_t ZeEvent;
  ze_event_pool_handle_t ZeEventPool = {};

  size_t Index = 0;

  if (auto Res = Context->getFreeSlotInExistingOrNewPool(
          ZeEventPool, Index, HostVisible, ProfilingEnabled))
    return Res;

  ZeStruct<ze_event_desc_t> ZeEventDesc;
  ZeEventDesc.index = Index;
  ZeEventDesc.wait = 0;

  if (HostVisible) {
    ZeEventDesc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
  } else {
    //
    // Set the scope to "device" for every event. This is sufficient for
    // global device access and peer device access. If needed to be seen on
    // the host we are doing special handling, see EventsScope options.
    //
    // TODO: see if "sub-device" (ZE_EVENT_SCOPE_FLAG_SUBDEVICE) can better be
    //       used in some circumstances.
    //
    ZeEventDesc.signal = 0;
  }

  ZE2UR_CALL(zeEventCreate, (ZeEventPool, &ZeEventDesc, &ZeEvent));

  try {
    *RetEvent = new ur_event_handle_t_(
        ZeEvent, ZeEventPool, reinterpret_cast<ur_context_handle_t>(Context),
        UR_EXT_COMMAND_TYPE_USER, true);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  if (HostVisible)
    (*RetEvent)->HostVisibleEvent =
        reinterpret_cast<ur_event_handle_t>(*RetEvent);

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_event_handle_t_::reset() {
  UrQueue = nullptr;
  CleanedUp = false;
  Completed = false;
  CommandData = nullptr;
  CommandType = UR_EXT_COMMAND_TYPE_USER;
  WaitList = {};
  RefCountExternal = 0;
  RefCount.reset();
  CommandList = std::nullopt;

  if (!isHostVisible())
    HostVisibleEvent = nullptr;

  ZE2UR_CALL(zeEventHostReset, (ZeEvent));
  return UR_RESULT_SUCCESS;
}

ur_result_t _ur_ze_event_list_t::createAndRetainUrZeEventList(
    uint32_t EventListLength, const ur_event_handle_t *EventList,
    ur_queue_handle_t CurQueue, bool UseCopyEngine) {
  this->Length = 0;
  this->ZeEventList = nullptr;
  this->UrEventList = nullptr;

  if (CurQueue->isInOrderQueue() && CurQueue->LastCommandEvent != nullptr) {
    if (CurQueue->UsingImmCmdLists) {
      if (ReuseDiscardedEvents && CurQueue->isDiscardEvents()) {
        // If queue is in-order with discarded events and if
        // new command list is different from the last used command list then
        // signal new event from the last immediate command list. We are going
        // to insert a barrier in the new command list waiting for that event.
        auto QueueGroup = CurQueue->getQueueGroup(UseCopyEngine);
        uint32_t QueueGroupOrdinal, QueueIndex;
        auto NextIndex =
            QueueGroup.getQueueIndex(&QueueGroupOrdinal, &QueueIndex,
                                     /*QueryOnly */ true);
        auto NextImmCmdList = QueueGroup.ImmCmdLists[NextIndex];
        if (CurQueue->LastUsedCommandList != CurQueue->CommandListMap.end() &&
            CurQueue->LastUsedCommandList != NextImmCmdList) {
          CurQueue->signalEventFromCmdListIfLastEventDiscarded(
              CurQueue->LastUsedCommandList);
        }
      }
    } else {
      // Ensure LastCommandEvent's batch is submitted if it is differrent
      // from the one this command is going to. If we reuse discarded events
      // then signalEventFromCmdListIfLastEventDiscarded will be called at batch
      // close if needed.
      const auto &OpenCommandList =
          CurQueue->eventOpenCommandList(CurQueue->LastCommandEvent);
      if (OpenCommandList != CurQueue->CommandListMap.end() &&
          OpenCommandList->second.isCopy(CurQueue) != UseCopyEngine) {

        if (auto Res = CurQueue->executeOpenCommandList(
                OpenCommandList->second.isCopy(CurQueue)))
          return Res;
      }
    }
  }

  // For in-order queues, every command should be executed only after the
  // previous command has finished. The event associated with the last
  // enqueued command is added into the waitlist to ensure in-order semantics.
  bool IncludeLastCommandEvent =
      CurQueue->isInOrderQueue() && CurQueue->LastCommandEvent != nullptr;

  // If the last event is discarded then we already have a barrier waiting for
  // that event, so must not include the last command event into the wait
  // list because it will cause waiting for event which was reset.
  if (ReuseDiscardedEvents && CurQueue->isDiscardEvents() &&
      CurQueue->LastCommandEvent && CurQueue->LastCommandEvent->IsDiscarded)
    IncludeLastCommandEvent = false;

  try {
    uint32_t TmpListLength = 0;

    if (IncludeLastCommandEvent) {
      this->ZeEventList = new ze_event_handle_t[EventListLength + 1];
      this->UrEventList = new ur_event_handle_t[EventListLength + 1];
      std::shared_lock<ur_shared_mutex> Lock(CurQueue->LastCommandEvent->Mutex);
      this->ZeEventList[0] = CurQueue->LastCommandEvent->ZeEvent;
      this->UrEventList[0] = CurQueue->LastCommandEvent;
      this->UrEventList[0]->RefCount.increment();
      TmpListLength = 1;
    } else if (EventListLength > 0) {
      this->ZeEventList = new ze_event_handle_t[EventListLength];
      this->UrEventList = new ur_event_handle_t[EventListLength];
    }

    if (EventListLength > 0) {
      for (uint32_t I = 0; I < EventListLength; I++) {
        {
          std::shared_lock<ur_shared_mutex> Lock(EventList[I]->Mutex);
          if (EventList[I]->Completed)
            continue;

          // Poll of the host-visible events.
          auto HostVisibleEvent = EventList[I]->HostVisibleEvent;
          if (FilterEventWaitList && HostVisibleEvent) {
            auto Res = ZE_CALL_NOCHECK(zeEventQueryStatus,
                                       (HostVisibleEvent->ZeEvent));
            if (Res == ZE_RESULT_SUCCESS) {
              // Event has already completed, don't put it into the list
              continue;
            }
          }
        }

        auto Queue = EventList[I]->UrQueue;
        if (Queue) {
          // The caller of createAndRetainUrZeEventList must already hold
          // a lock of the CurQueue. Additionally lock the Queue if it
          // is different from CurQueue.
          // TODO: rework this to avoid deadlock when another thread is
          //       locking the same queues but in a different order.
          auto Lock = ((Queue == CurQueue)
                           ? std::unique_lock<ur_shared_mutex>()
                           : std::unique_lock<ur_shared_mutex>(Queue->Mutex));

          // If the event that is going to be waited is in an open batch
          // different from where this next command is going to be added,
          // then we have to force execute of that open command-list
          // to avoid deadlocks.
          //
          const auto &OpenCommandList =
              Queue->eventOpenCommandList(EventList[I]);
          if (OpenCommandList != Queue->CommandListMap.end()) {

            if (Queue == CurQueue &&
                OpenCommandList->second.isCopy(Queue) == UseCopyEngine) {
              // Don't force execute the batch yet since the new command
              // is going to the same open batch as the dependent event.
            } else {
              if (auto Res = Queue->executeOpenCommandList(
                      OpenCommandList->second.isCopy(Queue)))
                return Res;
            }
          }
        } else {
          // There is a dependency on an interop-event.
          // Similarily to the above to avoid dead locks ensure that
          // execution of all prior commands in the current command-
          // batch is visible to the host. This may not be the case
          // when we intended to have only last command in the batch
          // produce host-visible event, e.g.
          //
          //  event0 = interop event
          //  event1 = command1 (already in batch, no deps)
          //  event2 = command2 (is being added, dep on event0)
          //  event3 = signal host-visible event for the batch
          //  event1.wait()
          //  event0.signal()
          //
          // Make sure that event1.wait() will wait for a host-visible
          // event that is signalled before the command2 is enqueued.
          if (CurQueue->ZeEventsScope != AllHostVisible) {
            CurQueue->executeAllOpenCommandLists();
          }
        }

        std::shared_lock<ur_shared_mutex> Lock(EventList[I]->Mutex);
        this->ZeEventList[TmpListLength] = EventList[I]->ZeEvent;
        this->UrEventList[TmpListLength] = EventList[I];
        this->UrEventList[TmpListLength]->RefCount.increment();
        TmpListLength += 1;
      }
    }

    this->Length = TmpListLength;

  } catch (...) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t _ur_ze_event_list_t::insert(_ur_ze_event_list_t &Other) {
  if (this != &Other) {
    // save of the previous object values
    uint32_t PreLength = this->Length;
    ze_event_handle_t *PreZeEventList = this->ZeEventList;
    ur_event_handle_t *PreUrEventList = this->UrEventList;

    // allocate new memory
    uint32_t Length = PreLength + Other.Length;
    this->ZeEventList = new ze_event_handle_t[Length];
    this->UrEventList = new ur_event_handle_t[Length];

    // copy elements
    uint32_t TmpListLength = 0;
    for (uint32_t I = 0; I < PreLength; I++) {
      this->ZeEventList[TmpListLength] = std::move(PreZeEventList[I]);
      this->UrEventList[TmpListLength] = std::move(PreUrEventList[I]);
      TmpListLength += 1;
    }
    for (uint32_t I = 0; I < Other.Length; I++) {
      this->ZeEventList[TmpListLength] = std::move(Other.ZeEventList[I]);
      this->UrEventList[TmpListLength] = std::move(Other.UrEventList[I]);
      TmpListLength += 1;
    }
    this->Length = TmpListLength;

    // Free previous allocated memory
    delete[] PreZeEventList;
    delete[] PreUrEventList;
    delete[] Other.ZeEventList;
    delete[] Other.UrEventList;
    Other.ZeEventList = nullptr;
    Other.UrEventList = nullptr;
    Other.Length = 0;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t _ur_ze_event_list_t::collectEventsForReleaseAndDestroyUrZeEventList(
    std::list<ur_event_handle_t> &EventsToBeReleased) {
  // event wait lists are owned by events, this function is called with owning
  // event lock taken, hence it is thread safe
  for (uint32_t I = 0; I < Length; I++) {
    // Add the event to be released to the list
    EventsToBeReleased.push_back(UrEventList[I]);
  }
  Length = 0;

  if (ZeEventList != nullptr) {
    delete[] ZeEventList;
    ZeEventList = nullptr;
  }
  if (UrEventList != nullptr) {
    delete[] UrEventList;
    UrEventList = nullptr;
  }

  return UR_RESULT_SUCCESS;
}

// Tells if this event is with profiling capabilities.
bool ur_event_handle_t_::isProfilingEnabled() const {
  return !UrQueue || // tentatively assume user events are profiling enabled
         (UrQueue->Properties & UR_QUEUE_FLAG_PROFILING_ENABLE) != 0;
}
