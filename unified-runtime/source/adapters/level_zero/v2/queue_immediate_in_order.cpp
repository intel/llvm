//===--------- queue_immediate_in_order.cpp - Level Zero Adapter ---------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "queue_immediate_in_order.hpp"
#include "command_buffer.hpp"
#include "kernel.hpp"
#include "memory.hpp"
#include "ur.hpp"

#include "../common/latency_tracker.hpp"
#include "../helpers/kernel_helpers.hpp"
#include "../image_common.hpp"

#include "../program.hpp"
#include "../ur_interface_loader.hpp"
#include <mutex>

namespace v2 {

ur_queue_immediate_in_order_t::ur_queue_immediate_in_order_t(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, uint32_t ordinal,
    ze_command_queue_priority_t priority, std::optional<int32_t> index,
    event_flags_t eventFlags, ur_queue_flags_t flags)
    : hContext(hContext), hDevice(hDevice),
      commandListManager(
          hContext, hDevice,
          hContext->getCommandListCache().getImmediateCommandList(
              hDevice->ZeDevice,
              {true, ordinal, true /* always enable copy offload */},
              ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, priority, index)),
      flags(flags),
      eventPool(hContext->getEventPoolCache(PoolCacheType::Immediate)
                    .borrow(hDevice->Id.value(), eventFlags)),
      normalEventsPool(hContext->getEventPoolCache(PoolCacheType::Immediate)
                           .borrow(hDevice->Id.value(), 0)) {}

ur_queue_immediate_in_order_t::ur_queue_immediate_in_order_t(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    raii::command_list_unique_handle commandListHandle,
    event_flags_t eventFlags, ur_queue_flags_t flags)
    : hContext(hContext), hDevice(hDevice),
      commandListManager(hContext, hDevice, std::move(commandListHandle)),
      flags(flags),
      eventPool(hContext->getEventPoolCache(PoolCacheType::Immediate)
                    .borrow(hDevice->Id.value(), eventFlags)),
      normalEventsPool(hContext->getEventPoolCache(PoolCacheType::Immediate)
                           .borrow(hDevice->Id.value(), 0)) {}

ur_result_t
ur_queue_immediate_in_order_t::queueGetInfo(ur_queue_info_t propName,
                                            size_t propSize, void *pPropValue,
                                            size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);
  // TODO: consider support for queue properties and size
  switch ((uint32_t)propName) { // cast to avoid warnings on EXT enum values
  case UR_QUEUE_INFO_CONTEXT:
    return ReturnValue(hContext);
  case UR_QUEUE_INFO_DEVICE:
    return ReturnValue(hDevice);
  case UR_QUEUE_INFO_REFERENCE_COUNT:
    return ReturnValue(uint32_t{RefCount.getCount()});
  case UR_QUEUE_INFO_FLAGS:
    return ReturnValue(flags);
  case UR_QUEUE_INFO_SIZE:
  case UR_QUEUE_INFO_DEVICE_DEFAULT:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  case UR_QUEUE_INFO_EMPTY: {
    auto status = ZE_CALL_NOCHECK(
        zeCommandListHostSynchronize,
        (commandListManager.get_no_lock()->getZeCommandList(), 0));
    if (status == ZE_RESULT_SUCCESS) {
      return ReturnValue(true);
    } else if (status == ZE_RESULT_NOT_READY) {
      return ReturnValue(false);
    } else {
      return ze2urResult(status);
    }
  }
  default:
    UR_LOG(ERR,
           "Unsupported ParamName in urQueueGetInfo: "
           "ParamName=ParamName={}(0x{})",
           propName, logger::toHex(propName));
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::queueGetNativeHandle(
    ur_queue_native_desc_t * /*pDesc*/, ur_native_handle_t *phNativeQueue) {
  *phNativeQueue = reinterpret_cast<ur_native_handle_t>(
      commandListManager.get_no_lock()->getZeCommandList());
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::queueFinish() {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::queueFinish");

  auto lockedCommandListManager = commandListManager.lock();

  ZE2UR_CALL(zeCommandListHostSynchronize,
             (lockedCommandListManager->getZeCommandList(), UINT64_MAX));

  hContext->getAsyncPool()->cleanupPoolsForQueue(this);
  hContext->forEachUsmPool([this](ur_usm_pool_handle_t hPool) {
    hPool->cleanupPoolsForQueue(this);
    return true;
  });

  UR_CALL(lockedCommandListManager->releaseSubmittedKernels());

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::queueFlush() {
  return UR_RESULT_SUCCESS;
}

ur_queue_immediate_in_order_t::~ur_queue_immediate_in_order_t() {
  try {
    if (HostTaskWorker.has_value()) {
      HostTaskSender->close();
      HostTaskWorker->join();
    }

    UR_CALL_THROWS(queueFinish());
  } catch (...) {
    // Ignore errors during destruction
  }
}

ur_result_t ur_queue_immediate_in_order_t::enqueueEventsWaitWithBarrier(
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_in_order_t::enqueueEventsWaitWithBarrier");
  // For in-order queue we don't need a real barrier, just wait for
  // requested events in potentially different queues and add a "barrier"
  // event signal because it is already guaranteed that previous commands
  // in this queue are completed when the signal is started. However, we do
  // need to use barrier if profiling is enabled: see
  // zeCommandListAppendWaitOnEvents
  if ((flags & UR_QUEUE_FLAG_PROFILING_ENABLE) != 0) {
    return commandListManager.lock()->appendEventsWaitWithBarrier(
        numEventsInWaitList, phEventWaitList,
        createEventIfRequested(eventPool.get(), phEvent, this));
  } else {
    return commandListManager.lock()->appendEventsWait(
        numEventsInWaitList, phEventWaitList,
        createEventIfRequested(eventPool.get(), phEvent, this));
  }
}

ur_result_t ur_queue_immediate_in_order_t::enqueueHostTaskExp(
    ur_exp_host_task_function_t pfnHostTask, void *data,
    const ur_exp_host_task_properties_t *, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  // L0 does not support enqueuing host tasks on a command list yet.
  // The implementation below works around that by spawning a worker thread
  // that waits on the input events, executes the host task callback, and
  // then signals the output event.

  // Since this functionality relies on the worker thread to complete some work
  // and then signal the event from host, the output event of this function is a
  // normal (not counter-based) event so that calling zeEventHostSignal is
  // legal. This leads to a slightly more complicated event lifetime
  // management...

  std::call_once(HostTaskInit, [&]() {
    auto [sender, receiver] = mpmc::createChannel<HostTaskData>();

    HostTaskWorker = std::thread([Receiver = std::move(receiver)]() mutable {
      std::queue<HostTaskData> Local;
      std::vector<HostTaskData> Cleanup;

      // We can't immediately release the (normal) output event after it has
      // been signaled because that might lead to use-after-free inside of the
      // driver. Instead, we create a "cleanup" event, to be signaled on the
      // command list immediately after our output event is signaled. This
      // ensures that both the output event and the cleanup event are safe to be
      // released.
      auto EventCleanup = [&Cleanup]() {
        auto CleanupNewEnd = std::remove_if(
            Cleanup.begin(), Cleanup.end(), [](const HostTaskData &task) {
              bool Completed =
                  ZE_CALL_NOCHECK(zeEventQueryStatus,
                                  (task.CleanupEvent->getZeEvent())) ==
                  ZE_RESULT_SUCCESS;
              if (Completed) {
                task.CleanupEvent->release();
                task.OutputEvent->release();
              }
              return Completed;
            });
        Cleanup.erase(CleanupNewEnd, Cleanup.end());
      };

      for (;;) {
        EventCleanup();

        std::optional<HostTaskData> Data;
        if (Local.empty()) {
          // Blocking. It will unblock either on new data or when the channel
          // has closed.
          Data = Receiver.receive();
          if (!Data.has_value()) // no data left and the channel has closed
            break;
          Local.push(*Data);
        } else { // if we already have some host tasks being processed...
          // Nonblocking. Opportunistically tries to get data from the channel.
          while ((Data = Receiver.tryReceive()) != std::nullopt)
            Local.push(*Data);
        }

        auto &Task = Local.front();
        // we need to wait for the host tasks' input event to signal. We need to
        // do this on host.
        bool Completed = ZE_CALL_NOCHECK(zeEventQueryStatus,
                                         (Task.InputEvent->getZeEvent())) ==
                         ZE_RESULT_SUCCESS;
        if (!Completed) {
          continue;
        }
        Task.InputEvent->release();

        Task.pfnHostTask(Task.data);

        ZE_CALL_NOCHECK(zeEventHostSignal, (Task.OutputEvent->getZeEvent()));

        Cleanup.push_back(Task);
        Local.pop();
      }

      while (!Cleanup.empty()) {
        EventCleanup();
      }
    });

    HostTaskSender = std::move(sender);
  });

  auto CommandListLocked = commandListManager.lock();

  HostTaskData HostTask;
  HostTask.pfnHostTask = pfnHostTask;
  HostTask.data = data;

  // Aggregate all input events into a single event for the host task worker
  // thread to poll.
  HostTask.InputEvent = createCounterEvent(UR_COMMAND_HOST_TASK_EXP);

  // Since this is an in-order queue, we also need to make sure the host task
  // executes only after the previous operations on the queue.
  CommandListLocked->appendEventsWait(numEventsInWaitList, phEventWaitList,
                                      HostTask.InputEvent);

  HostTask.CleanupEvent = createCounterEvent(UR_COMMAND_HOST_TASK_EXP);

  // We can't use counter-based events. We need to explicitly use the normal
  // events pool so that the event can be signaled from host (the worker
  // thread).
  HostTask.OutputEvent = createNormalEvent(UR_COMMAND_HOST_TASK_EXP);

  auto *OutputZeEvent = HostTask.OutputEvent->getZeEvent();
  // this will block the command list until the host task completes
  ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
             (CommandListLocked->getZeCommandList(), 1, &OutputZeEvent));

  // this will signal the cleanup event, triggering the release of host task's
  // cleanup and output events inside of the worker thread.
  ZE2UR_CALL(zeCommandListAppendSignalEvent,
             (CommandListLocked->getZeCommandList(),
              HostTask.CleanupEvent->getZeEvent()));

  if (phEvent) {
    // we return a cleanup event to avoid ever having a normal event (the
    // OutputEvent) on a wait list.
    *phEvent = HostTask.CleanupEvent;
    // we need the extra retain since the event is going to be used internally
    // by the worker thread.
    HostTask.CleanupEvent->retain();
  }

  HostTaskSender->send(std::move(HostTask));

  return UR_RESULT_SUCCESS;
}

} // namespace v2
