//===--------- queue_immediate_in_order.cpp - Level Zero Adapter ---------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "queue_immediate_out_of_order.hpp"
#include "../common/latency_tracker.hpp"
#include "command_list_manager.hpp"
#include "ur.hpp"

namespace v2 {

template <size_t N>
std::array<ur_command_list_manager, N> createCommandListManagers(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, uint32_t ordinal,
    ze_command_queue_priority_t priority, std::optional<int32_t> index) {
  return createArrayOf<ur_command_list_manager, N>([&](size_t) {
    return ur_command_list_manager(
        hContext, hDevice,
        hContext->getCommandListCache().getImmediateCommandList(
            hDevice->ZeDevice,
            {true, ordinal, true /* always enable copy offload */},
            ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, priority, index));
  });
}

ur_queue_immediate_out_of_order_t::ur_queue_immediate_out_of_order_t(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, uint32_t ordinal,
    ze_command_queue_priority_t priority, std::optional<int32_t> index,
    event_flags_t eventFlags, ur_queue_flags_t flags)
    : hContext(hContext), hDevice(hDevice),
      eventPool(hContext->getEventPoolCache(PoolCacheType::Immediate)
                    .borrow(hDevice->Id.value(), eventFlags)),
      commandListManagers(createCommandListManagers<numCommandLists>(
          hContext, hDevice, ordinal, priority, index)),
      flags(flags) {
  for (size_t i = 0; i < numCommandLists; i++) {
    barrierEvents[i] = eventPool->allocate();
  }
}

ur_result_t ur_queue_immediate_out_of_order_t::queueGetInfo(
    ur_queue_info_t propName, size_t propSize, void *pPropValue,
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
    auto isCmdListEmpty = [](ze_command_list_handle_t cmdList) {
      auto status = ZE_CALL_NOCHECK(zeCommandListHostSynchronize, (cmdList, 0));
      if (status == ZE_RESULT_SUCCESS) {
        return true;
      } else if (status == ZE_RESULT_NOT_READY) {
        return false;
      } else {
        UR_DFAILURE("getting queue info failed with: " << status);
        throw ze2urResult(status);
      }
    };

    auto commandListManagersLocked = commandListManagers.lock();

    bool empty = std::all_of(
        commandListManagersLocked->begin(), commandListManagersLocked->end(),
        [&](auto &cmdListManager) {
          return isCmdListEmpty(cmdListManager.getZeCommandList());
        });

    return ReturnValue(empty);
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

ur_result_t ur_queue_immediate_out_of_order_t::queueGetNativeHandle(
    ur_queue_native_desc_t *pDesc, ur_native_handle_t *phNativeQueue) {
  *phNativeQueue = reinterpret_cast<ur_native_handle_t>(
      (*commandListManagers.get_no_lock())[getNextCommandListId()]
          .getZeCommandList());
  if (pDesc && pDesc->pNativeData) {
    // pNativeData == isImmediateQueue
    *(reinterpret_cast<int32_t *>(pDesc->pNativeData)) = 1;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_out_of_order_t::queueFinish() {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_out_of_order_t::queueFinish");

  auto commandListManagersLocked = commandListManagers.lock();

  for (size_t i = 0; i < numCommandLists; i++) {
    ZE2UR_CALL(zeCommandListHostSynchronize,
               (commandListManagersLocked[i].getZeCommandList(), UINT64_MAX));
    UR_CALL(commandListManagersLocked[i].releaseSubmittedKernels());
  }

  hContext->getAsyncPool()->cleanupPoolsForQueue(this);
  hContext->forEachUsmPool([this](ur_usm_pool_handle_t hPool) {
    hPool->cleanupPoolsForQueue(this);
    return true;
  });

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_out_of_order_t::queueFlush() {
  return UR_RESULT_SUCCESS;
}

ur_queue_immediate_out_of_order_t::~ur_queue_immediate_out_of_order_t() {
  try {
    UR_CALL_THROWS(queueFinish());

    for (size_t i = 0; i < numCommandLists; i++) {
      barrierEvents[i]->release();
    }
  } catch (...) {
    // Ignore errors during destruction
  }
}

ur_result_t ur_queue_immediate_out_of_order_t::enqueueEventsWaitWithBarrier(
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_out_of_order_t::enqueueEventsWaitWithBarrier");
  // Since we use L0 in-order command lists, we don't need a real L0 barrier,
  // just wait for requested events in potentially different queues and add a
  // "barrier" event signal because it is already guaranteed that previous
  // commands in this queue are completed when the signal is started. However,
  // we do need to use barrier if profiling is enabled: see
  // zeCommandListAppendWaitOnEvents
  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList);

  bool needsRealBarrier = (flags & UR_QUEUE_FLAG_PROFILING_ENABLE) != 0;
  auto barrierFn = needsRealBarrier
                       ? &ur_command_list_manager::appendEventsWaitWithBarrier
                       : &ur_command_list_manager::appendEventsWait;

  auto commandListManagersLocked = commandListManagers.lock();

  // Enqueue wait for the user-provider events on the first command list.
  UR_CALL(commandListManagersLocked[0].appendEventsWait(waitListView,
                                                        barrierEvents[0]));

  wait_list_view emptyWaitlist = wait_list_view(nullptr, 0);

  // Request barrierEvents[id] to be signaled on remaining command lists.
  for (size_t id = 1; id < numCommandLists; id++) {
    UR_CALL(commandListManagersLocked[id].appendEventsWait(emptyWaitlist,
                                                           barrierEvents[id]));
  }

  // Enqueue barriers on all command lists by waiting on barrierEvents.
  wait_list_view barrierEventsWaitList =
      wait_list_view(barrierEvents.data(), numCommandLists);

  if (phEvent) {
    UR_CALL(std::invoke(
        barrierFn, commandListManagersLocked[0], barrierEventsWaitList,
        createEventIfRequested(eventPool.get(), phEvent, this)));
  }

  for (size_t id = phEvent ? 1 : 0; id < numCommandLists; id++) {
    UR_CALL(std::invoke(barrierFn, commandListManagersLocked[0],
                        barrierEventsWaitList, nullptr));
  }

  return UR_RESULT_SUCCESS;
}

} // namespace v2
