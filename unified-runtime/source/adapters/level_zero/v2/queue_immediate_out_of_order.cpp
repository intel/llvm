//===--------- queue_immediate_in_order.cpp - Level Zero Adapter ---------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "queue_immediate_out_of_order.hpp"
#include "command_buffer.hpp"
#include "kernel.hpp"
#include "memory.hpp"
#include "ur.hpp"

#include "../common/latency_tracker.hpp"
#include "../helpers/kernel_helpers.hpp"
#include "../image_common.hpp"

#include "../program.hpp"
#include "../ur_interface_loader.hpp"

namespace v2 {

template <size_t... Is>
std::array<lockable<ur_command_list_manager>, sizeof...(Is)>
createCommandListManagers(ur_context_handle_t hContext,
                          ur_device_handle_t hDevice, uint32_t ordinal,
                          ze_command_queue_priority_t priority,
                          std::index_sequence<Is...>) {
  return {
      ((void)Is, lockable<ur_command_list_manager>(
                     hContext, hDevice,
                     hContext->getCommandListCache().getImmediateCommandList(
                         hDevice->ZeDevice,
                         {true, ordinal, true /* always enable copy offload */},
                         ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, priority)))...};
}

template <size_t N>
std::array<lockable<ur_command_list_manager>, N>
createCommandListManagers(ur_context_handle_t hContext,
                          ur_device_handle_t hDevice, uint32_t ordinal,
                          ze_command_queue_priority_t priority) {
  return createCommandListManagers(hContext, hDevice, ordinal, priority,
                                   std::make_index_sequence<N>{});
}

ur_queue_immediate_out_of_order_t::ur_queue_immediate_out_of_order_t(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, uint32_t ordinal,
    ze_command_queue_priority_t priority, event_flags_t eventFlags,
    ur_queue_flags_t flags)
    : hContext(hContext), hDevice(hDevice),
      commandListManagers(createCommandListManagers<numCommandLists>(
          hContext, hDevice, ordinal, priority)),
      eventPool(hContext->getEventPoolCache().borrow(hDevice->Id.value(),
                                                     eventFlags)),
      flags(flags) {
  // TODO: dummy operation to ensure that counter-based events are signaled
  // rewrite this using zeCreateCounterBasedEventExt
  void *tmpMem = nullptr;
  uint32_t tmpPattern = 0;
  UR_CALL_THROWS(ur::level_zero::urUSMHostAlloc(hContext, nullptr, nullptr,
                                                sizeof(tmpPattern), &tmpMem));

  for (size_t i = 0; i < numCommandLists; ++i) {
    internalSignalEvents[i] = eventPool->allocate();
    commandListManagers[i].get_no_lock()->enqueueUSMFill(
        tmpMem, sizeof(tmpPattern), &tmpPattern, sizeof(tmpPattern), 0, nullptr,
        &internalSignalEvents[i]);
    ZE2UR_CALL_THROWS(
        zeCommandListHostSynchronize,
        (commandListManagers[i].get_no_lock()->getZeCommandList(), UINT64_MAX));

    signalEvents.assign(i, internalSignalEvents[i], false);
  }

  UR_CALL_THROWS(ur::level_zero::urUSMFree(hContext, tmpMem));
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
    return ReturnValue(uint32_t{RefCount.load()});
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
        throw ze2urResult(status);
      }
    };

    bool empty = std::all_of(
        commandListManagers.begin(), commandListManagers.end(),
        [&](auto &cmdListManager) {
          return isCmdListEmpty(cmdListManager.lock()->getZeCommandList());
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
    ur_queue_native_desc_t * /*pDesc*/, ur_native_handle_t *phNativeQueue) {
  *phNativeQueue = reinterpret_cast<ur_native_handle_t>(
      commandListManagers[getNextCommandListId()]
          .get_no_lock()
          ->getZeCommandList());
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_out_of_order_t::queueFinish() {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_out_of_order_t::queueFinish");

  auto lastCommandListId =
      commandListIndex.load(std::memory_order_relaxed) % numCommandLists;

  UR_CALL(commandListManagers[lastCommandListId].lock()->enqueueEventsWait(
      numCommandLists, signalEvents.events.data(), nullptr));
  ZE2UR_CALL(zeCommandListHostSynchronize,
             (commandListManagers[lastCommandListId].lock()->getZeCommandList(),
              UINT64_MAX));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_out_of_order_t::queueFlush() {
  return UR_RESULT_SUCCESS;
}

ur_queue_immediate_out_of_order_t::~ur_queue_immediate_out_of_order_t() {
  try {
    UR_CALL_THROWS(queueFinish());
  } catch (...) {
    // Ignore errors during destruction
  }
}

ur_result_t ur_queue_immediate_out_of_order_t::enqueueEventsWait(
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  auto lastCommandListId =
      commandListIndex.load(std::memory_order_relaxed) % numCommandLists;

  UR_CALL(commandListManagers[lastCommandListId].lock()->enqueueEventsWait(
      numEventsInWaitList, phEventWaitList,
      createOrForwardSignalEvent(lastCommandListId, phEvent,
                                 UR_COMMAND_EVENTS_WAIT)));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_out_of_order_t::enqueueEventsWaitWithBarrierImpl(
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  auto lastCommandListId =
      commandListIndex.load(std::memory_order_relaxed) % numCommandLists;

  UR_CALL(commandListManagers[lastCommandListId]
              .lock()
              ->enqueueEventsWaitWithBarrier(
                  numEventsInWaitList, phEventWaitList,
                  createOrForwardSignalEvent(lastCommandListId, phEvent,
                                             UR_COMMAND_EVENTS_WAIT)));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_out_of_order_t::enqueueEventsWaitWithBarrier(
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  // For in-order command lists we don't need a real barrier, just wait for
  // requested events in potentially different queues and add a "barrier"
  // event signal because it is already guaranteed that previous commands
  // in this queue are completed when the signal is started. However, we do
  // need to use barrier if profiling is enabled: see
  // zeCommandListAppendWaitOnEvents
  if ((flags & UR_QUEUE_FLAG_PROFILING_ENABLE) != 0) {
    return enqueueEventsWaitWithBarrierImpl(numEventsInWaitList,
                                            phEventWaitList, phEvent);
  } else {
    return enqueueEventsWait(numEventsInWaitList, phEventWaitList, phEvent);
  }
}

ur_result_t ur_queue_immediate_out_of_order_t::enqueueEventsWaitWithBarrierExt(
    const ur_exp_enqueue_ext_properties_t *, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  return enqueueEventsWaitWithBarrier(numEventsInWaitList, phEventWaitList,
                                      phEvent);
}

} // namespace v2
