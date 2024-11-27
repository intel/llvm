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
#include "kernel.hpp"
#include "memory.hpp"
#include "ur.hpp"

#include "../common/latency_tracker.hpp"
#include "../helpers/kernel_helpers.hpp"
#include "../helpers/memory_helpers.hpp"
#include "../program.hpp"
#include "../ur_interface_loader.hpp"

namespace v2 {

std::pair<ze_event_handle_t *, uint32_t>
ur_queue_immediate_in_order_t::getWaitListView(
    const ur_event_handle_t *phWaitEvents, uint32_t numWaitEvents) {

  waitList.resize(numWaitEvents);
  for (uint32_t i = 0; i < numWaitEvents; i++) {
    waitList[i] = phWaitEvents[i]->getZeEvent();
  }

  return {waitList.data(), static_cast<uint32_t>(numWaitEvents)};
}

static int32_t getZeOrdinal(ur_device_handle_t hDevice) {
  return hDevice->QueueGroup[queue_group_type::Compute].ZeOrdinal;
}

static std::optional<int32_t> getZeIndex(const ur_queue_properties_t *pProps) {
  if (pProps && pProps->pNext) {
    const ur_base_properties_t *extendedDesc =
        reinterpret_cast<const ur_base_properties_t *>(pProps->pNext);
    if (extendedDesc->stype == UR_STRUCTURE_TYPE_QUEUE_INDEX_PROPERTIES) {
      const ur_queue_index_properties_t *indexProperties =
          reinterpret_cast<const ur_queue_index_properties_t *>(extendedDesc);
      return indexProperties->computeIndex;
    }
  }
  return std::nullopt;
}

static ze_command_queue_priority_t getZePriority(ur_queue_flags_t flags) {
  if ((flags & UR_QUEUE_FLAG_PRIORITY_LOW) != 0)
    return ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW;
  if ((flags & UR_QUEUE_FLAG_PRIORITY_HIGH) != 0)
    return ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH;
  return ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
}

ur_command_list_handler_t::ur_command_list_handler_t(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_queue_properties_t *pProps)
    : commandList(hContext->commandListCache.getImmediateCommandList(
          hDevice->ZeDevice, true, getZeOrdinal(hDevice),
          true /* always enable copy offload */,
          ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
          getZePriority(pProps ? pProps->flags : ur_queue_flags_t{}),
          getZeIndex(pProps))) {}

ur_command_list_handler_t::ur_command_list_handler_t(
    ze_command_list_handle_t hZeCommandList, bool ownZeHandle)
    : commandList(hZeCommandList,
                  [ownZeHandle](ze_command_list_handle_t hZeCommandList) {
                    if (ownZeHandle) {
                      zeCommandListDestroy(hZeCommandList);
                    }
                  }) {}

static event_flags_t eventFlagsFromQueueFlags(ur_queue_flags_t flags) {
  event_flags_t eventFlags = EVENT_FLAGS_COUNTER;
  if (flags & UR_QUEUE_FLAG_PROFILING_ENABLE)
    eventFlags |= EVENT_FLAGS_PROFILING_ENABLED;
  return eventFlags;
}

ur_queue_immediate_in_order_t::ur_queue_immediate_in_order_t(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_queue_properties_t *pProps)
    : hContext(hContext), hDevice(hDevice), flags(pProps ? pProps->flags : 0),
      eventPool(hContext->eventPoolCache.borrow(
          hDevice->Id.value(), eventFlagsFromQueueFlags(flags))),
      handler(hContext, hDevice, pProps) {}

ur_queue_immediate_in_order_t::ur_queue_immediate_in_order_t(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_native_handle_t hNativeHandle, ur_queue_flags_t flags, bool ownZeQueue)
    : hContext(hContext), hDevice(hDevice), flags(flags),
      eventPool(hContext->eventPoolCache.borrow(
          hDevice->Id.value(), eventFlagsFromQueueFlags(flags))),
      handler(reinterpret_cast<ze_command_list_handle_t>(hNativeHandle),
              ownZeQueue) {}

ur_event_handle_t
ur_queue_immediate_in_order_t::getSignalEvent(ur_event_handle_t *hUserEvent,
                                              ur_command_t commandType) {
  if (hUserEvent) {
    *hUserEvent = eventPool->allocate(this, commandType);
    return *hUserEvent;
  } else {
    return nullptr;
  }
}

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
    return ReturnValue(uint32_t{RefCount.load()});
  case UR_QUEUE_INFO_FLAGS:
    return ReturnValue(flags);
  case UR_QUEUE_INFO_SIZE:
  case UR_QUEUE_INFO_DEVICE_DEFAULT:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  case UR_QUEUE_INFO_EMPTY: {
    // We can't tell if the queue is empty as we don't hold to any events
    return ReturnValue(false);
  }
  default:
    logger::error("Unsupported ParamName in urQueueGetInfo: "
                  "ParamName=ParamName={}(0x{})",
                  propName, logger::toHex(propName));
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::queueRetain() {
  RefCount.increment();
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::queueRelease() {
  if (!RefCount.decrementAndTest())
    return UR_RESULT_SUCCESS;

  UR_CALL(queueFinish());

  delete this;
  return UR_RESULT_SUCCESS;
}

void ur_queue_immediate_in_order_t::deferEventFree(ur_event_handle_t hEvent) {
  std::unique_lock<ur_shared_mutex> lock(this->Mutex);
  deferredEvents.push_back(hEvent);
}

ur_result_t ur_queue_immediate_in_order_t::queueGetNativeHandle(
    ur_queue_native_desc_t *pDesc, ur_native_handle_t *phNativeQueue) {
  std::ignore = pDesc;
  *phNativeQueue =
      reinterpret_cast<ur_native_handle_t>(this->handler.commandList.get());
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::queueFinish() {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::queueFinish");

  std::unique_lock<ur_shared_mutex> lock(this->Mutex);

  // TODO: use zeEventHostSynchronize instead?
  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_in_order_t::zeCommandListHostSynchronize");
  ZE2UR_CALL(zeCommandListHostSynchronize,
             (handler.commandList.get(), UINT64_MAX));

  // Free deferred events
  for (auto &hEvent : deferredEvents) {
    hEvent->releaseDeferred();
  }
  deferredEvents.clear();

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::queueFlush() {
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueKernelLaunch(
    ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueKernelLaunch");

  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hKernel->getProgramHandle(), UR_RESULT_ERROR_INVALID_NULL_POINTER);

  UR_ASSERT(workDim > 0, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);
  UR_ASSERT(workDim < 4, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);

  ze_kernel_handle_t hZeKernel = hKernel->getZeHandle(hDevice);

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(this->Mutex,
                                                          hKernel->Mutex);

  ze_group_count_t zeThreadGroupDimensions{1, 1, 1};
  uint32_t WG[3]{};
  UR_CALL(calculateKernelWorkDimensions(hZeKernel, hDevice,
                                        zeThreadGroupDimensions, WG, workDim,
                                        pGlobalWorkSize, pLocalWorkSize));

  auto signalEvent = getSignalEvent(phEvent, UR_COMMAND_KERNEL_LAUNCH);

  auto waitList = getWaitListView(phEventWaitList, numEventsInWaitList);

  bool memoryMigrated = false;
  auto memoryMigrate = [&](void *src, void *dst, size_t size) {
    ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                      (handler.commandList.get(), dst, src, size, nullptr,
                       waitList.second, waitList.first));
    memoryMigrated = true;
  };

  UR_CALL(hKernel->prepareForSubmission(hContext, hDevice, pGlobalWorkOffset,
                                        workDim, WG[0], WG[1], WG[2],
                                        memoryMigrate));

  if (memoryMigrated) {
    // If memory was migrated, we don't need to pass the wait list to
    // the copy command again.
    waitList.first = nullptr;
    waitList.second = 0;
  }

  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_in_order_t::zeCommandListAppendLaunchKernel");
  auto zeSignalEvent = signalEvent ? signalEvent->getZeEvent() : nullptr;
  ZE2UR_CALL(zeCommandListAppendLaunchKernel,
             (handler.commandList.get(), hZeKernel, &zeThreadGroupDimensions,
              zeSignalEvent, waitList.second, waitList.first));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueEventsWait(
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueEventsWait");

  std::scoped_lock<ur_shared_mutex> lock(this->Mutex);

  if (!numEventsInWaitList && !phEvent) {
    // nop
    return UR_RESULT_SUCCESS;
  }

  auto signalEvent = getSignalEvent(phEvent, UR_COMMAND_EVENTS_WAIT);
  auto [pWaitEvents, numWaitEvents] =
      getWaitListView(phEventWaitList, numEventsInWaitList);

  if (numWaitEvents > 0) {
    ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
               (handler.commandList.get(), numWaitEvents, pWaitEvents));
  }

  if (signalEvent) {
    ZE2UR_CALL(zeCommandListAppendSignalEvent,
               (handler.commandList.get(), signalEvent->getZeEvent()));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueEventsWaitWithBarrierImpl(
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_in_order_t::enqueueEventsWaitWithBarrier");

  std::scoped_lock<ur_shared_mutex> lock(this->Mutex);

  if (!numEventsInWaitList && !phEvent) {
    // nop
    return UR_RESULT_SUCCESS;
  }

  auto signalEvent =
      getSignalEvent(phEvent, UR_COMMAND_EVENTS_WAIT_WITH_BARRIER);
  auto [pWaitEvents, numWaitEvents] =
      getWaitListView(phEventWaitList, numEventsInWaitList);

  ZE2UR_CALL(zeCommandListAppendBarrier,
             (handler.commandList.get(), signalEvent->getZeEvent(),
              numWaitEvents, pWaitEvents));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueEventsWaitWithBarrier(
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  // For in-order queue we don't need a real barrier, just wait for
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

ur_result_t ur_queue_immediate_in_order_t::enqueueEventsWaitWithBarrierExt(
    const ur_exp_enqueue_ext_properties_t *, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  return enqueueEventsWaitWithBarrier(numEventsInWaitList, phEventWaitList,
                                      phEvent);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueGenericCopyUnlocked(
    ur_mem_handle_t src, ur_mem_handle_t dst, bool blocking, size_t srcOffset,
    size_t dstOffset, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent,
    ur_command_t commandType) {
  auto signalEvent = getSignalEvent(phEvent, commandType);

  auto waitList = getWaitListView(phEventWaitList, numEventsInWaitList);

  bool memoryMigrated = false;
  auto pSrc = ur_cast<char *>(src->getDevicePtr(
      hDevice, ur_mem_handle_t_::device_access_mode_t::read_only, srcOffset,
      size, [&](void *src, void *dst, size_t size) {
        ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                          (handler.commandList.get(), dst, src, size, nullptr,
                           waitList.second, waitList.first));
        memoryMigrated = true;
      }));

  auto pDst = ur_cast<char *>(dst->getDevicePtr(
      hDevice, ur_mem_handle_t_::device_access_mode_t::write_only, dstOffset,
      size, [&](void *src, void *dst, size_t size) {
        ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                          (handler.commandList.get(), dst, src, size, nullptr,
                           waitList.second, waitList.first));
        memoryMigrated = true;
      }));

  if (memoryMigrated) {
    // If memory was migrated, we don't need to pass the wait list to
    // the copy command again.
    waitList.first = nullptr;
    waitList.second = 0;
  }

  auto zeSignalEvent = signalEvent ? signalEvent->getZeEvent() : nullptr;
  ZE2UR_CALL(zeCommandListAppendMemoryCopy,
             (handler.commandList.get(), pDst, pSrc, size, zeSignalEvent,
              waitList.second, waitList.first));

  if (blocking) {
    ZE2UR_CALL(zeCommandListHostSynchronize,
               (handler.commandList.get(), UINT64_MAX));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemBufferRead(
    ur_mem_handle_t hBuffer, bool blockingRead, size_t offset, size_t size,
    void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueMemBufferRead");

  UR_ASSERT(offset + size <= hBuffer->getSize(), UR_RESULT_ERROR_INVALID_SIZE);

  ur_usm_handle_t_ dstHandle(hContext, size, pDst);

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> lock(this->Mutex,
                                                          hBuffer->getMutex());

  return enqueueGenericCopyUnlocked(
      hBuffer, &dstHandle, blockingRead, offset, 0, size, numEventsInWaitList,
      phEventWaitList, phEvent, UR_COMMAND_MEM_BUFFER_READ);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemBufferWrite(
    ur_mem_handle_t hBuffer, bool blockingWrite, size_t offset, size_t size,
    const void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueMemBufferWrite");

  UR_ASSERT(offset + size <= hBuffer->getSize(), UR_RESULT_ERROR_INVALID_SIZE);

  ur_usm_handle_t_ srcHandle(hContext, size, pSrc);

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> lock(this->Mutex,
                                                          hBuffer->getMutex());

  return enqueueGenericCopyUnlocked(
      &srcHandle, hBuffer, blockingWrite, 0, offset, size, numEventsInWaitList,
      phEventWaitList, phEvent, UR_COMMAND_MEM_BUFFER_WRITE);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueRegionCopyUnlocked(
    ur_mem_handle_t src, ur_mem_handle_t dst, bool blocking,
    ur_rect_offset_t srcOrigin, ur_rect_offset_t dstOrigin,
    ur_rect_region_t region, size_t srcRowPitch, size_t srcSlicePitch,
    size_t dstRowPitch, size_t dstSlicePitch, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent,
    ur_command_t commandType) {
  auto zeParams = ur2zeRegionParams(srcOrigin, dstOrigin, region, srcRowPitch,
                                    dstRowPitch, srcSlicePitch, dstSlicePitch);

  auto signalEvent = getSignalEvent(phEvent, commandType);

  auto waitList = getWaitListView(phEventWaitList, numEventsInWaitList);

  bool memoryMigrated = false;
  auto pSrc = ur_cast<char *>(src->getDevicePtr(
      hDevice, ur_mem_handle_t_::device_access_mode_t::read_only, 0,
      src->getSize(), [&](void *src, void *dst, size_t size) {
        ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                          (handler.commandList.get(), dst, src, size, nullptr,
                           waitList.second, waitList.first));
        memoryMigrated = true;
      }));
  auto pDst = ur_cast<char *>(dst->getDevicePtr(
      hDevice, ur_mem_handle_t_::device_access_mode_t::write_only, 0,
      dst->getSize(), [&](void *src, void *dst, size_t size) {
        ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                          (handler.commandList.get(), dst, src, size, nullptr,
                           waitList.second, waitList.first));
        memoryMigrated = true;
      }));

  if (memoryMigrated) {
    // If memory was migrated, we don't need to pass the wait list to
    // the copy command again.
    waitList.first = nullptr;
    waitList.second = 0;
  }

  auto zeSignalEvent = signalEvent ? signalEvent->getZeEvent() : nullptr;
  ZE2UR_CALL(zeCommandListAppendMemoryCopyRegion,
             (handler.commandList.get(), pDst, &zeParams.dstRegion,
              zeParams.dstPitch, zeParams.dstSlicePitch, pSrc,
              &zeParams.srcRegion, zeParams.srcPitch, zeParams.srcSlicePitch,
              zeSignalEvent, waitList.second, waitList.first));

  if (blocking) {
    ZE2UR_CALL(zeCommandListHostSynchronize,
               (handler.commandList.get(), UINT64_MAX));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemBufferReadRect(
    ur_mem_handle_t hBuffer, bool blockingRead, ur_rect_offset_t bufferOrigin,
    ur_rect_offset_t hostOrigin, ur_rect_region_t region, size_t bufferRowPitch,
    size_t bufferSlicePitch, size_t hostRowPitch, size_t hostSlicePitch,
    void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_in_order_t::enqueueMemBufferReadRect");

  ur_usm_handle_t_ dstHandle(hContext, 0, pDst);

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> lock(this->Mutex,
                                                          hBuffer->getMutex());

  return enqueueRegionCopyUnlocked(
      hBuffer, &dstHandle, blockingRead, bufferOrigin, hostOrigin, region,
      bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch,
      numEventsInWaitList, phEventWaitList, phEvent,
      UR_COMMAND_MEM_BUFFER_READ_RECT);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemBufferWriteRect(
    ur_mem_handle_t hBuffer, bool blockingWrite, ur_rect_offset_t bufferOrigin,
    ur_rect_offset_t hostOrigin, ur_rect_region_t region, size_t bufferRowPitch,
    size_t bufferSlicePitch, size_t hostRowPitch, size_t hostSlicePitch,
    void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_in_order_t::enqueueMemBufferWriteRect");

  ur_usm_handle_t_ srcHandle(hContext, 0, pSrc);

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> lock(this->Mutex,
                                                          hBuffer->getMutex());

  return enqueueRegionCopyUnlocked(
      &srcHandle, hBuffer, blockingWrite, hostOrigin, bufferOrigin, region,
      hostRowPitch, hostSlicePitch, bufferRowPitch, bufferSlicePitch,
      numEventsInWaitList, phEventWaitList, phEvent,
      UR_COMMAND_MEM_BUFFER_WRITE_RECT);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemBufferCopy(
    ur_mem_handle_t hBufferSrc, ur_mem_handle_t hBufferDst, size_t srcOffset,
    size_t dstOffset, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueMemBufferCopy");

  UR_ASSERT(srcOffset + size <= hBufferSrc->getSize(),
            UR_RESULT_ERROR_INVALID_SIZE);
  UR_ASSERT(dstOffset + size <= hBufferDst->getSize(),
            UR_RESULT_ERROR_INVALID_SIZE);

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex, ur_shared_mutex> lock(
      this->Mutex, hBufferSrc->getMutex(), hBufferDst->getMutex());

  return enqueueGenericCopyUnlocked(hBufferSrc, hBufferDst, false, srcOffset,
                                    dstOffset, size, numEventsInWaitList,
                                    phEventWaitList, phEvent,
                                    UR_COMMAND_MEM_BUFFER_COPY);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemBufferCopyRect(
    ur_mem_handle_t hBufferSrc, ur_mem_handle_t hBufferDst,
    ur_rect_offset_t srcOrigin, ur_rect_offset_t dstOrigin,
    ur_rect_region_t region, size_t srcRowPitch, size_t srcSlicePitch,
    size_t dstRowPitch, size_t dstSlicePitch, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_in_order_t::enqueueMemBufferCopyRect");

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex, ur_shared_mutex> lock(
      this->Mutex, hBufferSrc->getMutex(), hBufferDst->getMutex());

  return enqueueRegionCopyUnlocked(
      hBufferSrc, hBufferDst, false, srcOrigin, dstOrigin, region, srcRowPitch,
      srcSlicePitch, dstRowPitch, dstSlicePitch, numEventsInWaitList,
      phEventWaitList, phEvent, UR_COMMAND_MEM_BUFFER_COPY_RECT);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemBufferFill(
    ur_mem_handle_t hBuffer, const void *pPattern, size_t patternSize,
    size_t offset, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueMemBufferFill");

  UR_ASSERT(offset + size <= hBuffer->getSize(), UR_RESULT_ERROR_INVALID_SIZE);

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> lock(this->Mutex,
                                                          hBuffer->getMutex());

  return enqueueGenericFillUnlocked(hBuffer, offset, patternSize, pPattern,
                                    size, numEventsInWaitList, phEventWaitList,
                                    phEvent, UR_COMMAND_MEM_BUFFER_FILL);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemImageRead(
    ur_mem_handle_t hImage, bool blockingRead, ur_rect_offset_t origin,
    ur_rect_region_t region, size_t rowPitch, size_t slicePitch, void *pDst,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
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
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemImageWrite(
    ur_mem_handle_t hImage, bool blockingWrite, ur_rect_offset_t origin,
    ur_rect_region_t region, size_t rowPitch, size_t slicePitch, void *pSrc,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
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
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemImageCopy(
    ur_mem_handle_t hImageSrc, ur_mem_handle_t hImageDst,
    ur_rect_offset_t srcOrigin, ur_rect_offset_t dstOrigin,
    ur_rect_region_t region, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = hImageSrc;
  std::ignore = hImageDst;
  std::ignore = srcOrigin;
  std::ignore = dstOrigin;
  std::ignore = region;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemBufferMap(
    ur_mem_handle_t hBuffer, bool blockingMap, ur_map_flags_t mapFlags,
    size_t offset, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent,
    void **ppRetMap) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueMemBufferMap");

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> lock(this->Mutex,
                                                          hBuffer->getMutex());

  auto signalEvent = getSignalEvent(phEvent, UR_COMMAND_MEM_BUFFER_MAP);

  auto waitList = getWaitListView(phEventWaitList, numEventsInWaitList);

  bool memoryMigrated = false;
  auto pDst = ur_cast<char *>(hBuffer->mapHostPtr(
      mapFlags, offset, size, [&](void *src, void *dst, size_t size) {
        ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                          (handler.commandList.get(), dst, src, size, nullptr,
                           waitList.second, waitList.first));
        memoryMigrated = true;
      }));
  *ppRetMap = pDst;

  if (!memoryMigrated && waitList.second) {
    // If memory was not migrated, we need to wait on the events here.
    ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
               (handler.commandList.get(), waitList.second, waitList.first));
    if (signalEvent) {
      ZE2UR_CALL(zeCommandListAppendSignalEvent,
                 (handler.commandList.get(), signalEvent->getZeEvent()));
    }
  }

  if (blockingMap) {
    ZE2UR_CALL(zeCommandListHostSynchronize,
               (handler.commandList.get(), UINT64_MAX));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemUnmap(
    ur_mem_handle_t hMem, void *pMappedPtr, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueMemUnmap");

  std::scoped_lock<ur_shared_mutex> lock(this->Mutex);

  auto signalEvent = getSignalEvent(phEvent, UR_COMMAND_MEM_UNMAP);

  auto waitList = getWaitListView(phEventWaitList, numEventsInWaitList);

  // TODO: currently unmapHostPtr deallocates memory immediately,
  // since the memory might be used by the user, we need to make sure
  // all dependencies are completed.
  ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
             (handler.commandList.get(), waitList.second, waitList.first));

  bool memoryMigrated = false;
  hMem->unmapHostPtr(pMappedPtr, [&](void *src, void *dst, size_t size) {
    ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                      (handler.commandList.get(), dst, src, size, nullptr,
                       waitList.second, waitList.first));
    memoryMigrated = true;
  });

  if (signalEvent) {
    ZE2UR_CALL(zeCommandListAppendSignalEvent,
               (handler.commandList.get(), signalEvent->getZeEvent()));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueGenericFillUnlocked(
    ur_mem_handle_t dst, size_t offset, size_t patternSize,
    const void *pPattern, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent,
    ur_command_t commandType) {

  auto signalEvent = getSignalEvent(phEvent, commandType);

  auto waitList = getWaitListView(phEventWaitList, numEventsInWaitList);

  bool memoryMigrated = false;
  auto pDst = ur_cast<char *>(dst->getDevicePtr(
      hDevice, ur_mem_handle_t_::device_access_mode_t::read_only, offset, size,
      [&](void *src, void *dst, size_t size) {
        ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                          (handler.commandList.get(), dst, src, size, nullptr,
                           waitList.second, waitList.first));
        memoryMigrated = true;
      }));

  if (memoryMigrated) {
    // If memory was migrated, we don't need to pass the wait list to
    // the copy command again.
    waitList.first = nullptr;
    waitList.second = 0;
  }

  // TODO: support non-power-of-two pattern sizes

  // PatternSize must be a power of two for zeCommandListAppendMemoryFill.
  // When it's not, the fill is emulated with zeCommandListAppendMemoryCopy.
  auto zeSignalEvent = signalEvent ? signalEvent->getZeEvent() : nullptr;
  ZE2UR_CALL(zeCommandListAppendMemoryFill,
             (handler.commandList.get(), pDst, pPattern, patternSize, size,
              zeSignalEvent, waitList.second, waitList.first));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueUSMFill(
    void *pMem, size_t patternSize, const void *pPattern, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueUSMFill");

  std::scoped_lock<ur_shared_mutex> lock(this->Mutex);

  ur_usm_handle_t_ dstHandle(hContext, size, pMem);
  return enqueueGenericFillUnlocked(&dstHandle, 0, patternSize, pPattern, size,
                                    numEventsInWaitList, phEventWaitList,
                                    phEvent, UR_COMMAND_USM_FILL);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueUSMMemcpy(
    bool blocking, void *pDst, const void *pSrc, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  // TODO: parametrize latency tracking with 'blocking'
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueUSMMemcpy");

  std::scoped_lock<ur_shared_mutex> lock(this->Mutex);

  auto signalEvent = getSignalEvent(phEvent, UR_COMMAND_USM_MEMCPY);

  auto [pWaitEvents, numWaitEvents] =
      getWaitListView(phEventWaitList, numEventsInWaitList);

  auto zeSignalEvent = signalEvent ? signalEvent->getZeEvent() : nullptr;
  ZE2UR_CALL(zeCommandListAppendMemoryCopy,
             (handler.commandList.get(), pDst, pSrc, size, zeSignalEvent,
              numWaitEvents, pWaitEvents));

  if (blocking) {
    ZE2UR_CALL(zeCommandListHostSynchronize,
               (handler.commandList.get(), UINT64_MAX));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueUSMPrefetch(
    const void *pMem, size_t size, ur_usm_migration_flags_t flags,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueUSMPrefetch");

  std::ignore = flags;

  std::scoped_lock<ur_shared_mutex> lock(this->Mutex);

  auto signalEvent = getSignalEvent(phEvent, UR_COMMAND_USM_PREFETCH);

  auto [pWaitEvents, numWaitEvents] =
      getWaitListView(phEventWaitList, numEventsInWaitList);

  if (pWaitEvents) {
    ZE2UR_CALL(zeCommandListAppendBarrier, (handler.commandList.get(), nullptr,
                                            numWaitEvents, pWaitEvents));
  }
  // TODO: figure out how to translate "flags"
  ZE2UR_CALL(zeCommandListAppendMemoryPrefetch,
             (handler.commandList.get(), pMem, size));

  if (signalEvent) {
    ZE2UR_CALL(zeCommandListAppendSignalEvent,
               (handler.commandList.get(), signalEvent->getZeEvent()));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t
ur_queue_immediate_in_order_t::enqueueUSMAdvise(const void *pMem, size_t size,
                                                ur_usm_advice_flags_t advice,
                                                ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueUSMAdvise");

  std::ignore = flags;

  std::scoped_lock<ur_shared_mutex> lock(this->Mutex);

  auto zeAdvice = ur_cast<ze_memory_advice_t>(advice);

  auto signalEvent = getSignalEvent(phEvent, UR_COMMAND_USM_ADVISE);

  auto [pWaitEvents, numWaitEvents] = getWaitListView(nullptr, 0);

  if (pWaitEvents) {
    ZE2UR_CALL(zeCommandListAppendBarrier, (handler.commandList.get(), nullptr,
                                            numWaitEvents, pWaitEvents));
  }

  // TODO: figure out how to translate "flags"
  ZE2UR_CALL(zeCommandListAppendMemAdvise,
             (handler.commandList.get(), this->hDevice->ZeDevice, pMem, size,
              zeAdvice));

  if (signalEvent) {
    ZE2UR_CALL(zeCommandListAppendSignalEvent,
               (handler.commandList.get(), signalEvent->getZeEvent()));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueUSMFill2D(
    void *pMem, size_t pitch, size_t patternSize, const void *pPattern,
    size_t width, size_t height, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = pMem;
  std::ignore = pitch;
  std::ignore = patternSize;
  std::ignore = pPattern;
  std::ignore = width;
  std::ignore = height;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueUSMMemcpy2D(
    bool blocking, void *pDst, size_t dstPitch, const void *pSrc,
    size_t srcPitch, size_t width, size_t height, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
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
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

static void *getGlobalPointerFromModule(ze_module_handle_t hModule,
                                        size_t offset, size_t count,
                                        const char *name) {
  // Find global variable pointer
  size_t globalVarSize = 0;
  void *globalVarPtr = nullptr;
  ZE2UR_CALL_THROWS(zeModuleGetGlobalPointer,
                    (hModule, name, &globalVarSize, &globalVarPtr));
  if (globalVarSize < offset + count) {
    setErrorMessage("Write device global variable is out of range.",
                    UR_RESULT_ERROR_INVALID_VALUE,
                    static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_ARGUMENT));
    throw UR_RESULT_ERROR_ADAPTER_SPECIFIC;
  }
  return globalVarPtr;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueDeviceGlobalVariableWrite(
    ur_program_handle_t hProgram, const char *name, bool blockingWrite,
    size_t count, size_t offset, const void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_in_order_t::enqueueDeviceGlobalVariableWrite");

  // TODO: make getZeModuleHandle thread-safe
  ze_module_handle_t zeModule =
      hProgram->getZeModuleHandle(this->hDevice->ZeDevice);

  // Find global variable pointer
  auto globalVarPtr = getGlobalPointerFromModule(zeModule, offset, count, name);

  // Locking is done inside enqueueUSMMemcpy
  return enqueueUSMMemcpy(blockingWrite, ur_cast<char *>(globalVarPtr) + offset,
                          pSrc, count, numEventsInWaitList, phEventWaitList,
                          phEvent);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueDeviceGlobalVariableRead(
    ur_program_handle_t hProgram, const char *name, bool blockingRead,
    size_t count, size_t offset, void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_in_order_t::enqueueDeviceGlobalVariableRead");

  // TODO: make getZeModuleHandle thread-safe
  ze_module_handle_t zeModule =
      hProgram->getZeModuleHandle(this->hDevice->ZeDevice);

  // Find global variable pointer
  auto globalVarPtr = getGlobalPointerFromModule(zeModule, offset, count, name);

  // Locking is done inside enqueueUSMMemcpy
  return enqueueUSMMemcpy(blockingRead, pDst,
                          ur_cast<char *>(globalVarPtr) + offset, count,
                          numEventsInWaitList, phEventWaitList, phEvent);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueReadHostPipe(
    ur_program_handle_t hProgram, const char *pipe_symbol, bool blocking,
    void *pDst, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = hProgram;
  std::ignore = pipe_symbol;
  std::ignore = blocking;
  std::ignore = pDst;
  std::ignore = size;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueWriteHostPipe(
    ur_program_handle_t hProgram, const char *pipe_symbol, bool blocking,
    void *pSrc, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = hProgram;
  std::ignore = pipe_symbol;
  std::ignore = blocking;
  std::ignore = pSrc;
  std::ignore = size;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t ur_queue_immediate_in_order_t::bindlessImagesImageCopyExp(
    const void *pSrc, void *pDst, const ur_image_desc_t *pSrcImageDesc,
    const ur_image_desc_t *pDstImageDesc,
    const ur_image_format_t *pSrcImageFormat,
    const ur_image_format_t *pDstImageFormat,
    ur_exp_image_copy_region_t *pCopyRegion,
    ur_exp_image_copy_flags_t imageCopyFlags, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = pDst;
  std::ignore = pSrc;
  std::ignore = pSrcImageDesc;
  std::ignore = pDstImageDesc;
  std::ignore = imageCopyFlags;
  std::ignore = pSrcImageFormat;
  std::ignore = pDstImageFormat;
  std::ignore = pCopyRegion;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t
ur_queue_immediate_in_order_t::bindlessImagesWaitExternalSemaphoreExp(
    ur_exp_external_semaphore_handle_t hSemaphore, bool hasWaitValue,
    uint64_t waitValue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = hSemaphore;
  std::ignore = hasWaitValue;
  std::ignore = waitValue;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t
ur_queue_immediate_in_order_t::bindlessImagesSignalExternalSemaphoreExp(
    ur_exp_external_semaphore_handle_t hSemaphore, bool hasSignalValue,
    uint64_t signalValue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = hSemaphore;
  std::ignore = hasSignalValue;
  std::ignore = signalValue;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueCooperativeKernelLaunchExp(
    ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_in_order_t::enqueueCooperativeKernelLaunchExp");

  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hKernel->getProgramHandle(), UR_RESULT_ERROR_INVALID_NULL_POINTER);

  UR_ASSERT(workDim > 0, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);
  UR_ASSERT(workDim < 4, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);

  ze_kernel_handle_t hZeKernel = hKernel->getZeHandle(hDevice);

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(this->Mutex,
                                                          hKernel->Mutex);

  ze_group_count_t zeThreadGroupDimensions{1, 1, 1};
  uint32_t WG[3]{};
  UR_CALL(calculateKernelWorkDimensions(hZeKernel, hDevice,
                                        zeThreadGroupDimensions, WG, workDim,
                                        pGlobalWorkSize, pLocalWorkSize));

  auto signalEvent = getSignalEvent(phEvent, UR_COMMAND_KERNEL_LAUNCH);

  auto waitList = getWaitListView(phEventWaitList, numEventsInWaitList);

  bool memoryMigrated = false;
  auto memoryMigrate = [&](void *src, void *dst, size_t size) {
    ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                      (handler.commandList.get(), dst, src, size, nullptr,
                       waitList.second, waitList.first));
    memoryMigrated = true;
  };

  UR_CALL(hKernel->prepareForSubmission(hContext, hDevice, pGlobalWorkOffset,
                                        workDim, WG[0], WG[1], WG[2],
                                        memoryMigrate));

  if (memoryMigrated) {
    // If memory was migrated, we don't need to pass the wait list to
    // the copy command again.
    waitList.first = nullptr;
    waitList.second = 0;
  }

  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::"
                      "zeCommandListAppendLaunchCooperativeKernel");
  auto zeSignalEvent = signalEvent ? signalEvent->getZeEvent() : nullptr;
  ZE2UR_CALL(zeCommandListAppendLaunchCooperativeKernel,
             (handler.commandList.get(), hZeKernel, &zeThreadGroupDimensions,
              zeSignalEvent, waitList.second, waitList.first));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueTimestampRecordingExp(
    bool blocking, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_in_order_t::enqueueTimestampRecordingExp");

  std::scoped_lock<ur_shared_mutex> lock(this->Mutex);

  auto signalEvent =
      getSignalEvent(phEvent, UR_COMMAND_TIMESTAMP_RECORDING_EXP);

  if (!signalEvent) {
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  }

  auto [pWaitEvents, numWaitEvents] =
      getWaitListView(phEventWaitList, numEventsInWaitList);

  signalEvent->recordStartTimestamp();

  auto [timestampPtr, zeSignalEvent] =
      signalEvent->getEventEndTimestampAndHandle();

  ZE2UR_CALL(zeCommandListAppendWriteGlobalTimestamp,
             (handler.commandList.get(), timestampPtr, zeSignalEvent,
              numWaitEvents, pWaitEvents));

  if (blocking) {
    ZE2UR_CALL(zeCommandListHostSynchronize,
               (handler.commandList.get(), UINT64_MAX));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueKernelLaunchCustomExp(
    ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numPropsInLaunchPropList,
    const ur_exp_launch_property_t *launchPropList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  std::ignore = hKernel;
  std::ignore = workDim;
  std::ignore = pGlobalWorkOffset;
  std::ignore = pGlobalWorkSize;
  std::ignore = pLocalWorkSize;
  std::ignore = numPropsInLaunchPropList;
  std::ignore = launchPropList;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueNativeCommandExp(
    ur_exp_enqueue_native_command_function_t, void *, uint32_t,
    const ur_mem_handle_t *, const ur_exp_enqueue_native_command_properties_t *,
    uint32_t, const ur_event_handle_t *, ur_event_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
} // namespace v2
