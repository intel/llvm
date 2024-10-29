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
    const ur_event_handle_t *phWaitEvents, uint32_t numWaitEvents,
    ur_command_list_handler_t *pHandler) {
  auto extraWaitEvent = (lastHandler && pHandler != lastHandler)
                            ? lastHandler->lastEvent->getZeEvent()
                            : nullptr;

  auto totalEvents = numWaitEvents + (extraWaitEvent != nullptr);
  waitList.reserve(totalEvents);

  for (uint32_t i = 0; i < numWaitEvents; i++) {
    waitList[i] = phWaitEvents[i]->getZeEvent();
  }

  if (extraWaitEvent) {
    waitList[numWaitEvents] = extraWaitEvent;
  }

  return {waitList.data(), static_cast<uint32_t>(totalEvents)};
}

static int32_t getZeOrdinal(ur_device_handle_t hDevice, queue_group_type type) {
  if (type == queue_group_type::MainCopy && hDevice->hasMainCopyEngine()) {
    return hDevice->QueueGroup[queue_group_type::MainCopy].ZeOrdinal;
  }
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
    const ur_queue_properties_t *pProps, queue_group_type type,
    event_pool *eventPool)
    : commandList(hContext->commandListCache.getImmediateCommandList(
          hDevice->ZeDevice, true, getZeOrdinal(hDevice, type),
          ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
          getZePriority(pProps ? pProps->flags : ur_queue_flags_t{}),
          getZeIndex(pProps))),
      internalEvent(eventPool->allocate(), [=](ur_event_handle_t event) {
        ur::level_zero::urEventRelease(event);
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
      copyHandler(hContext, hDevice, pProps, queue_group_type::MainCopy,
                  eventPool.get()),
      computeHandler(hContext, hDevice, pProps, queue_group_type::Compute,
                     eventPool.get()) {}

ur_command_list_handler_t *
ur_queue_immediate_in_order_t::getCommandListHandlerForCompute() {
  return &computeHandler;
}

ur_command_list_handler_t *
ur_queue_immediate_in_order_t::getCommandListHandlerForCopy() {
  // TODO: optimize for specific devices, see ../memory.cpp
  return &copyHandler;
}

ur_command_list_handler_t *
ur_queue_immediate_in_order_t::getCommandListHandlerForFill(
    size_t patternSize) {
  if (patternSize <= hDevice->QueueGroup[queue_group_type::MainCopy]
                         .ZeProperties.maxMemoryFillPatternSize)
    return &copyHandler;
  else
    return &computeHandler;
}

ur_event_handle_t ur_queue_immediate_in_order_t::getSignalEvent(
    ur_command_list_handler_t *handler, ur_event_handle_t *hUserEvent) {
  if (!hUserEvent) {
    handler->lastEvent = handler->internalEvent.get();
  } else {
    *hUserEvent = eventPool->allocate();
    handler->lastEvent = *hUserEvent;
  }

  return handler->lastEvent;
}

ur_result_t
ur_queue_immediate_in_order_t::queueGetInfo(ur_queue_info_t propName,
                                            size_t propSize, void *pPropValue,
                                            size_t *pPropSizeRet) {
  std::shared_lock<ur_shared_mutex> Lock(Mutex);
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
    // We can exit early if we have in-order queue.
    if (!lastHandler)
      return ReturnValue(true);
    [[fallthrough]];
  }
  default:
    logger::error(
        "Unsupported ParamName in urQueueGetInfo: ParamName=ParamName={}(0x{})",
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

  delete this;
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::queueGetNativeHandle(
    ur_queue_native_desc_t *pDesc, ur_native_handle_t *phNativeQueue) {
  std::ignore = pDesc;
  std::ignore = phNativeQueue;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t ur_queue_immediate_in_order_t::finalizeHandler(
    ur_command_list_handler_t *handler) {
  lastHandler = handler;
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::finalizeHandler(
    ur_command_list_handler_t *handler, bool blocking) {
  if (blocking) {
    ZE2UR_CALL(zeCommandListHostSynchronize,
               (handler->commandList.get(), UINT64_MAX));
    lastHandler = nullptr;
  } else {
    finalizeHandler(handler);
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::queueFinish() {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::queueFinish");
  std::unique_lock<ur_shared_mutex> lock(this->Mutex);

  if (!lastHandler) {
    return UR_RESULT_SUCCESS;
  }

  auto lastCmdList = lastHandler->commandList.get();
  lastHandler = nullptr;
  lock.unlock();

  // TODO: use zeEventHostSynchronize instead?
  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_in_order_t::zeCommandListHostSynchronize");
  ZE2UR_CALL(zeCommandListHostSynchronize, (lastCmdList, UINT64_MAX));

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

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex, ur_shared_mutex> Lock(
      hKernel->Mutex, hKernel->getProgramHandle()->Mutex, this->Mutex);

  if (pGlobalWorkOffset != NULL) {
    UR_CALL(
        setKernelGlobalOffset(hContext, hZeKernel, workDim, pGlobalWorkOffset));
  }

  ze_group_count_t zeThreadGroupDimensions{1, 1, 1};
  uint32_t WG[3];
  UR_CALL(calculateKernelWorkDimensions(hZeKernel, hDevice,
                                        zeThreadGroupDimensions, WG, workDim,
                                        pGlobalWorkSize, pLocalWorkSize));

  ZE2UR_CALL(zeKernelSetGroupSize, (hZeKernel, WG[0], WG[1], WG[2]));

  auto handler = getCommandListHandlerForCompute();
  auto signalEvent = getSignalEvent(handler, phEvent);

  auto [pWaitEvents, numWaitEvents] =
      getWaitListView(phEventWaitList, numEventsInWaitList, handler);

  // TODO: consider migrating memory to the device if memory buffers are used

  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_in_order_t::zeCommandListAppendLaunchKernel");
  ZE2UR_CALL(zeCommandListAppendLaunchKernel,
             (handler->commandList.get(), hZeKernel, &zeThreadGroupDimensions,
              signalEvent->getZeEvent(), numWaitEvents, pWaitEvents));

  return finalizeHandler(handler);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueEventsWait(
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueEventsWait");

  std::unique_lock<ur_shared_mutex> lock(this->Mutex);

  auto handler = getCommandListHandlerForCompute();
  auto signalEvent = getSignalEvent(handler, phEvent);
  auto [pWaitEvents, numWaitEvents] =
      getWaitListView(phEventWaitList, numEventsInWaitList, handler);

  ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
             (handler->commandList.get(), numWaitEvents, pWaitEvents));
  ZE2UR_CALL(zeCommandListAppendSignalEvent,
             (handler->commandList.get(), signalEvent->getZeEvent()));

  return finalizeHandler(handler);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueEventsWaitWithBarrier(
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  // For in-order queue we don't need a real barrier, just wait for
  // requested events in potentially different queues and add a "barrier"
  // event signal because it is already guaranteed that previous commands
  // in this queue are completed when the signal is started.
  return enqueueEventsWait(numEventsInWaitList, phEventWaitList, phEvent);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueGenericCopyUnlocked(
    ur_mem_handle_t src, ur_mem_handle_t dst, bool blocking, size_t srcOffset,
    size_t dstOffset, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  auto handler = getCommandListHandlerForCopy();
  auto signalEvent = getSignalEvent(handler, phEvent);

  auto waitList =
      getWaitListView(phEventWaitList, numEventsInWaitList, handler);

  bool memoryMigrated = false;
  auto pSrc = ur_cast<char *>(src->getDevicePtr(
      hDevice, ur_mem_handle_t_::access_mode_t::read_only, srcOffset, size,
      [&](void *src, void *dst, size_t size) {
        ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                          (handler->commandList.get(), dst, src, size, nullptr,
                           waitList.second, waitList.first));
        memoryMigrated = true;
      }));

  auto pDst = ur_cast<char *>(dst->getDevicePtr(
      hDevice, ur_mem_handle_t_::access_mode_t::write_only, dstOffset, size,
      [&](void *src, void *dst, size_t size) {
        ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                          (handler->commandList.get(), dst, src, size, nullptr,
                           waitList.second, waitList.first));
        memoryMigrated = true;
      }));

  if (memoryMigrated) {
    // If memory was migrated, we don't need to pass the wait list to
    // the copy command again.
    waitList.first = nullptr;
    waitList.second = 0;
  }

  ZE2UR_CALL(zeCommandListAppendMemoryCopy,
             (handler->commandList.get(), pDst, pSrc, size,
              signalEvent->getZeEvent(), waitList.second, waitList.first));

  return finalizeHandler(handler, blocking);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemBufferRead(
    ur_mem_handle_t hBuffer, bool blockingRead, size_t offset, size_t size,
    void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueMemBufferRead");

  UR_ASSERT(offset + size <= hBuffer->getSize(), UR_RESULT_ERROR_INVALID_SIZE);

  std::scoped_lock<ur_shared_mutex> Lock(this->Mutex);

  ur_usm_handle_t_ dstHandle(hContext, size, pDst);
  return enqueueGenericCopyUnlocked(hBuffer, &dstHandle, blockingRead, offset,
                                    0, size, numEventsInWaitList,
                                    phEventWaitList, phEvent);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemBufferWrite(
    ur_mem_handle_t hBuffer, bool blockingWrite, size_t offset, size_t size,
    const void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueMemBufferWrite");

  UR_ASSERT(offset + size <= hBuffer->getSize(), UR_RESULT_ERROR_INVALID_SIZE);

  std::scoped_lock<ur_shared_mutex> Lock(this->Mutex);

  ur_usm_handle_t_ srcHandle(hContext, size, pSrc);
  return enqueueGenericCopyUnlocked(&srcHandle, hBuffer, blockingWrite, 0,
                                    offset, size, numEventsInWaitList,
                                    phEventWaitList, phEvent);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueRegionCopyUnlocked(
    ur_mem_handle_t src, ur_mem_handle_t dst, bool blocking,
    ur_rect_offset_t srcOrigin, ur_rect_offset_t dstOrigin,
    ur_rect_region_t region, size_t srcRowPitch, size_t srcSlicePitch,
    size_t dstRowPitch, size_t dstSlicePitch, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  auto zeParams = ur2zeRegionParams(srcOrigin, dstOrigin, region, srcRowPitch,
                                    dstRowPitch, srcSlicePitch, dstSlicePitch);

  auto handler = getCommandListHandlerForCopy();
  auto signalEvent = getSignalEvent(handler, phEvent);

  auto waitList =
      getWaitListView(phEventWaitList, numEventsInWaitList, handler);

  bool memoryMigrated = false;
  auto pSrc = ur_cast<char *>(src->getDevicePtr(
      hDevice, ur_mem_handle_t_::access_mode_t::read_only, 0, src->getSize(),
      [&](void *src, void *dst, size_t size) {
        ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                          (handler->commandList.get(), dst, src, size, nullptr,
                           waitList.second, waitList.first));
        memoryMigrated = true;
      }));
  auto pDst = ur_cast<char *>(dst->getDevicePtr(
      hDevice, ur_mem_handle_t_::access_mode_t::write_only, 0, dst->getSize(),
      [&](void *src, void *dst, size_t size) {
        ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                          (handler->commandList.get(), dst, src, size, nullptr,
                           waitList.second, waitList.first));
        memoryMigrated = true;
      }));

  if (memoryMigrated) {
    // If memory was migrated, we don't need to pass the wait list to
    // the copy command again.
    waitList.first = nullptr;
    waitList.second = 0;
  }

  ZE2UR_CALL(zeCommandListAppendMemoryCopyRegion,
             (handler->commandList.get(), pDst, &zeParams.dstRegion,
              zeParams.dstPitch, zeParams.dstSlicePitch, pSrc,
              &zeParams.srcRegion, zeParams.srcPitch, zeParams.srcSlicePitch,
              signalEvent->getZeEvent(), waitList.second, waitList.first));

  return finalizeHandler(handler, blocking);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemBufferReadRect(
    ur_mem_handle_t hBuffer, bool blockingRead, ur_rect_offset_t bufferOrigin,
    ur_rect_offset_t hostOrigin, ur_rect_region_t region, size_t bufferRowPitch,
    size_t bufferSlicePitch, size_t hostRowPitch, size_t hostSlicePitch,
    void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_in_order_t::enqueueMemBufferReadRect");

  std::scoped_lock<ur_shared_mutex> Lock(this->Mutex);

  ur_usm_handle_t_ dstHandle(hContext, 0, pDst);
  return enqueueRegionCopyUnlocked(
      hBuffer, &dstHandle, blockingRead, bufferOrigin, hostOrigin, region,
      bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch,
      numEventsInWaitList, phEventWaitList, phEvent);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemBufferWriteRect(
    ur_mem_handle_t hBuffer, bool blockingWrite, ur_rect_offset_t bufferOrigin,
    ur_rect_offset_t hostOrigin, ur_rect_region_t region, size_t bufferRowPitch,
    size_t bufferSlicePitch, size_t hostRowPitch, size_t hostSlicePitch,
    void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_in_order_t::enqueueMemBufferWriteRect");

  std::scoped_lock<ur_shared_mutex> Lock(this->Mutex);

  ur_usm_handle_t_ srcHandle(hContext, 0, pSrc);
  return enqueueRegionCopyUnlocked(
      &srcHandle, hBuffer, blockingWrite, hostOrigin, bufferOrigin, region,
      hostRowPitch, hostSlicePitch, bufferRowPitch, bufferSlicePitch,
      numEventsInWaitList, phEventWaitList, phEvent);
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

  std::scoped_lock<ur_shared_mutex> Lock(this->Mutex);

  return enqueueGenericCopyUnlocked(hBufferSrc, hBufferDst, false, srcOffset,
                                    dstOffset, size, numEventsInWaitList,
                                    phEventWaitList, phEvent);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemBufferCopyRect(
    ur_mem_handle_t hBufferSrc, ur_mem_handle_t hBufferDst,
    ur_rect_offset_t srcOrigin, ur_rect_offset_t dstOrigin,
    ur_rect_region_t region, size_t srcRowPitch, size_t srcSlicePitch,
    size_t dstRowPitch, size_t dstSlicePitch, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_in_order_t::enqueueMemBufferCopyRect");

  std::scoped_lock<ur_shared_mutex> Lock(this->Mutex);

  return enqueueRegionCopyUnlocked(
      hBufferSrc, hBufferDst, false, srcOrigin, dstOrigin, region, srcRowPitch,
      srcSlicePitch, dstRowPitch, dstSlicePitch, numEventsInWaitList,
      phEventWaitList, phEvent);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemBufferFill(
    ur_mem_handle_t hBuffer, const void *pPattern, size_t patternSize,
    size_t offset, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueMemBufferFill");

  UR_ASSERT(offset + size <= hBuffer->getSize(), UR_RESULT_ERROR_INVALID_SIZE);

  std::scoped_lock<ur_shared_mutex> Lock(this->Mutex);

  return enqueueGenericFillUnlocked(hBuffer, offset, patternSize, pPattern,
                                    size, numEventsInWaitList, phEventWaitList,
                                    phEvent);
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

static ur_mem_handle_t_::access_mode_t getAccessMode(ur_map_flags_t mapFlags) {
  if (mapFlags & UR_MAP_FLAG_WRITE_INVALIDATE_REGION) {
    return ur_mem_handle_t_::access_mode_t::write_invalidate;
  } else if ((mapFlags & UR_MAP_FLAG_READ) && (mapFlags & UR_MAP_FLAG_WRITE)) {
    return ur_mem_handle_t_::access_mode_t::read_write;
  } else if (mapFlags & UR_MAP_FLAG_READ) {
    return ur_mem_handle_t_::access_mode_t::read_only;
  } else if (mapFlags & UR_MAP_FLAG_WRITE) {
    return ur_mem_handle_t_::access_mode_t::write_only;
  } else {
    throw UR_RESULT_ERROR_INVALID_VALUE;
  }
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemBufferMap(
    ur_mem_handle_t hBuffer, bool blockingMap, ur_map_flags_t mapFlags,
    size_t offset, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent,
    void **ppRetMap) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueMemBufferMap");

  ur_mem_handle_t_::access_mode_t accessMode = getAccessMode(mapFlags);

  std::scoped_lock<ur_shared_mutex> Lock(this->Mutex);

  auto handler = getCommandListHandlerForCopy();
  auto signalEvent = getSignalEvent(handler, phEvent);

  auto waitList =
      getWaitListView(phEventWaitList, numEventsInWaitList, handler);

  bool memoryMigrated = false;
  auto pDst = ur_cast<char *>(hBuffer->mapHostPtr(
      accessMode, offset, size, [&](void *src, void *dst, size_t size) {
        ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                          (handler->commandList.get(), dst, src, size, nullptr,
                           waitList.second, waitList.first));
        memoryMigrated = true;
      }));
  *ppRetMap = pDst;

  if (!memoryMigrated && waitList.second) {
    // If memory was not migrated, we need to wait on the events here.
    ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
               (handler->commandList.get(), waitList.second, waitList.first));
    if (signalEvent) {
      ZE2UR_CALL(zeCommandListAppendSignalEvent,
                 (handler->commandList.get(), signalEvent->getZeEvent()));
    }
  }

  return finalizeHandler(handler, blockingMap);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemUnmap(
    ur_mem_handle_t hMem, void *pMappedPtr, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueMemUnmap");

  std::scoped_lock<ur_shared_mutex> Lock(this->Mutex);

  auto handler = getCommandListHandlerForCopy();
  auto signalEvent = getSignalEvent(handler, phEvent);

  auto waitList =
      getWaitListView(phEventWaitList, numEventsInWaitList, handler);

  // TODO: currently unmapHostPtr deallocates memory immediately,
  // since the memory might be used by the user, we need to make sure
  // all dependencies are completed.
  ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
             (handler->commandList.get(), waitList.second, waitList.first));

  bool memoryMigrated = false;
  hMem->unmapHostPtr(pMappedPtr, [&](void *src, void *dst, size_t size) {
    ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                      (handler->commandList.get(), dst, src, size, nullptr,
                       waitList.second, waitList.first));
    memoryMigrated = true;
  });

  if (signalEvent) {
    ZE2UR_CALL(zeCommandListAppendSignalEvent,
               (handler->commandList.get(), signalEvent->getZeEvent()));
  }

  return finalizeHandler(handler);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueGenericFillUnlocked(
    ur_mem_handle_t dst, size_t offset, size_t patternSize,
    const void *pPattern, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {

  auto handler = getCommandListHandlerForFill(patternSize);
  auto signalEvent = getSignalEvent(handler, phEvent);

  auto waitList =
      getWaitListView(phEventWaitList, numEventsInWaitList, handler);

  bool memoryMigrated = false;
  auto pDst = ur_cast<char *>(dst->getDevicePtr(
      hDevice, ur_mem_handle_t_::access_mode_t::read_only, offset, size,
      [&](void *src, void *dst, size_t size) {
        ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                          (handler->commandList.get(), dst, src, size, nullptr,
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
  ZE2UR_CALL(zeCommandListAppendMemoryFill,
             (handler->commandList.get(), pDst, pPattern, patternSize, size,
              signalEvent->getZeEvent(), waitList.second, waitList.first));

  return finalizeHandler(handler);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueUSMFill(
    void *pMem, size_t patternSize, const void *pPattern, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueUSMFill");

  std::scoped_lock<ur_shared_mutex> Lock(this->Mutex);

  ur_usm_handle_t_ dstHandle(hContext, size, pMem);
  return enqueueGenericFillUnlocked(&dstHandle, 0, patternSize, pPattern, size,
                                    numEventsInWaitList, phEventWaitList,
                                    phEvent);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueUSMMemcpy(
    bool blocking, void *pDst, const void *pSrc, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  // TODO: parametrize latency tracking with 'blocking'
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueUSMMemcpy");

  std::scoped_lock<ur_shared_mutex> Lock(this->Mutex);

  auto handler = getCommandListHandlerForCopy();
  auto signalEvent = getSignalEvent(handler, phEvent);

  auto [pWaitEvents, numWaitEvents] =
      getWaitListView(phEventWaitList, numEventsInWaitList, handler);

  ZE2UR_CALL(zeCommandListAppendMemoryCopy,
             (handler->commandList.get(), pDst, pSrc, size,
              signalEvent->getZeEvent(), numWaitEvents, pWaitEvents));

  return finalizeHandler(handler, blocking);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueUSMPrefetch(
    const void *pMem, size_t size, ur_usm_migration_flags_t flags,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  std::ignore = flags;

  std::scoped_lock<ur_shared_mutex> Lock(this->Mutex);

  auto handler = getCommandListHandlerForCompute();
  auto signalEvent = getSignalEvent(handler, phEvent);

  auto [pWaitEvents, numWaitEvents] =
      getWaitListView(phEventWaitList, numEventsInWaitList, handler);

  if (pWaitEvents) {
    ZE2UR_CALL(zeCommandListAppendBarrier, (handler->commandList.get(), nullptr,
                                            numWaitEvents, pWaitEvents));
  }
  // TODO: figure out how to translate "flags"
  ZE2UR_CALL(zeCommandListAppendMemoryPrefetch,
             (handler->commandList.get(), pMem, size));
  ZE2UR_CALL(zeCommandListAppendSignalEvent,
             (handler->commandList.get(), signalEvent->getZeEvent()));

  return finalizeHandler(handler);
}

ur_result_t
ur_queue_immediate_in_order_t::enqueueUSMAdvise(const void *pMem, size_t size,
                                                ur_usm_advice_flags_t advice,
                                                ur_event_handle_t *phEvent) {
  std::ignore = flags;

  auto zeAdvice = ur_cast<ze_memory_advice_t>(advice);

  std::scoped_lock<ur_shared_mutex> Lock(this->Mutex);

  auto handler = getCommandListHandlerForCompute();
  auto signalEvent = getSignalEvent(handler, phEvent);

  auto [pWaitEvents, numWaitEvents] = getWaitListView(nullptr, 0, handler);

  if (pWaitEvents) {
    ZE2UR_CALL(zeCommandListAppendBarrier, (handler->commandList.get(), nullptr,
                                            numWaitEvents, pWaitEvents));
  }

  // TODO: figure out how to translate "flags"
  ZE2UR_CALL(zeCommandListAppendMemAdvise,
             (handler->commandList.get(), this->hDevice->ZeDevice, pMem, size,
              zeAdvice));
  ZE2UR_CALL(zeCommandListAppendSignalEvent,
             (handler->commandList.get(), signalEvent->getZeEvent()));

  return finalizeHandler(handler);
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
  // TODO: make getZeModuleHandle thread-safe
  ze_module_handle_t zeModule =
      hProgram->getZeModuleHandle(this->hDevice->ZeDevice);

  // Find global variable pointer
  auto globalVarPtr = getGlobalPointerFromModule(zeModule, offset, count, name);

  return enqueueUSMMemcpy(blockingWrite, ur_cast<char *>(globalVarPtr) + offset,
                          pSrc, count, numEventsInWaitList, phEventWaitList,
                          phEvent);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueDeviceGlobalVariableRead(
    ur_program_handle_t hProgram, const char *name, bool blockingRead,
    size_t count, size_t offset, void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  // TODO: make getZeModuleHandle thread-safe
  ze_module_handle_t zeModule =
      hProgram->getZeModuleHandle(this->hDevice->ZeDevice);

  // Find global variable pointer
  auto globalVarPtr = getGlobalPointerFromModule(zeModule, offset, count, name);

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
  std::ignore = hKernel;
  std::ignore = workDim;
  std::ignore = pGlobalWorkOffset;
  std::ignore = pGlobalWorkSize;
  std::ignore = pLocalWorkSize;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueTimestampRecordingExp(
    bool blocking, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::scoped_lock<ur_shared_mutex> lock(this->Mutex);

  auto handler = getCommandListHandlerForCompute();
  auto signalEvent = getSignalEvent(handler, phEvent);

  if (!signalEvent) {
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  }

  auto [pWaitEvents, numWaitEvents] =
      getWaitListView(phEventWaitList, numEventsInWaitList, handler);

  signalEvent->recordStartTimestamp();

  ZE2UR_CALL(zeCommandListAppendWriteGlobalTimestamp,
             (handler->commandList.get(),
              signalEvent->getEventEndTimestampPtr(), signalEvent->getZeEvent(),
              numWaitEvents, pWaitEvents));

  return finalizeHandler(handler, blocking);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueKernelLaunchCustomExp(
    ur_kernel_handle_t hKernel, uint32_t workDim, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numPropsInLaunchPropList,
    const ur_exp_launch_property_t *launchPropList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  std::ignore = hKernel;
  std::ignore = workDim;
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
