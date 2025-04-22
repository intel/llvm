//===--------- command_list_manager.cpp - Level Zero Adapter --------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "command_list_manager.hpp"
#include "../helpers/kernel_helpers.hpp"
#include "../helpers/memory_helpers.hpp"
#include "../ur_interface_loader.hpp"
#include "context.hpp"
#include "kernel.hpp"

ur_command_list_manager::ur_command_list_manager(
    ur_context_handle_t context, ur_device_handle_t device,
    v2::raii::command_list_unique_handle &&commandList, v2::event_flags_t flags,
    ur_queue_t_ *queue, bool isImmediateCommandList)
    : context(context), device(device), zeCommandList(std::move(commandList)),
      queue(queue), isImmediateCommandList(isImmediateCommandList) {
  auto &eventPoolTmp = isImmediateCommandList
                           ? context->getEventPoolCacheImmediate()
                           : context->getEventPoolCacheRegular();
  eventPool = eventPoolTmp.borrow(device->Id.value(), flags);
  UR_CALL_THROWS(ur::level_zero::urContextRetain(context));
  UR_CALL_THROWS(ur::level_zero::urDeviceRetain(device));
}
ur_command_list_manager::~ur_command_list_manager() {
  ur::level_zero::urContextRelease(context);
  ur::level_zero::urDeviceRelease(device);
}

ur_result_t ur_command_list_manager::appendGenericFillUnlocked(
    ur_mem_buffer_t *dst, size_t offset, size_t patternSize,
    const void *pPattern, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent,
    ur_command_t commandType) {

  auto zeSignalEvent = getSignalEvent(phEvent, commandType);

  auto waitListView = getWaitListView(phEventWaitList, numEventsInWaitList);

  auto pDst = ur_cast<char *>(dst->getDevicePtr(
      device, ur_mem_buffer_t::device_access_mode_t::read_only, offset, size,
      [&](void *src, void *dst, size_t size) {
        ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                          (zeCommandList.get(), dst, src, size, nullptr,
                           waitListView.num, waitListView.handles));
        waitListView.clear();
      }));

  // PatternSize must be a power of two for zeCommandListAppendMemoryFill.
  // When it's not, the fill is emulated with zeCommandListAppendMemoryCopy.
  if (isPowerOf2(patternSize)) {
    ZE2UR_CALL(zeCommandListAppendMemoryFill,
               (zeCommandList.get(), pDst, pPattern, patternSize, size,
                zeSignalEvent, waitListView.num, waitListView.handles));
  } else {
    // Copy pattern into every entry in memory array pointed by Ptr.
    uint32_t numOfCopySteps = size / patternSize;
    const void *src = pPattern;

    for (uint32_t step = 0; step < numOfCopySteps; ++step) {
      void *dst = reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(pDst) +
                                           step * patternSize);
      ZE2UR_CALL(zeCommandListAppendMemoryCopy,
                 (zeCommandList.get(), dst, src, patternSize,
                  step == numOfCopySteps - 1 ? zeSignalEvent : nullptr,
                  waitListView.num, waitListView.handles));
      waitListView.clear();
    }
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::appendGenericCopyUnlocked(
    ur_mem_buffer_t *src, ur_mem_buffer_t *dst, bool blocking, size_t srcOffset,
    size_t dstOffset, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent,
    ur_command_t commandType) {
  auto zeSignalEvent = getSignalEvent(phEvent, commandType);

  auto waitListView = getWaitListView(phEventWaitList, numEventsInWaitList);

  auto pSrc = ur_cast<char *>(src->getDevicePtr(
      device, ur_mem_buffer_t::device_access_mode_t::read_only, srcOffset, size,
      [&](void *src, void *dst, size_t size) {
        ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                          (zeCommandList.get(), dst, src, size, nullptr,
                           waitListView.num, waitListView.handles));
        waitListView.clear();
      }));

  auto pDst = ur_cast<char *>(dst->getDevicePtr(
      device, ur_mem_buffer_t::device_access_mode_t::write_only, dstOffset,
      size, [&](void *src, void *dst, size_t size) {
        ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                          (zeCommandList.get(), dst, src, size, nullptr,
                           waitListView.num, waitListView.handles));
        waitListView.clear();
      }));

  ZE2UR_CALL(zeCommandListAppendMemoryCopy,
             (zeCommandList.get(), pDst, pSrc, size, zeSignalEvent,
              waitListView.num, waitListView.handles));

  if (blocking) {
    ZE2UR_CALL(zeCommandListHostSynchronize, (zeCommandList.get(), UINT64_MAX));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::appendRegionCopyUnlocked(
    ur_mem_buffer_t *src, ur_mem_buffer_t *dst, bool blocking,
    ur_rect_offset_t srcOrigin, ur_rect_offset_t dstOrigin,
    ur_rect_region_t region, size_t srcRowPitch, size_t srcSlicePitch,
    size_t dstRowPitch, size_t dstSlicePitch, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent,
    ur_command_t commandType) {
  auto zeParams = ur2zeRegionParams(srcOrigin, dstOrigin, region, srcRowPitch,
                                    dstRowPitch, srcSlicePitch, dstSlicePitch);

  auto zeSignalEvent = getSignalEvent(phEvent, commandType);

  auto waitListView = getWaitListView(phEventWaitList, numEventsInWaitList);

  auto pSrc = ur_cast<char *>(src->getDevicePtr(
      device, ur_mem_buffer_t::device_access_mode_t::read_only, 0,
      src->getSize(), [&](void *src, void *dst, size_t size) {
        ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                          (zeCommandList.get(), dst, src, size, nullptr,
                           waitListView.num, waitListView.handles));
        waitListView.clear();
      }));
  auto pDst = ur_cast<char *>(dst->getDevicePtr(
      device, ur_mem_buffer_t::device_access_mode_t::write_only, 0,
      dst->getSize(), [&](void *src, void *dst, size_t size) {
        ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                          (zeCommandList.get(), dst, src, size, nullptr,
                           waitListView.num, waitListView.handles));
        waitListView.clear();
      }));

  ZE2UR_CALL(zeCommandListAppendMemoryCopyRegion,
             (zeCommandList.get(), pDst, &zeParams.dstRegion, zeParams.dstPitch,
              zeParams.dstSlicePitch, pSrc, &zeParams.srcRegion,
              zeParams.srcPitch, zeParams.srcSlicePitch, zeSignalEvent,
              waitListView.num, waitListView.handles));

  if (blocking) {
    ZE2UR_CALL(zeCommandListHostSynchronize, (zeCommandList.get(), UINT64_MAX));
  }

  return UR_RESULT_SUCCESS;
}

wait_list_view ur_command_list_manager::getWaitListView(
    const ur_event_handle_t *phWaitEvents, uint32_t numWaitEvents,
    ur_event_handle_t additionalWaitEvent) {
  uint32_t numWaitEventsEnabled = 0;
  if (isImmediateCommandList) {
    for (uint32_t i = 0; i < numWaitEvents; i++) {
      if (phWaitEvents[i]->getIsEventInUse()) {
        numWaitEventsEnabled++;
      }
    }
  } else {
    numWaitEventsEnabled = numWaitEvents;
  }
  uint32_t totalNumWaitEvents =
      numWaitEvents + (additionalWaitEvent != nullptr ? 1 : 0);
  waitList.resize(totalNumWaitEvents);
  for (uint32_t i = 0; i < numWaitEvents; i++) {
    if (isImmediateCommandList && !phWaitEvents[i]->getIsEventInUse()) {
      // We skip events on adding to immediate command list if they are not
      // enabled
      // TODO: This is a partial workaround for the underlying inconsistency
      // between normal and counter events in L0 driver
      // (the events that are not in use should be signaled by default, see
      // /test/conformance/exp_command_buffer/kernel_event_sync.cpp
      // KernelCommandEventSyncTest.SignalWaitBeforeEnqueue)
      continue;
    }
    waitList[i] = phWaitEvents[i]->getZeEvent();
  }
  if (additionalWaitEvent != nullptr) {
    waitList[totalNumWaitEvents - 1] = additionalWaitEvent->getZeEvent();
  }
  return {waitList.data(), static_cast<uint32_t>(totalNumWaitEvents)};
}

ze_event_handle_t
ur_command_list_manager::getSignalEvent(ur_event_handle_t *hUserEvent,
                                        ur_command_t commandType) {
  if (hUserEvent) {
    *hUserEvent = eventPool->allocate();
    (*hUserEvent)->resetQueueAndCommand(queue, commandType);
    return (*hUserEvent)->getZeEvent();
  } else {
    return nullptr;
  }
}

ur_result_t ur_command_list_manager::appendKernelLaunch(
    ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendKernelLaunch");

  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hKernel->getProgramHandle(), UR_RESULT_ERROR_INVALID_NULL_POINTER);

  UR_ASSERT(workDim > 0, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);
  UR_ASSERT(workDim < 4, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);

  ze_kernel_handle_t hZeKernel = hKernel->getZeHandle(device);

  std::scoped_lock<ur_shared_mutex> Lock(hKernel->Mutex);

  ze_group_count_t zeThreadGroupDimensions{1, 1, 1};
  uint32_t WG[3]{};
  UR_CALL(calculateKernelWorkDimensions(hZeKernel, device,
                                        zeThreadGroupDimensions, WG, workDim,
                                        pGlobalWorkSize, pLocalWorkSize));

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_KERNEL_LAUNCH);

  auto waitListView = getWaitListView(phEventWaitList, numEventsInWaitList);

  auto memoryMigrate = [&](void *src, void *dst, size_t size) {
    ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                      (zeCommandList.get(), dst, src, size, nullptr,
                       waitListView.num, waitListView.handles));
    waitListView.clear();
  };

  // If the offset is {0, 0, 0}, pass NULL instead.
  // This allows us to skip setting the offset.
  bool hasOffset = false;
  for (uint32_t i = 0; i < workDim; ++i) {
    hasOffset |= pGlobalWorkOffset[i];
  }
  if (!hasOffset) {
    pGlobalWorkOffset = NULL;
  }

  UR_CALL(hKernel->prepareForSubmission(context, device, pGlobalWorkOffset,
                                        workDim, WG[0], WG[1], WG[2],
                                        memoryMigrate));

  TRACK_SCOPE_LATENCY(
      "ur_command_list_manager::zeCommandListAppendLaunchKernel");
  ZE2UR_CALL(zeCommandListAppendLaunchKernel,
             (zeCommandList.get(), hZeKernel, &zeThreadGroupDimensions,
              zeSignalEvent, waitListView.num, waitListView.handles));

  postSubmit(hZeKernel, pGlobalWorkOffset);

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::appendUSMMemcpy(
    bool blocking, void *pDst, const void *pSrc, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendUSMMemcpy");

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_USM_MEMCPY);

  auto [pWaitEvents, numWaitEvents] =
      getWaitListView(phEventWaitList, numEventsInWaitList);

  ZE2UR_CALL(zeCommandListAppendMemoryCopy,
             (zeCommandList.get(), pDst, pSrc, size, zeSignalEvent,
              numWaitEvents, pWaitEvents));

  if (blocking) {
    ZE2UR_CALL(zeCommandListHostSynchronize, (zeCommandList.get(), UINT64_MAX));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::appendMemBufferFill(
    ur_mem_handle_t hMem, const void *pPattern, size_t patternSize,
    size_t offset, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemBufferFill");

  auto hBuffer = hMem->getBuffer();
  UR_ASSERT(offset + size <= hBuffer->getSize(), UR_RESULT_ERROR_INVALID_SIZE);

  std::scoped_lock<ur_shared_mutex> lock(hBuffer->getMutex());

  return appendGenericFillUnlocked(hBuffer, offset, patternSize, pPattern, size,
                                   numEventsInWaitList, phEventWaitList,
                                   phEvent, UR_COMMAND_MEM_BUFFER_FILL);
}

ur_result_t ur_command_list_manager::appendUSMFill(
    void *pMem, size_t patternSize, const void *pPattern, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendUSMFill");

  ur_usm_handle_t dstHandle(context, size, pMem);
  return appendGenericFillUnlocked(&dstHandle, 0, patternSize, pPattern, size,
                                   numEventsInWaitList, phEventWaitList,
                                   phEvent, UR_COMMAND_USM_FILL);
}

ur_result_t ur_command_list_manager::appendUSMPrefetch(
    const void *pMem, size_t size, ur_usm_migration_flags_t /*flags*/,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendUSMPrefetch");

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_USM_PREFETCH);

  auto [pWaitEvents, numWaitEvents] =
      getWaitListView(phEventWaitList, numEventsInWaitList);

  if (pWaitEvents) {
    ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
               (zeCommandList.get(), numWaitEvents, pWaitEvents));
  }
  // TODO: figure out how to translate "flags"
  ZE2UR_CALL(zeCommandListAppendMemoryPrefetch,
             (zeCommandList.get(), pMem, size));
  if (zeSignalEvent) {
    ZE2UR_CALL(zeCommandListAppendSignalEvent,
               (zeCommandList.get(), zeSignalEvent));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::appendUSMAdvise(
    const void *pMem, size_t size, ur_usm_advice_flags_t advice,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendUSMAdvise");

  auto zeAdvice = ur_cast<ze_memory_advice_t>(advice);

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_USM_ADVISE);

  auto [pWaitEvents, numWaitEvents] =
      getWaitListView(phEventWaitList, numEventsInWaitList);

  if (pWaitEvents) {
    ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
               (zeCommandList.get(), numWaitEvents, pWaitEvents));
  }

  ZE2UR_CALL(zeCommandListAppendMemAdvise,
             (zeCommandList.get(), device->ZeDevice, pMem, size, zeAdvice));

  if (zeSignalEvent) {
    ZE2UR_CALL(zeCommandListAppendSignalEvent,
               (zeCommandList.get(), zeSignalEvent));
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t
ur_command_list_manager::appendBarrier(uint32_t numEventsInWaitList,
                                       const ur_event_handle_t *phEventWaitList,
                                       ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendBarrier");

  auto zeSignalEvent =
      getSignalEvent(phEvent, UR_COMMAND_EVENTS_WAIT_WITH_BARRIER);
  auto [pWaitEvents, numWaitEvents] =
      getWaitListView(phEventWaitList, numEventsInWaitList);

  ZE2UR_CALL(zeCommandListAppendBarrier,
             (zeCommandList.get(), zeSignalEvent, numWaitEvents, pWaitEvents));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::appendMemBufferRead(
    ur_mem_handle_t hMem, bool blockingRead, size_t offset, size_t size,
    void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemBufferRead");

  auto hBuffer = hMem->getBuffer();
  UR_ASSERT(offset + size <= hBuffer->getSize(), UR_RESULT_ERROR_INVALID_SIZE);

  ur_usm_handle_t dstHandle(context, size, pDst);

  std::scoped_lock<ur_shared_mutex> lock(hBuffer->getMutex());

  return appendGenericCopyUnlocked(hBuffer, &dstHandle, blockingRead, offset, 0,
                                   size, numEventsInWaitList, phEventWaitList,
                                   phEvent, UR_COMMAND_MEM_BUFFER_READ);
}

ur_result_t ur_command_list_manager::appendMemBufferWrite(
    ur_mem_handle_t hMem, bool blockingWrite, size_t offset, size_t size,
    const void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemBufferWrite");

  auto hBuffer = hMem->getBuffer();
  UR_ASSERT(offset + size <= hBuffer->getSize(), UR_RESULT_ERROR_INVALID_SIZE);

  ur_usm_handle_t srcHandle(context, size, pSrc);

  std::scoped_lock<ur_shared_mutex> lock(hBuffer->getMutex());

  return appendGenericCopyUnlocked(
      &srcHandle, hBuffer, blockingWrite, 0, offset, size, numEventsInWaitList,
      phEventWaitList, phEvent, UR_COMMAND_MEM_BUFFER_WRITE);
}

ur_result_t ur_command_list_manager::appendMemBufferCopy(
    ur_mem_handle_t hSrc, ur_mem_handle_t hDst, size_t srcOffset,
    size_t dstOffset, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemBufferCopy");

  auto hBufferSrc = hSrc->getBuffer();
  auto hBufferDst = hDst->getBuffer();

  UR_ASSERT(srcOffset + size <= hBufferSrc->getSize(),
            UR_RESULT_ERROR_INVALID_SIZE);
  UR_ASSERT(dstOffset + size <= hBufferDst->getSize(),
            UR_RESULT_ERROR_INVALID_SIZE);

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> lock(
      hBufferSrc->getMutex(), hBufferDst->getMutex());

  return appendGenericCopyUnlocked(hBufferSrc, hBufferDst, false, srcOffset,
                                   dstOffset, size, numEventsInWaitList,
                                   phEventWaitList, phEvent,
                                   UR_COMMAND_MEM_BUFFER_COPY);
}

ur_result_t ur_command_list_manager::appendMemBufferReadRect(
    ur_mem_handle_t hMem, bool blockingRead, ur_rect_offset_t bufferOrigin,
    ur_rect_offset_t hostOrigin, ur_rect_region_t region, size_t bufferRowPitch,
    size_t bufferSlicePitch, size_t hostRowPitch, size_t hostSlicePitch,
    void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemBufferReadRect");

  auto hBuffer = hMem->getBuffer();
  ur_usm_handle_t dstHandle(context, 0, pDst);

  std::scoped_lock<ur_shared_mutex> lock(hBuffer->getMutex());

  return appendRegionCopyUnlocked(
      hBuffer, &dstHandle, blockingRead, bufferOrigin, hostOrigin, region,
      bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch,
      numEventsInWaitList, phEventWaitList, phEvent,
      UR_COMMAND_MEM_BUFFER_READ_RECT);
}

ur_result_t ur_command_list_manager::appendMemBufferWriteRect(
    ur_mem_handle_t hMem, bool blockingWrite, ur_rect_offset_t bufferOrigin,
    ur_rect_offset_t hostOrigin, ur_rect_region_t region, size_t bufferRowPitch,
    size_t bufferSlicePitch, size_t hostRowPitch, size_t hostSlicePitch,
    void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemBufferWriteRect");

  auto hBuffer = hMem->getBuffer();
  ur_usm_handle_t srcHandle(context, 0, pSrc);

  std::scoped_lock<ur_shared_mutex> lock(hBuffer->getMutex());

  return appendRegionCopyUnlocked(
      &srcHandle, hBuffer, blockingWrite, hostOrigin, bufferOrigin, region,
      hostRowPitch, hostSlicePitch, bufferRowPitch, bufferSlicePitch,
      numEventsInWaitList, phEventWaitList, phEvent,
      UR_COMMAND_MEM_BUFFER_WRITE_RECT);
}

ur_result_t ur_command_list_manager::appendMemBufferCopyRect(
    ur_mem_handle_t hSrc, ur_mem_handle_t hDst, ur_rect_offset_t srcOrigin,
    ur_rect_offset_t dstOrigin, ur_rect_region_t region, size_t srcRowPitch,
    size_t srcSlicePitch, size_t dstRowPitch, size_t dstSlicePitch,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemBufferCopyRect");

  auto hBufferSrc = hSrc->getBuffer();
  auto hBufferDst = hDst->getBuffer();

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> lock(
      hBufferSrc->getMutex(), hBufferDst->getMutex());

  return appendRegionCopyUnlocked(
      hBufferSrc, hBufferDst, false, srcOrigin, dstOrigin, region, srcRowPitch,
      srcSlicePitch, dstRowPitch, dstSlicePitch, numEventsInWaitList,
      phEventWaitList, phEvent, UR_COMMAND_MEM_BUFFER_COPY_RECT);
}

ur_result_t ur_command_list_manager::appendUSMMemcpy2D(
    bool blocking, void *pDst, size_t dstPitch, const void *pSrc,
    size_t srcPitch, size_t width, size_t height, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendUSMMemcpy2D");

  ur_rect_offset_t zeroOffset{0, 0, 0};
  ur_rect_region_t region{width, height, 0};

  ur_usm_handle_t srcHandle(context, 0, pSrc);
  ur_usm_handle_t dstHandle(context, 0, pDst);

  return appendRegionCopyUnlocked(&srcHandle, &dstHandle, blocking, zeroOffset,
                                  zeroOffset, region, srcPitch, 0, dstPitch, 0,
                                  numEventsInWaitList, phEventWaitList, phEvent,
                                  UR_COMMAND_MEM_BUFFER_COPY_RECT);
}

ze_command_list_handle_t ur_command_list_manager::getZeCommandList() {
  return zeCommandList.get();
}
