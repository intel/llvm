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
#include "command_buffer.hpp"
#include "context.hpp"
#include "kernel.hpp"
#include "memory.hpp"

ur_command_list_manager::ur_command_list_manager(
    ur_context_handle_t context, ur_device_handle_t device,
    v2::raii::command_list_unique_handle &&commandList)
    : hContext(context), hDevice(device),
      zeCommandList(std::move(commandList)) {
  UR_CALL_THROWS(ur::level_zero::urContextRetain(context));
  UR_CALL_THROWS(ur::level_zero::urDeviceRetain(device));
}

ur_command_list_manager::~ur_command_list_manager() {
  ur::level_zero::urContextRelease(hContext);
  ur::level_zero::urDeviceRelease(hDevice);
}

ur_result_t ur_command_list_manager::appendGenericFillUnlocked(
    ur_mem_buffer_t *dst, size_t offset, size_t patternSize,
    const void *pPattern, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t phEvent,
    ur_command_t commandType) {

  auto zeSignalEvent = getSignalEvent(phEvent, commandType);
  auto waitListView = getWaitListView(phEventWaitList, numEventsInWaitList);

  auto pDst = ur_cast<char *>(dst->getDevicePtr(
      hDevice, ur_mem_buffer_t::device_access_mode_t::read_only, offset, size,
      zeCommandList.get(), waitListView));

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
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t phEvent,
    ur_command_t commandType) {
  auto zeSignalEvent = getSignalEvent(phEvent, commandType);
  auto waitListView = getWaitListView(phEventWaitList, numEventsInWaitList);

  auto pSrc = ur_cast<char *>(src->getDevicePtr(
      hDevice, ur_mem_buffer_t::device_access_mode_t::read_only, srcOffset,
      size, zeCommandList.get(), waitListView));

  auto pDst = ur_cast<char *>(dst->getDevicePtr(
      hDevice, ur_mem_buffer_t::device_access_mode_t::write_only, dstOffset,
      size, zeCommandList.get(), waitListView));

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
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t phEvent,
    ur_command_t commandType) {
  auto zeParams = ur2zeRegionParams(srcOrigin, dstOrigin, region, srcRowPitch,
                                    dstRowPitch, srcSlicePitch, dstSlicePitch);

  auto zeSignalEvent = getSignalEvent(phEvent, commandType);
  auto waitListView = getWaitListView(phEventWaitList, numEventsInWaitList);

  auto pSrc = ur_cast<char *>(src->getDevicePtr(
      hDevice, ur_mem_buffer_t::device_access_mode_t::read_only, 0,
      src->getSize(), zeCommandList.get(), waitListView));
  auto pDst = ur_cast<char *>(dst->getDevicePtr(
      hDevice, ur_mem_buffer_t::device_access_mode_t::write_only, 0,
      dst->getSize(), zeCommandList.get(), waitListView));

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

  uint32_t totalNumWaitEvents =
      numWaitEvents + (additionalWaitEvent != nullptr ? 1 : 0);
  waitList.resize(totalNumWaitEvents);
  for (uint32_t i = 0; i < numWaitEvents; i++) {
    waitList[i] = phWaitEvents[i]->getZeEvent();
  }
  if (additionalWaitEvent != nullptr) {
    waitList[totalNumWaitEvents - 1] = additionalWaitEvent->getZeEvent();
  }
  return {waitList.data(), static_cast<uint32_t>(totalNumWaitEvents)};
}

ze_event_handle_t
ur_command_list_manager::getSignalEvent(ur_event_handle_t hUserEvent,
                                        ur_command_t commandType) {
  if (hUserEvent) {
    hUserEvent->setCommandType(commandType);
    return hUserEvent->getZeEvent();
  } else {
    return nullptr;
  }
}

ur_result_t ur_command_list_manager::appendKernelLaunchUnlocked(
    ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t phEvent,
    bool cooperative) {
  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hKernel->getProgramHandle(), UR_RESULT_ERROR_INVALID_NULL_POINTER);

  UR_ASSERT(workDim > 0, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);
  UR_ASSERT(workDim < 4, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);

  ze_kernel_handle_t hZeKernel = hKernel->getZeHandle(hDevice);

  std::scoped_lock<ur_shared_mutex> Lock(hKernel->Mutex);

  ze_group_count_t zeThreadGroupDimensions{1, 1, 1};
  uint32_t WG[3]{};
  UR_CALL(calculateKernelWorkDimensions(hZeKernel, hDevice,
                                        zeThreadGroupDimensions, WG, workDim,
                                        pGlobalWorkSize, pLocalWorkSize));

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_KERNEL_LAUNCH);
  auto waitListView = getWaitListView(phEventWaitList, numEventsInWaitList);

  UR_CALL(hKernel->prepareForSubmission(hContext, hDevice, pGlobalWorkOffset,
                                        workDim, WG[0], WG[1], WG[2],
                                        getZeCommandList(), waitListView));

  if (cooperative) {
    TRACK_SCOPE_LATENCY("ur_command_list_manager::"
                        "zeCommandListAppendLaunchCooperativeKernel");
    ZE2UR_CALL(zeCommandListAppendLaunchCooperativeKernel,
               (getZeCommandList(), hZeKernel, &zeThreadGroupDimensions,
                zeSignalEvent, waitListView.num, waitListView.handles));
  } else {
    TRACK_SCOPE_LATENCY("ur_command_list_manager::"
                        "zeCommandListAppendLaunchKernel");
    ZE2UR_CALL(zeCommandListAppendLaunchKernel,
               (getZeCommandList(), hZeKernel, &zeThreadGroupDimensions,
                zeSignalEvent, waitListView.num, waitListView.handles));
  }

  recordSubmittedKernel(hKernel);

  postSubmit(hZeKernel, pGlobalWorkOffset);

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::appendKernelLaunch(
    ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numPropsInLaunchPropList,
    const ur_kernel_launch_property_t *launchPropList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendKernelLaunch");

  for (uint32_t propIndex = 0; propIndex < numPropsInLaunchPropList;
       propIndex++) {
    if (launchPropList[propIndex].id ==
            UR_KERNEL_LAUNCH_PROPERTY_ID_COOPERATIVE &&
        launchPropList[propIndex].value.cooperative) {
      UR_CALL(appendKernelLaunchUnlocked(hKernel, workDim, pGlobalWorkOffset,
                                         pGlobalWorkSize, pLocalWorkSize,
                                         numEventsInWaitList, phEventWaitList,
                                         phEvent, true /* cooperative */));
      return UR_RESULT_SUCCESS;
    }
    if (launchPropList[propIndex].id != UR_KERNEL_LAUNCH_PROPERTY_ID_IGNORE &&
        launchPropList[propIndex].id !=
            UR_KERNEL_LAUNCH_PROPERTY_ID_COOPERATIVE) {
      // We don't support any other properties.
      return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }
  }

  UR_CALL(appendKernelLaunchUnlocked(
      hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize,
      numEventsInWaitList, phEventWaitList, phEvent, false /* cooperative */));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::appendUSMMemcpy(
    bool blocking, void *pDst, const void *pSrc, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t phEvent) {
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
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t phEvent) {
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
    ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendUSMFill");

  ur_usm_handle_t dstHandle(hContext, size, pMem);
  return appendGenericFillUnlocked(&dstHandle, 0, patternSize, pPattern, size,
                                   numEventsInWaitList, phEventWaitList,
                                   phEvent, UR_COMMAND_USM_FILL);
}

ur_result_t ur_command_list_manager::appendUSMPrefetch(
    const void *pMem, size_t size, ur_usm_migration_flags_t /*flags*/,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t phEvent) {
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
    ur_event_handle_t phEvent) {
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
             (zeCommandList.get(), hDevice->ZeDevice, pMem, size, zeAdvice));

  if (zeSignalEvent) {
    ZE2UR_CALL(zeCommandListAppendSignalEvent,
               (zeCommandList.get(), zeSignalEvent));
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::appendMemBufferRead(
    ur_mem_handle_t hMem, bool blockingRead, size_t offset, size_t size,
    void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemBufferRead");

  auto hBuffer = hMem->getBuffer();
  UR_ASSERT(offset + size <= hBuffer->getSize(), UR_RESULT_ERROR_INVALID_SIZE);

  ur_usm_handle_t dstHandle(hContext, size, pDst);

  std::scoped_lock<ur_shared_mutex> lock(hBuffer->getMutex());

  return appendGenericCopyUnlocked(hBuffer, &dstHandle, blockingRead, offset, 0,
                                   size, numEventsInWaitList, phEventWaitList,
                                   phEvent, UR_COMMAND_MEM_BUFFER_READ);
}

ur_result_t ur_command_list_manager::appendMemBufferWrite(
    ur_mem_handle_t hMem, bool blockingWrite, size_t offset, size_t size,
    const void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemBufferWrite");

  auto hBuffer = hMem->getBuffer();
  UR_ASSERT(offset + size <= hBuffer->getSize(), UR_RESULT_ERROR_INVALID_SIZE);

  ur_usm_handle_t srcHandle(hContext, size, pSrc);

  std::scoped_lock<ur_shared_mutex> lock(hBuffer->getMutex());

  return appendGenericCopyUnlocked(
      &srcHandle, hBuffer, blockingWrite, 0, offset, size, numEventsInWaitList,
      phEventWaitList, phEvent, UR_COMMAND_MEM_BUFFER_WRITE);
}

ur_result_t ur_command_list_manager::appendMemBufferCopy(
    ur_mem_handle_t hSrc, ur_mem_handle_t hDst, size_t srcOffset,
    size_t dstOffset, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t phEvent) {
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
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemBufferReadRect");

  auto hBuffer = hMem->getBuffer();
  ur_usm_handle_t dstHandle(hContext, 0, pDst);

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
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemBufferWriteRect");

  auto hBuffer = hMem->getBuffer();
  ur_usm_handle_t srcHandle(hContext, 0, pSrc);

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
    ur_event_handle_t phEvent) {
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
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendUSMMemcpy2D");

  ur_rect_offset_t zeroOffset{0, 0, 0};
  ur_rect_region_t region{width, height, 0};

  ur_usm_handle_t srcHandle(hContext, 0, pSrc);
  ur_usm_handle_t dstHandle(hContext, 0, pDst);

  return appendRegionCopyUnlocked(&srcHandle, &dstHandle, blocking, zeroOffset,
                                  zeroOffset, region, srcPitch, 0, dstPitch, 0,
                                  numEventsInWaitList, phEventWaitList, phEvent,
                                  UR_COMMAND_USM_MEMCPY_2D);
}

ur_result_t ur_command_list_manager::appendTimestampRecordingExp(
    bool blocking, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendTimestampRecordingExp");

  if (!phEvent) {
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  }

  auto [pWaitEvents, numWaitEvents] =
      getWaitListView(phEventWaitList, numEventsInWaitList);

  phEvent->recordStartTimestamp();

  auto [timestampPtr, zeSignalEvent] =
      (phEvent)->getEventEndTimestampAndHandle();

  ZE2UR_CALL(zeCommandListAppendWriteGlobalTimestamp,
             (getZeCommandList(), timestampPtr, zeSignalEvent, numWaitEvents,
              pWaitEvents));

  if (blocking) {
    ZE2UR_CALL(zeCommandListHostSynchronize, (getZeCommandList(), UINT64_MAX));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::appendGenericCommandListsExp(
    uint32_t numCommandLists, ze_command_list_handle_t *phCommandLists,
    ur_event_handle_t phEvent, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_command_t callerCommand,
    ur_event_handle_t additionalWaitEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendGenericCommandListsExp");

  auto zeSignalEvent = getSignalEvent(phEvent, callerCommand);
  auto [pWaitEvents, numWaitEvents] = getWaitListView(
      phEventWaitList, numEventsInWaitList, additionalWaitEvent);

  ZE2UR_CALL(zeCommandListImmediateAppendCommandListsExp,
             (getZeCommandList(), numCommandLists, phCommandLists,
              zeSignalEvent, numWaitEvents, pWaitEvents));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::appendCommandBufferExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t phEvent) {

  auto bufferCommandListLocked = hCommandBuffer->commandListManager.lock();
  ze_command_list_handle_t commandBufferCommandList =
      bufferCommandListLocked->zeCommandList.get();

  assert(phEvent);

  ur_event_handle_t executionEvent =
      hCommandBuffer->getExecutionEventUnlocked();

  if (executionEvent != nullptr) {
    ZE2UR_CALL(zeEventHostSynchronize,
               (executionEvent->getZeEvent(), UINT64_MAX));
  }

  UR_CALL(appendGenericCommandListsExp(
      1, &commandBufferCommandList, phEvent, numEventsInWaitList,
      phEventWaitList, UR_COMMAND_ENQUEUE_COMMAND_BUFFER_EXP, executionEvent));
  UR_CALL(hCommandBuffer->registerExecutionEventUnlocked(phEvent));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::appendMemImageRead(
    ur_mem_handle_t hMem, bool blockingRead, ur_rect_offset_t origin,
    ur_rect_region_t region, size_t rowPitch, size_t slicePitch, void *pDst,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemImageRead");

  auto hImage = hMem->getImage();

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_MEM_IMAGE_READ);
  auto waitListView = getWaitListView(phEventWaitList, numEventsInWaitList);

  auto [zeImage, zeRegion] =
      hImage->getRWRegion(origin, region, rowPitch, slicePitch);

  ZE2UR_CALL(zeCommandListAppendImageCopyToMemory,
             (getZeCommandList(), pDst, zeImage, &zeRegion, zeSignalEvent,
              waitListView.num, waitListView.handles));

  if (blockingRead) {
    ZE2UR_CALL(zeCommandListHostSynchronize, (getZeCommandList(), UINT64_MAX));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::appendMemImageWrite(
    ur_mem_handle_t hMem, bool blockingWrite, ur_rect_offset_t origin,
    ur_rect_region_t region, size_t rowPitch, size_t slicePitch, void *pSrc,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemImageWrite");

  auto hImage = hMem->getImage();

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_MEM_IMAGE_WRITE);
  auto waitListView = getWaitListView(phEventWaitList, numEventsInWaitList);

  auto [zeImage, zeRegion] =
      hImage->getRWRegion(origin, region, rowPitch, slicePitch);

  ZE2UR_CALL(zeCommandListAppendImageCopyFromMemory,
             (getZeCommandList(), zeImage, pSrc, &zeRegion, zeSignalEvent,
              waitListView.num, waitListView.handles));

  if (blockingWrite) {
    ZE2UR_CALL(zeCommandListHostSynchronize, (getZeCommandList(), UINT64_MAX));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::appendMemImageCopy(
    ur_mem_handle_t hSrc, ur_mem_handle_t hDst, ur_rect_offset_t srcOrigin,
    ur_rect_offset_t dstOrigin, ur_rect_region_t region,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemImageWrite");

  auto hImageSrc = hSrc->getImage();
  auto hImageDst = hDst->getImage();

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_MEM_IMAGE_COPY);
  auto waitListView = getWaitListView(phEventWaitList, numEventsInWaitList);

  auto desc = ur_mem_image_t::getCopyRegions(*hImageSrc, *hImageDst, srcOrigin,
                                             dstOrigin, region);

  auto [zeImageSrc, zeRegionSrc] = desc.src;
  auto [zeImageDst, zeRegionDst] = desc.dst;

  ZE2UR_CALL(zeCommandListAppendImageCopyRegion,
             (getZeCommandList(), zeImageDst, zeImageSrc, &zeRegionDst,
              &zeRegionSrc, zeSignalEvent, waitListView.num,
              waitListView.handles));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::appendMemBufferMap(
    ur_mem_handle_t hMem, bool blockingMap, ur_map_flags_t mapFlags,
    size_t offset, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t phEvent,
    void **ppRetMap) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemBufferMap");

  auto hBuffer = hMem->getBuffer();

  std::scoped_lock<ur_shared_mutex> lock(hBuffer->getMutex());

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_MEM_BUFFER_MAP);
  auto waitListView = getWaitListView(phEventWaitList, numEventsInWaitList);

  auto pDst = ur_cast<char *>(hBuffer->mapHostPtr(
      mapFlags, offset, size, zeCommandList.get(), waitListView));
  *ppRetMap = pDst;

  if (waitListView) {
    // If memory was not migrated, we need to wait on the events here.
    ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
               (getZeCommandList(), waitListView.num, waitListView.handles));
  }

  if (zeSignalEvent) {
    ZE2UR_CALL(zeCommandListAppendSignalEvent,
               (getZeCommandList(), zeSignalEvent));
  }

  if (blockingMap) {
    ZE2UR_CALL(zeCommandListHostSynchronize, (getZeCommandList(), UINT64_MAX));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::appendMemUnmap(
    ur_mem_handle_t hMem, void *pMappedPtr, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemUnmap");

  auto hBuffer = hMem->getBuffer();

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_MEM_UNMAP);
  auto waitListView = getWaitListView(phEventWaitList, numEventsInWaitList);

  // TODO: currently unmapHostPtr deallocates memory immediately,
  // since the memory might be used by the user, we need to make sure
  // all dependencies are completed.
  ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
             (getZeCommandList(), waitListView.num, waitListView.handles));
  waitListView.clear();

  hBuffer->unmapHostPtr(pMappedPtr, zeCommandList.get(), waitListView);
  if (zeSignalEvent) {
    ZE2UR_CALL(zeCommandListAppendSignalEvent,
               (getZeCommandList(), zeSignalEvent));
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::appendUSMFill2D(
    void * /*pMem*/, size_t /*pitch*/, size_t /*patternSize*/,
    const void * /*pPattern*/, size_t /*width*/, size_t /*height*/,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_event_handle_t /*phEvent*/) {
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

ur_result_t ur_command_list_manager::appendDeviceGlobalVariableWrite(
    ur_program_handle_t hProgram, const char *name, bool blockingWrite,
    size_t count, size_t offset, const void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY(
      "ur_command_list_manager::appendDeviceGlobalVariableWrite");

  // TODO: make getZeModuleHandle thread-safe
  ze_module_handle_t zeModule =
      hProgram->getZeModuleHandle(this->hDevice->ZeDevice);

  // Find global variable pointer
  auto globalVarPtr = getGlobalPointerFromModule(zeModule, offset, count, name);

  // Locking is done inside appendUSMMemcpy
  return appendUSMMemcpy(blockingWrite, ur_cast<char *>(globalVarPtr) + offset,
                         pSrc, count, numEventsInWaitList, phEventWaitList,
                         phEvent);
}

ur_result_t ur_command_list_manager::appendDeviceGlobalVariableRead(
    ur_program_handle_t hProgram, const char *name, bool blockingRead,
    size_t count, size_t offset, void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY(
      "ur_command_list_manager::appendDeviceGlobalVariableRead");

  // TODO: make getZeModuleHandle thread-safe
  ze_module_handle_t zeModule =
      hProgram->getZeModuleHandle(this->hDevice->ZeDevice);

  // Find global variable pointer
  auto globalVarPtr = getGlobalPointerFromModule(zeModule, offset, count, name);

  // Locking is done inside appendUSMMemcpy
  return appendUSMMemcpy(blockingRead, pDst,
                         ur_cast<char *>(globalVarPtr) + offset, count,
                         numEventsInWaitList, phEventWaitList, phEvent);
}

ur_result_t ur_command_list_manager::appendReadHostPipe(
    ur_program_handle_t /*hProgram*/, const char * /*pipe_symbol*/,
    bool /*blocking*/, void * /*pDst*/, size_t /*size*/,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_event_handle_t /*phEvent*/) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t ur_command_list_manager::appendWriteHostPipe(
    ur_program_handle_t /*hProgram*/, const char * /*pipe_symbol*/,
    bool /*blocking*/, void * /*pSrc*/, size_t /*size*/,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_event_handle_t /*phEvent*/) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t ur_command_list_manager::appendUSMAllocHelper(
    ur_queue_t_ *Queue, ur_usm_pool_handle_t pPool, const size_t size,
    const ur_exp_async_usm_alloc_properties_t *, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, void **ppMem,
    ur_event_handle_t phEvent, ur_usm_type_t type) {
  if (!pPool) {
    pPool = hContext->getAsyncPool();
  }

  auto device = (type == UR_USM_TYPE_HOST) ? nullptr : hDevice;

  ur_event_handle_t originAllocEvent = nullptr;
  auto asyncAlloc = pPool->allocateEnqueued(hContext, Queue, true, device,
                                            nullptr, type, size);
  if (!asyncAlloc) {
    auto Ret = pPool->allocate(hContext, device, nullptr, type, size, ppMem);
    if (Ret) {
      return Ret;
    }
  } else {
    std::tie(*ppMem, originAllocEvent) = *asyncAlloc;
  }

  auto waitListView =
      getWaitListView(phEventWaitList, numEventsInWaitList, originAllocEvent);

  ur_command_t commandType = UR_COMMAND_FORCE_UINT32;
  switch (type) {
  case UR_USM_TYPE_HOST:
    commandType = UR_COMMAND_ENQUEUE_USM_HOST_ALLOC_EXP;
    break;
  case UR_USM_TYPE_DEVICE:
    commandType = UR_COMMAND_ENQUEUE_USM_DEVICE_ALLOC_EXP;
    break;
  case UR_USM_TYPE_SHARED:
    commandType = UR_COMMAND_ENQUEUE_USM_SHARED_ALLOC_EXP;
    break;
  default:
    UR_LOG(ERR, "enqueueUSMAllocHelper: unsupported USM type");
    throw UR_RESULT_ERROR_INVALID_ARGUMENT;
  }

  auto zeSignalEvent = getSignalEvent(phEvent, commandType);
  auto [pWaitEvents, numWaitEvents] = waitListView;

  if (numWaitEvents > 0) {
    ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
               (getZeCommandList(), numWaitEvents, pWaitEvents));
  }
  if (zeSignalEvent) {
    ZE2UR_CALL(zeCommandListAppendSignalEvent,
               (getZeCommandList(), zeSignalEvent));
  }
  if (originAllocEvent) {
    originAllocEvent->release();
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::appendUSMFreeExp(
    ur_queue_t_ *Queue, ur_usm_pool_handle_t, void *pMem,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendUSMFreeExp");
  assert(phEvent);

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_ENQUEUE_USM_FREE_EXP);
  auto [pWaitEvents, numWaitEvents] =
      getWaitListView(phEventWaitList, numEventsInWaitList);

  umf_memory_pool_handle_t hPool = umfPoolByPtr(pMem);
  if (!hPool) {
    return UR_RESULT_ERROR_INVALID_MEM_OBJECT;
  }

  UsmPool *usmPool = nullptr;
  auto ret = umfPoolGetTag(hPool, (void **)&usmPool);
  if (ret != UMF_RESULT_SUCCESS || !usmPool) {
    // This should never happen
    UR_LOG(ERR, "enqueueUSMFreeExp: invalid pool tag");
    return UR_RESULT_ERROR_UNKNOWN;
  }

  size_t size = umfPoolMallocUsableSize(hPool, pMem);

  if (numWaitEvents > 0) {
    ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
               (getZeCommandList(), numWaitEvents, pWaitEvents));
  }

  ZE2UR_CALL(zeCommandListAppendSignalEvent,
             (getZeCommandList(), zeSignalEvent));

  // Insert must be done after the signal event is appended.
  usmPool->asyncPool.insert(pMem, size, phEvent, Queue);

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::bindlessImagesImageCopyExp(
    const void *pSrc, void *pDst, const ur_image_desc_t *pSrcImageDesc,
    const ur_image_desc_t *pDstImageDesc,
    const ur_image_format_t *pSrcImageFormat,
    const ur_image_format_t *pDstImageFormat,
    ur_exp_image_copy_region_t *pCopyRegion,
    ur_exp_image_copy_flags_t imageCopyFlags, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t phEvent) {

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_MEM_IMAGE_COPY);
  auto waitListView = getWaitListView(phEventWaitList, numEventsInWaitList);

  return bindlessImagesHandleCopyFlags(
      pSrc, pDst, pSrcImageDesc, pDstImageDesc, pSrcImageFormat,
      pDstImageFormat, pCopyRegion, imageCopyFlags, getZeCommandList(),
      zeSignalEvent, waitListView.num, waitListView.handles);
}

ur_result_t ur_command_list_manager::bindlessImagesWaitExternalSemaphoreExp(
    ur_exp_external_semaphore_handle_t /*hSemaphore*/, bool /*hasWaitValue*/,
    uint64_t /*waitValue*/, uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_event_handle_t /*phEvent*/) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t ur_command_list_manager::bindlessImagesSignalExternalSemaphoreExp(
    ur_exp_external_semaphore_handle_t /*hSemaphore*/, bool /*hasSignalValue*/,
    uint64_t /*signalValue*/, uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_event_handle_t /*phEvent*/) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t ur_command_list_manager::appendNativeCommandExp(
    ur_exp_enqueue_native_command_function_t, void *, uint32_t,
    const ur_mem_handle_t *, const ur_exp_enqueue_native_command_properties_t *,
    uint32_t, const ur_event_handle_t *, ur_event_handle_t) {
  UR_LOG_LEGACY(
      ERR, logger::LegacyMessage("[UR][L0_v2] {} function not implemented!"),
      "{} function not implemented!", __FUNCTION__);

  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

void ur_command_list_manager::recordSubmittedKernel(
    ur_kernel_handle_t hKernel) {
  submittedKernels.push_back(hKernel);
  hKernel->RefCount.increment();
}

ze_command_list_handle_t ur_command_list_manager::getZeCommandList() {
  return zeCommandList.get();
}

ur_result_t ur_command_list_manager::appendEventsWait(
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendEventsWait");

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_EVENTS_WAIT);
  auto [pWaitEvents, numWaitEvents] =
      getWaitListView(phEventWaitList, numEventsInWaitList);

  if (numWaitEvents > 0) {
    ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
               (zeCommandList.get(), numWaitEvents, pWaitEvents));
  }

  if (zeSignalEvent) {
    ZE2UR_CALL(zeCommandListAppendSignalEvent,
               (zeCommandList.get(), zeSignalEvent));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::appendEventsWaitWithBarrier(
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendEventsWaitWithBarrier");

  auto zeSignalEvent =
      getSignalEvent(phEvent, UR_COMMAND_EVENTS_WAIT_WITH_BARRIER);
  auto [pWaitEvents, numWaitEvents] =
      getWaitListView(phEventWaitList, numEventsInWaitList);

  ZE2UR_CALL(zeCommandListAppendBarrier,
             (zeCommandList.get(), zeSignalEvent, numWaitEvents, pWaitEvents));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::releaseSubmittedKernels() {
  // Free deferred kernels
  for (auto &hKernel : submittedKernels) {
    UR_CALL(hKernel->release());
  }
  submittedKernels.clear();
  return UR_RESULT_SUCCESS;
}
