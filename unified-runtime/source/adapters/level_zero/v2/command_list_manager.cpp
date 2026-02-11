//===--------- command_list_manager.cpp - Level Zero Adapter --------------===//
//
// Copyright (C) 2024-2026 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "command_list_manager.hpp"
#include "../helpers/kernel_helpers.hpp"
#include "../helpers/memory_helpers.hpp"
#include "../sampler.hpp"
#include "../ur_interface_loader.hpp"
#include "command_buffer.hpp"
#include "common.hpp"
#include "context.hpp"
#include "graph.hpp"
#include "kernel.hpp"
#include "memory.hpp"

thread_local std::vector<ze_event_handle_t> waitList;
// The wait_list_view is a wrapper for eventsWaitLists, which:
// -  enables passing a ze_event_handle_t buffer created from events as an
// argument for the driver API;
// - handles enqueueing operations associated with given events if these
// operations have not already been set for execution.
//
// Previously, it only stored the waitlist and the corresponding event count in
// a single container. Currently, the constructor also ensures that all
// associated operations will eventually be executed, which is required for
// batched queues in L0v2.
//
// Wait events might have been created in batched queues, which use regular
// command lists (batches). Since regular command lists are not executed
// immediately, but only after enqueueing on immediate lists, it is necessary to
// enqueue the regular command list associated with the given event. Otherwise,
// the event would never be signalled. The enqueueing is performed in
// onWaitListUse().
//
// In the case of batched queues, the function onWaitListUse() is not called if
// the current queue created the given event. The operation associated with the
// given wait_list_view is added to the current batch of the queue. The entire
// batch is then enqueued for execution, i.e., as part of queueFinish or
// queueFlush. For the same queue, events from the given eventsWaitList are
// enqueued before the associated operation is executed.

template <bool HasBatchedQueue>
void getZeHandlesBuffer(const ur_event_handle_t *phWaitEvents,
                        uint32_t numWaitEvents,
                        ur_queue_t_ *currentBatchedQueue) {
  for (uint32_t i = 0; i < numWaitEvents; i++) {
    // checking if the current queue has created the given event applies only
    // to batched queues
    if constexpr (HasBatchedQueue) {
      if (currentBatchedQueue != phWaitEvents[i]->getQueue()) {
        phWaitEvents[i]->onWaitListUse();
      }
    }
    waitList[i] = phWaitEvents[i]->getZeEvent();
  }
}

void wait_list_view::init(uint32_t numWaitEvents) {
  num = numWaitEvents;
  max_size = num + 1;

  waitList.resize(max_size);
}

void wait_list_view::setHandles(const ur_event_handle_t *phWaitEvents) {
  // vector.data() does not guarantee the null being returned in case of an
  // empty vector.
  // Explicit handling nullptr prevents passing uninitialized buffer to the
  // driver
  handles = phWaitEvents == nullptr ? nullptr : waitList.data();
}

wait_list_view::wait_list_view(const ur_event_handle_t *phWaitEvents,
                               uint32_t numWaitEvents) {
  init(numWaitEvents);
  getZeHandlesBuffer<false>(phWaitEvents, numWaitEvents, nullptr);
  setHandles(phWaitEvents);
}

wait_list_view::wait_list_view(const ur_event_handle_t *phWaitEvents,
                               uint32_t numWaitEvents,
                               ur_queue_t_ *currentBatchedQueue) {

  init(numWaitEvents);
  getZeHandlesBuffer<true>(phWaitEvents, numWaitEvents, currentBatchedQueue);
  setHandles(phWaitEvents);
}

// At most one additional event might be added after creating the given waitlist
void wait_list_view::addEvent(ur_event_handle_t phEvent) {
  if (!phEvent) {
    return;
  }

  if (handles) {
    assert(num != max_size);
    handles[num] = phEvent->getZeEvent();
    num++;
  } else {
    waitList.resize(0);
    waitList.emplace_back(phEvent->getZeEvent());
    num++;
    handles = waitList.data();
  }
}

ur_command_list_manager::ur_command_list_manager(
    ur_context_handle_t context, ur_device_handle_t device,
    v2::raii::command_list_unique_handle &&commandList)
    : hContext(context), hDevice(device),
      zeCommandList(std::move(commandList)) {}

v2::raii::command_list_unique_handle &&
ur_command_list_manager::releaseCommandList() {
  return std::move(zeCommandList);
}

void ur_command_list_manager::replaceCommandList(
    v2::raii::command_list_unique_handle &&cmdlist) {
  zeCommandList = std::move(cmdlist);
}

ur_result_t ur_command_list_manager::appendGenericFillUnlocked(
    ur_mem_buffer_t *dst, size_t offset, size_t patternSize,
    const void *pPattern, size_t size, wait_list_view &waitListView,
    ur_event_handle_t phEvent, ur_command_t commandType) {

  auto zeSignalEvent = getSignalEvent(phEvent, commandType);

  auto pDst = ur_cast<char *>(dst->getDevicePtr(
      hDevice.get(), ur_mem_buffer_t::device_access_mode_t::read_only, offset,
      size, zeCommandList.get(), waitListView));

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
    size_t dstOffset, size_t size, wait_list_view &waitListView,
    ur_event_handle_t phEvent, ur_command_t commandType) {
  auto zeSignalEvent = getSignalEvent(phEvent, commandType);

  auto pSrc = ur_cast<char *>(src->getDevicePtr(
      hDevice.get(), ur_mem_buffer_t::device_access_mode_t::read_only,
      srcOffset, size, zeCommandList.get(), waitListView));

  auto pDst = ur_cast<char *>(dst->getDevicePtr(
      hDevice.get(), ur_mem_buffer_t::device_access_mode_t::write_only,
      dstOffset, size, zeCommandList.get(), waitListView));

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
    size_t dstRowPitch, size_t dstSlicePitch, wait_list_view &waitListView,
    ur_event_handle_t phEvent, ur_command_t commandType) {
  auto zeParams = ur2zeRegionParams(srcOrigin, dstOrigin, region, srcRowPitch,
                                    dstRowPitch, srcSlicePitch, dstSlicePitch);

  auto zeSignalEvent = getSignalEvent(phEvent, commandType);

  auto pSrc = ur_cast<char *>(src->getDevicePtr(
      hDevice.get(), ur_mem_buffer_t::device_access_mode_t::read_only, 0,
      src->getSize(), zeCommandList.get(), waitListView));
  auto pDst = ur_cast<char *>(dst->getDevicePtr(
      hDevice.get(), ur_mem_buffer_t::device_access_mode_t::write_only, 0,
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

// must be called with hKernel->Mutex held
ur_result_t ur_command_list_manager::appendKernelLaunchLocked(
    ur_kernel_handle_t hKernel, ze_kernel_handle_t hZeKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, wait_list_view &waitListView,
    ur_event_handle_t phEvent, bool cooperative, bool callWithArgs,
    void *pNext) {

  ze_group_count_t zeThreadGroupDimensions{1, 1, 1};
  uint32_t WG[3]{};
  UR_CALL(calculateKernelWorkDimensions(hZeKernel, hDevice.get(),
                                        zeThreadGroupDimensions, WG, workDim,
                                        pGlobalWorkSize, pLocalWorkSize));

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_KERNEL_LAUNCH);

  UR_CALL(hKernel->prepareForSubmission(
      hContext.get(), hDevice.get(), pGlobalWorkOffset, workDim, WG[0], WG[1],
      WG[2], getZeCommandList(), waitListView));

  if (callWithArgs) {
    // zeCommandListAppendLaunchKernelWithArguments
    TRACK_SCOPE_LATENCY("ur_command_list_manager::"
                        "zeCommandListAppendLaunchKernelWithArguments");
    ze_group_size_t groupSize = {WG[0], WG[1], WG[2]};
    ZE2UR_CALL(zeCommandListAppendLaunchKernelWithArguments,
               (getZeCommandList(), hZeKernel, zeThreadGroupDimensions,
                groupSize, hKernel->kernelArgs.data(), pNext, zeSignalEvent,
                waitListView.num, waitListView.handles));
  } else if (cooperative) {
    // zeCommandListAppendLaunchCooperativeKernel
    TRACK_SCOPE_LATENCY("ur_command_list_manager::"
                        "zeCommandListAppendLaunchCooperativeKernel");
    ZE2UR_CALL(zeCommandListAppendLaunchCooperativeKernel,
               (getZeCommandList(), hZeKernel, &zeThreadGroupDimensions,
                zeSignalEvent, waitListView.num, waitListView.handles));
  } else {
    // zeCommandListAppendLaunchKernel
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

static ur_result_t kernelLaunchChecks(ur_kernel_handle_t hKernel,
                                      uint32_t workDim) {
  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hKernel->getProgramHandle(), UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(workDim > 0, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);
  UR_ASSERT(workDim < 4, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::appendKernelLaunchUnlocked(
    ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, wait_list_view &waitListView,
    ur_event_handle_t phEvent, bool cooperative) {

  ur_result_t checkResult = kernelLaunchChecks(hKernel, workDim);
  if (checkResult != UR_RESULT_SUCCESS) {
    return checkResult;
  }

  ze_kernel_handle_t hZeKernel = hKernel->getZeHandle(hDevice.get());

  std::scoped_lock<ur_shared_mutex> Lock(hKernel->Mutex);

  return appendKernelLaunchLocked(
      hKernel, hZeKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize,
      pLocalWorkSize, waitListView, phEvent, cooperative);
}

ur_result_t ur_command_list_manager::appendKernelLaunch(
    ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize,
    const ur_kernel_launch_ext_properties_t *launchPropList,
    wait_list_view &waitListView, ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendKernelLaunch");

  ur_kernel_launch_ext_properties_t *_launchPropList =
      const_cast<ur_kernel_launch_ext_properties_t *>(launchPropList);
  if (_launchPropList &&
      _launchPropList->flags & UR_KERNEL_LAUNCH_FLAG_COOPERATIVE) {
    UR_CALL(appendKernelLaunchUnlocked(
        hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize,
        waitListView, phEvent, true /* cooperative */));
    return UR_RESULT_SUCCESS;
  }

  if (_launchPropList &&
      _launchPropList->flags & ~UR_KERNEL_LAUNCH_FLAG_COOPERATIVE) {
    // We don't support any other flags.
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  while (_launchPropList != nullptr) {
    if (_launchPropList->stype !=
        as_stype<ur_kernel_launch_ext_properties_t>()) {
      // We don't support any other properties.
      return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }
    _launchPropList = static_cast<ur_kernel_launch_ext_properties_t *>(
        _launchPropList->pNext);
  }

  UR_CALL(appendKernelLaunchUnlocked(
      hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize,
      waitListView, phEvent, false /* cooperative */));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::appendUSMMemcpy(
    bool blocking, void *pDst, const void *pSrc, size_t size,
    wait_list_view &waitListView, ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendUSMMemcpy");

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_USM_MEMCPY);
  auto [pWaitEvents, numWaitEvents, _] = waitListView;

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
    size_t offset, size_t size, wait_list_view &waitListView,
    ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemBufferFill");

  auto hBuffer = hMem->getBuffer();
  UR_ASSERT(offset + size <= hBuffer->getSize(), UR_RESULT_ERROR_INVALID_SIZE);

  std::scoped_lock<ur_shared_mutex> lock(hBuffer->getMutex());

  return appendGenericFillUnlocked(hBuffer, offset, patternSize, pPattern, size,
                                   waitListView, phEvent,
                                   UR_COMMAND_MEM_BUFFER_FILL);
}

ur_result_t ur_command_list_manager::appendUSMFill(
    void *pMem, size_t patternSize, const void *pPattern, size_t size,
    wait_list_view &waitListView, ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendUSMFill");

  ur_usm_handle_t dstHandle(hContext.get(), size, pMem);
  return appendGenericFillUnlocked(&dstHandle, 0, patternSize, pPattern, size,
                                   waitListView, phEvent, UR_COMMAND_USM_FILL);
}

ur_result_t ur_command_list_manager::appendUSMPrefetch(
    const void *pMem, size_t size, ur_usm_migration_flags_t flags,
    wait_list_view &waitListView, ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendUSMPrefetch");

  switch (flags) {
  case UR_USM_MIGRATION_FLAG_HOST_TO_DEVICE:
    break;
  case UR_USM_MIGRATION_FLAG_DEVICE_TO_HOST:
    UR_LOG(WARN,
           "appendUSMPrefetch: L0v2 does not support prefetch to host yet");
    break;
  default:
    UR_LOG(ERR, "appendUSMPrefetch: invalid USM migration flag");
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_USM_PREFETCH);
  auto [pWaitEvents, numWaitEvents, _] = waitListView;

  if (pWaitEvents) {
    ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
               (zeCommandList.get(), numWaitEvents, pWaitEvents));
  }
  // TODO: Support migration flags after L0 backend support is added
  if (flags == UR_USM_MIGRATION_FLAG_HOST_TO_DEVICE) {
    ZE2UR_CALL(zeCommandListAppendMemoryPrefetch,
               (zeCommandList.get(), pMem, size));
  }
  if (zeSignalEvent) {
    ZE2UR_CALL(zeCommandListAppendSignalEvent,
               (zeCommandList.get(), zeSignalEvent));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::appendUSMAdvise(
    const void *pMem, size_t size, ur_usm_advice_flags_t advice,
    wait_list_view &waitListView, ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendUSMAdvise");

  auto zeAdvice = ur_cast<ze_memory_advice_t>(advice);

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_USM_ADVISE);
  auto [pWaitEvents, numWaitEvents, _] = waitListView;

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
    void *pDst, wait_list_view &waitListView, ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemBufferRead");

  auto hBuffer = hMem->getBuffer();
  UR_ASSERT(offset + size <= hBuffer->getSize(), UR_RESULT_ERROR_INVALID_SIZE);

  ur_usm_handle_t dstHandle(hContext.get(), size, pDst);

  std::scoped_lock<ur_shared_mutex> lock(hBuffer->getMutex());

  return appendGenericCopyUnlocked(hBuffer, &dstHandle, blockingRead, offset, 0,
                                   size, waitListView, phEvent,
                                   UR_COMMAND_MEM_BUFFER_READ);
}

ur_result_t ur_command_list_manager::appendMemBufferWrite(
    ur_mem_handle_t hMem, bool blockingWrite, size_t offset, size_t size,
    const void *pSrc, wait_list_view &waitListView, ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemBufferWrite");

  auto hBuffer = hMem->getBuffer();
  UR_ASSERT(offset + size <= hBuffer->getSize(), UR_RESULT_ERROR_INVALID_SIZE);

  ur_usm_handle_t srcHandle(hContext.get(), size, pSrc);

  std::scoped_lock<ur_shared_mutex> lock(hBuffer->getMutex());

  return appendGenericCopyUnlocked(&srcHandle, hBuffer, blockingWrite, 0,
                                   offset, size, waitListView, phEvent,
                                   UR_COMMAND_MEM_BUFFER_WRITE);
}

ur_result_t ur_command_list_manager::appendMemBufferCopy(
    ur_mem_handle_t hSrc, ur_mem_handle_t hDst, size_t srcOffset,
    size_t dstOffset, size_t size, wait_list_view &waitListView,
    ur_event_handle_t phEvent) {
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
                                   dstOffset, size, waitListView, phEvent,
                                   UR_COMMAND_MEM_BUFFER_COPY);
}

ur_result_t ur_command_list_manager::appendMemBufferReadRect(
    ur_mem_handle_t hMem, bool blockingRead, ur_rect_offset_t bufferOrigin,
    ur_rect_offset_t hostOrigin, ur_rect_region_t region, size_t bufferRowPitch,
    size_t bufferSlicePitch, size_t hostRowPitch, size_t hostSlicePitch,
    void *pDst, wait_list_view &waitListView, ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemBufferReadRect");

  auto hBuffer = hMem->getBuffer();
  ur_usm_handle_t dstHandle(hContext.get(), 0, pDst);

  std::scoped_lock<ur_shared_mutex> lock(hBuffer->getMutex());

  return appendRegionCopyUnlocked(
      hBuffer, &dstHandle, blockingRead, bufferOrigin, hostOrigin, region,
      bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch,
      waitListView, phEvent, UR_COMMAND_MEM_BUFFER_READ_RECT);
}

ur_result_t ur_command_list_manager::appendMemBufferWriteRect(
    ur_mem_handle_t hMem, bool blockingWrite, ur_rect_offset_t bufferOrigin,
    ur_rect_offset_t hostOrigin, ur_rect_region_t region, size_t bufferRowPitch,
    size_t bufferSlicePitch, size_t hostRowPitch, size_t hostSlicePitch,
    void *pSrc, wait_list_view &waitListView, ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemBufferWriteRect");

  auto hBuffer = hMem->getBuffer();
  ur_usm_handle_t srcHandle(hContext.get(), 0, pSrc);

  std::scoped_lock<ur_shared_mutex> lock(hBuffer->getMutex());

  return appendRegionCopyUnlocked(
      &srcHandle, hBuffer, blockingWrite, hostOrigin, bufferOrigin, region,
      hostRowPitch, hostSlicePitch, bufferRowPitch, bufferSlicePitch,
      waitListView, phEvent, UR_COMMAND_MEM_BUFFER_WRITE_RECT);
}

ur_result_t ur_command_list_manager::appendMemBufferCopyRect(
    ur_mem_handle_t hSrc, ur_mem_handle_t hDst, ur_rect_offset_t srcOrigin,
    ur_rect_offset_t dstOrigin, ur_rect_region_t region, size_t srcRowPitch,
    size_t srcSlicePitch, size_t dstRowPitch, size_t dstSlicePitch,
    wait_list_view &waitListView, ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemBufferCopyRect");

  auto hBufferSrc = hSrc->getBuffer();
  auto hBufferDst = hDst->getBuffer();

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> lock(
      hBufferSrc->getMutex(), hBufferDst->getMutex());

  return appendRegionCopyUnlocked(hBufferSrc, hBufferDst, false, srcOrigin,
                                  dstOrigin, region, srcRowPitch, srcSlicePitch,
                                  dstRowPitch, dstSlicePitch, waitListView,
                                  phEvent, UR_COMMAND_MEM_BUFFER_COPY_RECT);
}

ur_result_t ur_command_list_manager::appendUSMMemcpy2D(
    bool blocking, void *pDst, size_t dstPitch, const void *pSrc,
    size_t srcPitch, size_t width, size_t height, wait_list_view &waitListView,
    ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendUSMMemcpy2D");

  ur_rect_offset_t zeroOffset{0, 0, 0};
  ur_rect_region_t region{width, height, 0};

  ur_usm_handle_t srcHandle(hContext.get(), 0, pSrc);
  ur_usm_handle_t dstHandle(hContext.get(), 0, pDst);

  return appendRegionCopyUnlocked(&srcHandle, &dstHandle, blocking, zeroOffset,
                                  zeroOffset, region, srcPitch, 0, dstPitch, 0,
                                  waitListView, phEvent,
                                  UR_COMMAND_USM_MEMCPY_2D);
}

ur_result_t ur_command_list_manager::appendTimestampRecordingExp(
    bool blocking, wait_list_view &waitListView, ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendTimestampRecordingExp");

  if (!phEvent) {
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  }

  auto [pWaitEvents, numWaitEvents, _] = waitListView;

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
    ur_event_handle_t phEvent, wait_list_view &waitListView,
    ur_command_t callerCommand) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendGenericCommandListsExp");

  auto zeSignalEvent = getSignalEvent(phEvent, callerCommand);

  auto [pWaitEvents, numWaitEvents, _] = waitListView;

  ZE2UR_CALL(zeCommandListImmediateAppendCommandListsExp,
             (getZeCommandList(), numCommandLists, phCommandLists,
              zeSignalEvent, numWaitEvents, pWaitEvents));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::appendCommandBufferExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, wait_list_view &waitListView,
    ur_event_handle_t phEvent) {

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

  UR_CALL(appendGenericCommandListsExp(1, &commandBufferCommandList, phEvent,
                                       waitListView,
                                       UR_COMMAND_ENQUEUE_COMMAND_BUFFER_EXP));
  UR_CALL(hCommandBuffer->registerExecutionEventUnlocked(phEvent));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::appendMemImageRead(
    ur_mem_handle_t hMem, bool blockingRead, ur_rect_offset_t origin,
    ur_rect_region_t region, size_t rowPitch, size_t slicePitch, void *pDst,
    wait_list_view &waitListView, ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemImageRead");

  auto hImage = hMem->getImage();

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_MEM_IMAGE_READ);

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
    wait_list_view &waitListView, ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemImageWrite");

  auto hImage = hMem->getImage();

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_MEM_IMAGE_WRITE);

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
    wait_list_view &waitListView, ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemImageWrite");

  auto hImageSrc = hSrc->getImage();
  auto hImageDst = hDst->getImage();

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_MEM_IMAGE_COPY);

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
    size_t offset, size_t size, wait_list_view &waitListView,
    ur_event_handle_t phEvent, void **ppRetMap) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemBufferMap");

  auto hBuffer = hMem->getBuffer();

  std::scoped_lock<ur_shared_mutex> lock(hBuffer->getMutex());

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_MEM_BUFFER_MAP);

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

ur_result_t
ur_command_list_manager::appendMemUnmap(ur_mem_handle_t hMem, void *pMappedPtr,
                                        wait_list_view &waitListView,
                                        ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemUnmap");

  auto hBuffer = hMem->getBuffer();

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_MEM_UNMAP);

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
    wait_list_view & /* waitListView */, ur_event_handle_t /*phEvent*/) {
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
    UR_DFAILURE("Write device global variable is out of range");
    setErrorMessage("Write device global variable is out of range.",
                    UR_RESULT_ERROR_INVALID_VALUE,
                    static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_ARGUMENT));
    throw UR_RESULT_ERROR_ADAPTER_SPECIFIC;
  }
  return globalVarPtr;
}

ur_result_t ur_command_list_manager::appendDeviceGlobalVariableWrite(
    ur_program_handle_t hProgram, const char *name, bool blockingWrite,
    size_t count, size_t offset, const void *pSrc, wait_list_view &waitListView,
    ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY(
      "ur_command_list_manager::appendDeviceGlobalVariableWrite");

  // TODO: make getZeModuleHandle thread-safe
  ze_module_handle_t zeModule =
      hProgram->getZeModuleHandle(this->hDevice->ZeDevice);

  // Find global variable pointer
  auto globalVarPtr = getGlobalPointerFromModule(zeModule, offset, count, name);

  // Locking is done inside appendUSMMemcpy
  return appendUSMMemcpy(blockingWrite, ur_cast<char *>(globalVarPtr) + offset,
                         pSrc, count, waitListView, phEvent);
}

ur_result_t ur_command_list_manager::appendDeviceGlobalVariableRead(
    ur_program_handle_t hProgram, const char *name, bool blockingRead,
    size_t count, size_t offset, void *pDst, wait_list_view &waitListView,
    ur_event_handle_t phEvent) {
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
                         waitListView, phEvent);
}

ur_result_t ur_command_list_manager::appendReadHostPipe(
    ur_program_handle_t /*hProgram*/, const char * /*pipe_symbol*/,
    bool /*blocking*/, void * /*pDst*/, size_t /*size*/,
    wait_list_view & /* waitListView */, ur_event_handle_t /*phEvent*/) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t ur_command_list_manager::appendWriteHostPipe(
    ur_program_handle_t /*hProgram*/, const char * /*pipe_symbol*/,
    bool /*blocking*/, void * /*pSrc*/, size_t /*size*/,
    wait_list_view & /* waitListView */, ur_event_handle_t /*phEvent*/) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t ur_command_list_manager::appendUSMAllocHelper(
    ur_queue_t_ *Queue, ur_usm_pool_handle_t pPool, const size_t size,
    const ur_exp_async_usm_alloc_properties_t *, wait_list_view &waitListView,
    void **ppMem, ur_event_handle_t phEvent, ur_usm_type_t type) {
  if (!pPool) {
    pPool = hContext->getAsyncPool();
  }

  auto device = (type == UR_USM_TYPE_HOST) ? nullptr : hDevice.get();

  ur_event_handle_t originAllocEvent = nullptr;
  auto asyncAlloc = pPool->allocateEnqueued(hContext.get(), Queue, true, device,
                                            nullptr, type, size);
  if (!asyncAlloc) {
    auto Ret =
        pPool->allocate(hContext.get(), device, nullptr, type, size, ppMem);
    if (Ret) {
      return Ret;
    }
  } else {
    std::tie(*ppMem, originAllocEvent) = *asyncAlloc;
  }

  waitListView.addEvent(originAllocEvent);

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
    UR_FFAILURE("enqueueUSMAllocHelper: unsupported USM type:" << type);
  }

  auto zeSignalEvent = getSignalEvent(phEvent, commandType);
  auto [pWaitEvents, numWaitEvents, _] = waitListView;

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
    wait_list_view &waitListView, ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendUSMFreeExp");
  assert(phEvent);

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_ENQUEUE_USM_FREE_EXP);
  auto [pWaitEvents, numWaitEvents, _] = waitListView;

  umf_memory_pool_handle_t hPool = nullptr;
  auto umfRet = umfPoolByPtr(pMem, &hPool);
  if (umfRet != UMF_RESULT_SUCCESS || !hPool) {
    return UR_RESULT_ERROR_INVALID_MEM_OBJECT;
  }

  UsmPool *usmPool = nullptr;
  umfRet = umfPoolGetTag(hPool, (void **)&usmPool);
  if (umfRet != UMF_RESULT_SUCCESS || !usmPool) {
    // This should never happen
    UR_LOG(ERR, "enqueueUSMFreeExp: invalid pool tag");
    return UR_RESULT_ERROR_UNKNOWN;
  }

  size_t size = 0;
  umfRet = umfPoolMallocUsableSize(hPool, pMem, &size);
  if (umfRet != UMF_RESULT_SUCCESS) {
    UR_LOG(ERR, "enqueueUSMFreeExp: failed to retrieve usable malloc size");
    return UR_RESULT_ERROR_UNKNOWN;
  }

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
    ur_exp_image_copy_flags_t imageCopyFlags,
    ur_exp_image_copy_input_types_t imageCopyInputTypes,
    wait_list_view &waitListView, ur_event_handle_t phEvent) {

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_MEM_IMAGE_COPY);

  return bindlessImagesHandleCopyFlags(
      pSrc, pDst, pSrcImageDesc, pDstImageDesc, pSrcImageFormat,
      pDstImageFormat, pCopyRegion, imageCopyFlags, imageCopyInputTypes,
      getZeCommandList(), zeSignalEvent, waitListView.num,
      waitListView.handles);
}

ur_result_t ur_command_list_manager::bindlessImagesWaitExternalSemaphoreExp(
    ur_exp_external_semaphore_handle_t hSemaphore, bool hasWaitValue,
    uint64_t waitValue, wait_list_view &waitListView,
    ur_event_handle_t phEvent) {
  auto hPlatform = hContext->getPlatform();
  if (hPlatform->ZeExternalSemaphoreExt.Supported == false) {
    UR_LOG_LEGACY(ERR,
                  logger::LegacyMessage("[UR][L0] {} function not supported!"),
                  "{} function not supported!", __FUNCTION__);
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  auto zeSignalEvent =
      getSignalEvent(phEvent, UR_COMMAND_EXTERNAL_SEMAPHORE_WAIT_EXP);
  auto [pWaitEvents, numWaitEvents, _] = waitListView;

  ze_external_semaphore_wait_params_ext_t waitParams = {
      ZE_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_WAIT_PARAMS_EXT, nullptr, 0};
  waitParams.value = hasWaitValue ? waitValue : 0;
  ze_external_semaphore_ext_handle_t hExtSemaphore =
      reinterpret_cast<ze_external_semaphore_ext_handle_t>(hSemaphore);
  ZE2UR_CALL(hPlatform->ZeExternalSemaphoreExt
                 .zexCommandListAppendWaitExternalSemaphoresExp,
             (zeCommandList.get(), 1, &hExtSemaphore, &waitParams,
              zeSignalEvent, numWaitEvents, pWaitEvents));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::bindlessImagesSignalExternalSemaphoreExp(
    ur_exp_external_semaphore_handle_t hSemaphore, bool hasSignalValue,
    uint64_t signalValue, wait_list_view &waitListView,
    ur_event_handle_t phEvent) {
  auto hPlatform = hContext->getPlatform();
  if (hPlatform->ZeExternalSemaphoreExt.Supported == false) {
    UR_LOG_LEGACY(ERR,
                  logger::LegacyMessage("[UR][L0] {} function not supported!"),
                  "{} function not supported!", __FUNCTION__);
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  auto zeSignalEvent =
      getSignalEvent(phEvent, UR_COMMAND_EXTERNAL_SEMAPHORE_SIGNAL_EXP);
  auto [pWaitEvents, numWaitEvents, _] = waitListView;

  ze_external_semaphore_signal_params_ext_t signalParams = {
      ZE_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_EXT, nullptr, 0};
  signalParams.value = hasSignalValue ? signalValue : 0;
  ze_external_semaphore_ext_handle_t hExtSemaphore =
      reinterpret_cast<ze_external_semaphore_ext_handle_t>(hSemaphore);

  ZE2UR_CALL(hPlatform->ZeExternalSemaphoreExt
                 .zexCommandListAppendSignalExternalSemaphoresExp,
             (zeCommandList.get(), 1, &hExtSemaphore, &signalParams,
              zeSignalEvent, numWaitEvents, pWaitEvents));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::appendNativeCommandExp(
    ur_exp_enqueue_native_command_function_t, void *, uint32_t,
    const ur_mem_handle_t *, const ur_exp_enqueue_native_command_properties_t *,
    wait_list_view &, ur_event_handle_t) {
  UR_LOG_LEGACY(
      ERR, logger::LegacyMessage("[UR][L0_v2] {} function not implemented!"),
      "{} function not implemented!", __FUNCTION__);

  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

void ur_command_list_manager::recordSubmittedKernel(
    ur_kernel_handle_t hKernel) {
  auto [_, inserted] = submittedKernels.insert(hKernel);
  if (inserted) {
    hKernel->RefCount.retain();
  }
}

ze_command_list_handle_t ur_command_list_manager::getZeCommandList() {
  return zeCommandList.get();
}

ur_result_t
ur_command_list_manager::appendEventsWait(wait_list_view &waitListView,
                                          ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendEventsWait");

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_EVENTS_WAIT);
  auto [pWaitEvents, numWaitEvents, _] = waitListView;

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
    wait_list_view &waitList, ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendEventsWaitWithBarrier");

  auto zeSignalEvent =
      getSignalEvent(phEvent, UR_COMMAND_EVENTS_WAIT_WITH_BARRIER);
  auto [pWaitEvents, numWaitEvents, _] = waitList;

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

ur_result_t ur_command_list_manager::appendKernelLaunchWithArgsExpOld(
    ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numArgs,
    const ur_exp_kernel_arg_properties_t *pArgs,
    const ur_kernel_launch_ext_properties_t *launchPropList,
    wait_list_view &waitListView, ur_event_handle_t phEvent) {
  {
    std::scoped_lock<ur_shared_mutex> guard(hKernel->Mutex);
    ur_device_handle_t hDevice = this->hDevice.get();
    for (uint32_t argIndex = 0; argIndex < numArgs; argIndex++) {
      switch (pArgs[argIndex].type) {
      case UR_EXP_KERNEL_ARG_TYPE_LOCAL:
        UR_CALL(hKernel->setArgValue(hDevice, pArgs[argIndex].index,
                                     pArgs[argIndex].size, nullptr, nullptr));
        break;
      case UR_EXP_KERNEL_ARG_TYPE_VALUE:
        UR_CALL(hKernel->setArgValue(hDevice, pArgs[argIndex].index,
                                     pArgs[argIndex].size, nullptr,
                                     pArgs[argIndex].value.value));
        break;
      case UR_EXP_KERNEL_ARG_TYPE_POINTER:
        UR_CALL(hKernel->setArgPointer(hDevice, pArgs[argIndex].index, nullptr,
                                       pArgs[argIndex].value.pointer));
        break;
      case UR_EXP_KERNEL_ARG_TYPE_MEM_OBJ:
        // TODO: import helper for converting ur flags to internal equivalent
        UR_CALL(hKernel->addPendingMemoryAllocation(
            {pArgs[argIndex].value.memObjTuple.hMem,
             ur_mem_buffer_t::device_access_mode_t::read_write,
             pArgs[argIndex].index}));
        break;
      case UR_EXP_KERNEL_ARG_TYPE_SAMPLER: {
        UR_CALL(
            hKernel->setArgValue(hDevice, argIndex, sizeof(void *), nullptr,
                                 &pArgs[argIndex].value.sampler->ZeSampler));
        break;
      }
      default:
        return UR_RESULT_ERROR_INVALID_ENUMERATION;
      }
    }
  }

  UR_CALL(appendKernelLaunch(hKernel, workDim, pGlobalWorkOffset,
                             pGlobalWorkSize, pLocalWorkSize, launchPropList,
                             waitListView, phEvent));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::appendKernelLaunchWithArgsExpNew(
    ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numArgs,
    const ur_exp_kernel_arg_properties_t *pArgs,
    const ur_kernel_launch_ext_properties_t *launchPropList,
    wait_list_view &waitListView, ur_event_handle_t phEvent) {

  ur_result_t checkResult = kernelLaunchChecks(hKernel, workDim);
  if (checkResult != UR_RESULT_SUCCESS) {
    return checkResult;
  }

  // It is needed in case of UR_KERNEL_LAUNCH_PROPERTY_ID_COOPERATIVE
  // to launch the cooperative kernel.
  ZeStruct<ze_command_list_append_launch_kernel_param_cooperative_desc_t>
      cooperativeDesc;
  cooperativeDesc.isCooperative = static_cast<ze_bool_t>(true);
  void *pNext = nullptr;
  bool cooperativeKernelLaunchRequested = false;

  ur_kernel_launch_ext_properties_t *_launchPropList =
      const_cast<ur_kernel_launch_ext_properties_t *>(launchPropList);
  if (_launchPropList &&
      _launchPropList->flags & UR_KERNEL_LAUNCH_FLAG_COOPERATIVE) {
    cooperativeKernelLaunchRequested = true;
    pNext = &cooperativeDesc;
  }

  if (_launchPropList &&
      _launchPropList->flags & ~UR_KERNEL_LAUNCH_FLAG_COOPERATIVE) {
    // We don't support any other flags.
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  while (_launchPropList != nullptr) {
    if (_launchPropList->stype !=
        as_stype<ur_kernel_launch_ext_properties_t>()) {
      // We don't support any other properties.
      return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }
    _launchPropList = static_cast<ur_kernel_launch_ext_properties_t *>(
        _launchPropList->pNext);
  }

  ze_kernel_handle_t hZeKernel = hKernel->getZeHandle(hDevice.get());

  std::scoped_lock<ur_shared_mutex> Lock(hKernel->Mutex);

  // kernelMemObj contains kernel memory objects that
  // UR_EXP_KERNEL_ARG_TYPE_MEM_OBJ kernelArgs pointers point to
  hKernel->kernelMemObj.resize(numArgs, 0);
  hKernel->kernelArgs.resize(numArgs, 0);

  for (uint32_t argIndex = 0; argIndex < numArgs; argIndex++) {
    switch (pArgs[argIndex].type) {
    case UR_EXP_KERNEL_ARG_TYPE_LOCAL:
      hKernel->kernelArgs[argIndex] =
          reinterpret_cast<void *>(const_cast<size_t *>(&pArgs[argIndex].size));
      break;
    case UR_EXP_KERNEL_ARG_TYPE_VALUE:
      hKernel->kernelArgs[argIndex] =
          const_cast<void *>(pArgs[argIndex].value.value);
      break;
    case UR_EXP_KERNEL_ARG_TYPE_POINTER:
      hKernel->kernelArgs[argIndex] = reinterpret_cast<void *>(
          const_cast<void **>(&pArgs[argIndex].value.pointer));
      break;
    case UR_EXP_KERNEL_ARG_TYPE_MEM_OBJ:
      // compute zePtr for the given memory handle and store it in
      // hKernel->kernelMemObj[argIndex]
      UR_CALL(hKernel->computeZePtr(
          pArgs[argIndex].value.memObjTuple.hMem, hDevice.get(),
          ur_mem_buffer_t::device_access_mode_t::read_write, getZeCommandList(),
          waitListView, &hKernel->kernelMemObj[argIndex]));
      hKernel->kernelArgs[argIndex] = &hKernel->kernelMemObj[argIndex];
      break;
    case UR_EXP_KERNEL_ARG_TYPE_SAMPLER:
      hKernel->kernelArgs[argIndex] = &pArgs[argIndex].value.sampler->ZeSampler;
      break;
    default:
      return UR_RESULT_ERROR_INVALID_ENUMERATION;
    }
  }

  return appendKernelLaunchLocked(
      hKernel, hZeKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize,
      pLocalWorkSize, waitListView, phEvent, cooperativeKernelLaunchRequested,
      true /* callWithArgs */, pNext);
}

ur_result_t ur_command_list_manager::appendKernelLaunchWithArgsExp(
    ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numArgs,
    const ur_exp_kernel_arg_properties_t *pArgs,
    const ur_kernel_launch_ext_properties_t *launchPropList,
    wait_list_view &waitListView, ur_event_handle_t phEvent) {
  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_in_order_t::enqueueKernelLaunchWithArgsExp");

  bool cooperativeKernelLaunchRequested = false;

  ur_kernel_launch_ext_properties_t *_launchPropList =
      const_cast<ur_kernel_launch_ext_properties_t *>(launchPropList);

  if (_launchPropList &&
      _launchPropList->flags & UR_KERNEL_LAUNCH_FLAG_COOPERATIVE) {
    cooperativeKernelLaunchRequested = true;
  }

  ur_platform_handle_t hPlatform = hContext->getPlatform();
  bool KernelWithArgsSupported =
      hPlatform->ZeCommandListAppendLaunchKernelWithArgumentsExt.Supported;
  bool CooperativeCompatible =
      hPlatform->ZeCommandListAppendLaunchKernelWithArgumentsExt
          .DriverSupportsCooperativeKernelLaunchWithArgs;
  bool DisableZeLaunchKernelWithArgs =
      hPlatform->ZeCommandListAppendLaunchKernelWithArgumentsExt
          .DisableZeLaunchKernelWithArgs;
  bool RunNewPath =
      !DisableZeLaunchKernelWithArgs && KernelWithArgsSupported &&
      (!cooperativeKernelLaunchRequested ||
       (cooperativeKernelLaunchRequested && CooperativeCompatible));
  if (RunNewPath) {
    return appendKernelLaunchWithArgsExpNew(
        hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize,
        numArgs, pArgs, launchPropList, waitListView, phEvent);
  } else {
    // We cannot pass cooperativeKernelLaunchRequested to
    // appendKernelLaunchWithArgsExpOld() because appendKernelLaunch() must
    // check it on its own since it is called also from enqueueKernelLaunch().
    return appendKernelLaunchWithArgsExpOld(
        hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize,
        numArgs, pArgs, launchPropList, waitListView, phEvent);
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::beginGraphCapture() {
  if (!checkGraphExtensionSupport(hContext.get())) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  ZE2UR_CALL(hContext.get()
                 ->getPlatform()
                 ->ZeGraphExt.zeCommandListBeginGraphCaptureExp,
             (getZeCommandList(), nullptr));
  graphCapture.enableCapture();

  return UR_RESULT_SUCCESS;
}

ur_result_t
ur_command_list_manager::beginCaptureIntoGraph(ur_exp_graph_handle_t hGraph) {
  if (!checkGraphExtensionSupport(hContext.get())) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  ZE2UR_CALL(hContext.get()
                 ->getPlatform()
                 ->ZeGraphExt.zeCommandListBeginCaptureIntoGraphExp,
             (getZeCommandList(), hGraph->getZeHandle(), nullptr));
  graphCapture.enableCapture(hGraph);

  return UR_RESULT_SUCCESS;
}

ur_result_t
ur_command_list_manager::endGraphCapture(ur_exp_graph_handle_t *phGraph) {
  if (!checkGraphExtensionSupport(hContext.get())) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  ze_graph_handle_t zeGraph = nullptr;
  ZE2UR_CALL(
      hContext.get()->getPlatform()->ZeGraphExt.zeCommandListEndGraphCaptureExp,
      (getZeCommandList(), &zeGraph, nullptr));
  auto graph = graphCapture.getGraph();
  graphCapture.disableCapture();

  *phGraph =
      graph ? graph : new ur_exp_graph_handle_t_(hContext.get(), zeGraph);

  return UR_RESULT_SUCCESS;
}

ur_result_t
ur_command_list_manager::appendGraph(ur_exp_executable_graph_handle_t hGraph,
                                     wait_list_view &waitListView,
                                     ur_event_handle_t hEvent) {
  if (!checkGraphExtensionSupport(hContext.get())) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  auto zeSignalEvent = getSignalEvent(hEvent, UR_COMMAND_ENQUEUE_GRAPH_EXP);
  ZE2UR_CALL(
      hContext.get()->getPlatform()->ZeGraphExt.zeCommandListAppendGraphExp,
      (getZeCommandList(), hGraph->getZeHandle(), nullptr, zeSignalEvent,
       waitListView.num, waitListView.handles));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::isGraphCaptureActive(bool *pResult) {
  if (!checkGraphExtensionSupport(hContext.get())) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  ze_result_t ZeResult =
      ZE_CALL_NOCHECK(hContext.get()
                          ->getPlatform()
                          ->ZeGraphExt.zeCommandListIsGraphCaptureEnabledExp,
                      (getZeCommandList()));

  *pResult = (ZeResult == ZE_RESULT_QUERY_TRUE);

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::appendHostTaskExp(
    ur_exp_host_task_function_t pfnHostTask, void *data,
    const ur_exp_host_task_properties_t *pProperties,
    wait_list_view &waitListView, ur_event_handle_t phEvent) {

  ur_platform_handle_t hPlatform = hContext->getPlatform();

  if (!hPlatform->ZeHostTaskExt.Supported) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  ZE2UR_CALL(hPlatform->ZeHostTaskExt.zeCommandListAppendHostFunction,
             (getZeCommandList(), (void *)pfnHostTask, data,
              const_cast<void *>(reinterpret_cast<const void *>(pProperties)),
              getSignalEvent(phEvent, UR_COMMAND_HOST_TASK_EXP),
              waitListView.num, waitListView.handles));

  return UR_RESULT_SUCCESS;
}
