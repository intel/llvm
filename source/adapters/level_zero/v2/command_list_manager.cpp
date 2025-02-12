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
    ur_queue_t_ *queue)
    : context(context), device(device),
      eventPool(context->getEventPoolCache().borrow(device->Id.value(), flags)),
      zeCommandList(std::move(commandList)), queue(queue) {
  UR_CALL_THROWS(ur::level_zero::urContextRetain(context));
  UR_CALL_THROWS(ur::level_zero::urDeviceRetain(device));
}

ur_command_list_manager::~ur_command_list_manager() {
  ur::level_zero::urContextRelease(context);
  ur::level_zero::urDeviceRelease(device);
}

ur_result_t ur_command_list_manager::appendGenericCopyUnlocked(
    ur_mem_handle_t src, ur_mem_handle_t dst, bool blocking, size_t srcOffset,
    size_t dstOffset, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent,
    ur_command_t commandType) {
  auto zeSignalEvent = getSignalEvent(phEvent, commandType);

  auto waitListView = getWaitListView(phEventWaitList, numEventsInWaitList);

  auto pSrc = ur_cast<char *>(src->getDevicePtr(
      device, ur_mem_handle_t_::device_access_mode_t::read_only, srcOffset,
      size, [&](void *src, void *dst, size_t size) {
        ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                          (zeCommandList.get(), dst, src, size, nullptr,
                           waitListView.num, waitListView.handles));
        waitListView.clear();
      }));

  auto pDst = ur_cast<char *>(dst->getDevicePtr(
      device, ur_mem_handle_t_::device_access_mode_t::write_only, dstOffset,
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
    ur_mem_handle_t src, ur_mem_handle_t dst, bool blocking,
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
      device, ur_mem_handle_t_::device_access_mode_t::read_only, 0,
      src->getSize(), [&](void *src, void *dst, size_t size) {
        ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                          (zeCommandList.get(), dst, src, size, nullptr,
                           waitListView.num, waitListView.handles));
        waitListView.clear();
      }));
  auto pDst = ur_cast<char *>(dst->getDevicePtr(
      device, ur_mem_handle_t_::device_access_mode_t::write_only, 0,
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

wait_list_view
ur_command_list_manager::getWaitListView(const ur_event_handle_t *phWaitEvents,
                                         uint32_t numWaitEvents) {

  waitList.resize(numWaitEvents);
  for (uint32_t i = 0; i < numWaitEvents; i++) {
    waitList[i] = phWaitEvents[i]->getZeEvent();
  }

  return {waitList.data(), static_cast<uint32_t>(numWaitEvents)};
}

ze_event_handle_t
ur_command_list_manager::getSignalEvent(ur_event_handle_t *hUserEvent,
                                        ur_command_t commandType) {
  if (hUserEvent && queue) {
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

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(this->Mutex,
                                                          hKernel->Mutex);

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

  UR_CALL(hKernel->prepareForSubmission(context, device, pGlobalWorkOffset,
                                        workDim, WG[0], WG[1], WG[2],
                                        memoryMigrate));

  TRACK_SCOPE_LATENCY(
      "ur_command_list_manager::zeCommandListAppendLaunchKernel");
  ZE2UR_CALL(zeCommandListAppendLaunchKernel,
             (zeCommandList.get(), hZeKernel, &zeThreadGroupDimensions,
              zeSignalEvent, waitListView.num, waitListView.handles));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_command_list_manager::appendUSMMemcpy(
    bool blocking, void *pDst, const void *pSrc, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendUSMMemcpy");

  std::scoped_lock<ur_shared_mutex> lock(this->Mutex);

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

ur_result_t ur_command_list_manager::appendMemBufferRead(
    ur_mem_handle_t hBuffer, bool blockingRead, size_t offset, size_t size,
    void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemBufferRead");

  UR_ASSERT(offset + size <= hBuffer->getSize(), UR_RESULT_ERROR_INVALID_SIZE);

  ur_usm_handle_t_ dstHandle(context, size, pDst);

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> lock(this->Mutex,
                                                          hBuffer->getMutex());

  return appendGenericCopyUnlocked(hBuffer, &dstHandle, blockingRead, offset, 0,
                                   size, numEventsInWaitList, phEventWaitList,
                                   phEvent, UR_COMMAND_MEM_BUFFER_READ);
}

ur_result_t ur_command_list_manager::appendMemBufferWrite(
    ur_mem_handle_t hBuffer, bool blockingWrite, size_t offset, size_t size,
    const void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemBufferWrite");

  UR_ASSERT(offset + size <= hBuffer->getSize(), UR_RESULT_ERROR_INVALID_SIZE);

  ur_usm_handle_t_ srcHandle(context, size, pSrc);

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> lock(this->Mutex,
                                                          hBuffer->getMutex());

  return appendGenericCopyUnlocked(
      &srcHandle, hBuffer, blockingWrite, 0, offset, size, numEventsInWaitList,
      phEventWaitList, phEvent, UR_COMMAND_MEM_BUFFER_WRITE);
}

ur_result_t ur_command_list_manager::appendMemBufferCopy(
    ur_mem_handle_t hBufferSrc, ur_mem_handle_t hBufferDst, size_t srcOffset,
    size_t dstOffset, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemBufferCopy");

  UR_ASSERT(srcOffset + size <= hBufferSrc->getSize(),
            UR_RESULT_ERROR_INVALID_SIZE);
  UR_ASSERT(dstOffset + size <= hBufferDst->getSize(),
            UR_RESULT_ERROR_INVALID_SIZE);

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex, ur_shared_mutex> lock(
      this->Mutex, hBufferSrc->getMutex(), hBufferDst->getMutex());

  return appendGenericCopyUnlocked(hBufferSrc, hBufferDst, false, srcOffset,
                                   dstOffset, size, numEventsInWaitList,
                                   phEventWaitList, phEvent,
                                   UR_COMMAND_MEM_BUFFER_COPY);
}

ur_result_t ur_command_list_manager::appendMemBufferReadRect(
    ur_mem_handle_t hBuffer, bool blockingRead, ur_rect_offset_t bufferOrigin,
    ur_rect_offset_t hostOrigin, ur_rect_region_t region, size_t bufferRowPitch,
    size_t bufferSlicePitch, size_t hostRowPitch, size_t hostSlicePitch,
    void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemBufferReadRect");

  ur_usm_handle_t_ dstHandle(context, 0, pDst);

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> lock(this->Mutex,
                                                          hBuffer->getMutex());

  return appendRegionCopyUnlocked(
      hBuffer, &dstHandle, blockingRead, bufferOrigin, hostOrigin, region,
      bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch,
      numEventsInWaitList, phEventWaitList, phEvent,
      UR_COMMAND_MEM_BUFFER_READ_RECT);
}

ur_result_t ur_command_list_manager::appendMemBufferWriteRect(
    ur_mem_handle_t hBuffer, bool blockingWrite, ur_rect_offset_t bufferOrigin,
    ur_rect_offset_t hostOrigin, ur_rect_region_t region, size_t bufferRowPitch,
    size_t bufferSlicePitch, size_t hostRowPitch, size_t hostSlicePitch,
    void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemBufferWriteRect");

  ur_usm_handle_t_ srcHandle(context, 0, pSrc);

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> lock(this->Mutex,
                                                          hBuffer->getMutex());

  return appendRegionCopyUnlocked(
      &srcHandle, hBuffer, blockingWrite, hostOrigin, bufferOrigin, region,
      hostRowPitch, hostSlicePitch, bufferRowPitch, bufferSlicePitch,
      numEventsInWaitList, phEventWaitList, phEvent,
      UR_COMMAND_MEM_BUFFER_WRITE_RECT);
}

ur_result_t ur_command_list_manager::appendMemBufferCopyRect(
    ur_mem_handle_t hBufferSrc, ur_mem_handle_t hBufferDst,
    ur_rect_offset_t srcOrigin, ur_rect_offset_t dstOrigin,
    ur_rect_region_t region, size_t srcRowPitch, size_t srcSlicePitch,
    size_t dstRowPitch, size_t dstSlicePitch, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendMemBufferCopyRect");

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex, ur_shared_mutex> lock(
      this->Mutex, hBufferSrc->getMutex(), hBufferDst->getMutex());

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

  std::scoped_lock<ur_shared_mutex> lock(this->Mutex);

  ur_usm_handle_t_ srcHandle(context, 0, pSrc);
  ur_usm_handle_t_ dstHandle(context, 0, pDst);

  return appendRegionCopyUnlocked(&srcHandle, &dstHandle, blocking, zeroOffset,
                                  zeroOffset, region, srcPitch, 0, dstPitch, 0,
                                  numEventsInWaitList, phEventWaitList, phEvent,
                                  UR_COMMAND_MEM_BUFFER_COPY_RECT);
}

ze_command_list_handle_t ur_command_list_manager::getZeCommandList() {
  return zeCommandList.get();
}
