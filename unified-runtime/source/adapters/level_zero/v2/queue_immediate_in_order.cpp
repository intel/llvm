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
#include "../program.hpp"
#include "../ur_interface_loader.hpp"

namespace v2 {

wait_list_view ur_queue_immediate_in_order_t::getWaitListView(
    const ur_event_handle_t *phWaitEvents, uint32_t numWaitEvents) {
  return commandListManager.getWaitListView(phWaitEvents, numWaitEvents);
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
      commandListManager(
          hContext, hDevice,
          hContext->getCommandListCache().getImmediateCommandList(
              hDevice->ZeDevice, true, getZeOrdinal(hDevice),
              true /* always enable copy offload */,
              ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
              getZePriority(pProps ? pProps->flags : ur_queue_flags_t{}),
              getZeIndex(pProps)),
          eventFlagsFromQueueFlags(flags), this) {}

ur_queue_immediate_in_order_t::ur_queue_immediate_in_order_t(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_native_handle_t hNativeHandle, ur_queue_flags_t flags, bool ownZeQueue)
    : hContext(hContext), hDevice(hDevice), flags(flags),
      commandListManager(
          hContext, hDevice,
          raii::command_list_unique_handle(
              reinterpret_cast<ze_command_list_handle_t>(hNativeHandle),
              [ownZeQueue](ze_command_list_handle_t hZeCommandList) {
                if (ownZeQueue) {
                  ZE_CALL_NOCHECK(zeCommandListDestroy, (hZeCommandList));
                }
              }),
          eventFlagsFromQueueFlags(flags)) {}

ze_event_handle_t
ur_queue_immediate_in_order_t::getSignalEvent(ur_event_handle_t *hUserEvent,
                                              ur_command_t commandType) {
  return commandListManager.getSignalEvent(hUserEvent, commandType);
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
    auto status = ZE_CALL_NOCHECK(zeCommandListHostSynchronize,
                                  (commandListManager.getZeCommandList(), 0));
    if (status == ZE_RESULT_SUCCESS) {
      return ReturnValue(true);
    } else if (status == ZE_RESULT_NOT_READY) {
      return ReturnValue(false);
    } else {
      return ze2urResult(status);
    }
  }
  default:
    logger::error("Unsupported ParamName in urQueueGetInfo: "
                  "ParamName=ParamName={}(0x{})",
                  propName, logger::toHex(propName));
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  return UR_RESULT_SUCCESS;
}

void ur_queue_immediate_in_order_t::deferEventFree(ur_event_handle_t hEvent) {
  std::unique_lock<ur_shared_mutex> lock(this->Mutex);
  deferredEvents.push_back(hEvent);
}

ur_result_t ur_queue_immediate_in_order_t::queueGetNativeHandle(
    ur_queue_native_desc_t *pDesc, ur_native_handle_t *phNativeQueue) {
  std::ignore = pDesc;
  *phNativeQueue = reinterpret_cast<ur_native_handle_t>(
      this->commandListManager.getZeCommandList());
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::queueFinish() {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::queueFinish");

  std::unique_lock<ur_shared_mutex> lock(this->Mutex);

  // TODO: use zeEventHostSynchronize instead?
  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_in_order_t::zeCommandListHostSynchronize");
  ZE2UR_CALL(zeCommandListHostSynchronize,
             (commandListManager.getZeCommandList(), UINT64_MAX));

  // Free deferred events
  for (auto &hEvent : deferredEvents) {
    UR_CALL(hEvent->releaseDeferred());
  }
  deferredEvents.clear();

  // Free deferred kernels
  for (auto &hKernel : submittedKernels) {
    UR_CALL(hKernel->release());
  }
  submittedKernels.clear();

  return UR_RESULT_SUCCESS;
}

void ur_queue_immediate_in_order_t::recordSubmittedKernel(
    ur_kernel_handle_t hKernel) {
  submittedKernels.push_back(hKernel);
  hKernel->RefCount.increment();
}

ur_result_t ur_queue_immediate_in_order_t::queueFlush() {
  return UR_RESULT_SUCCESS;
}

ur_queue_immediate_in_order_t::~ur_queue_immediate_in_order_t() {
  try {
    UR_CALL_THROWS(queueFinish());
  } catch (...) {
    logger::error("Failed to finish queue on destruction");
  }
}

ur_result_t ur_queue_immediate_in_order_t::enqueueKernelLaunch(
    ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueKernelLaunch");

  UR_CALL(commandListManager.appendKernelLaunch(
      hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize,
      numEventsInWaitList, phEventWaitList, phEvent));

  recordSubmittedKernel(hKernel);

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

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_EVENTS_WAIT);
  auto [pWaitEvents, numWaitEvents] =
      getWaitListView(phEventWaitList, numEventsInWaitList);

  if (numWaitEvents > 0) {
    ZE2UR_CALL(
        zeCommandListAppendWaitOnEvents,
        (commandListManager.getZeCommandList(), numWaitEvents, pWaitEvents));
  }

  if (zeSignalEvent) {
    ZE2UR_CALL(zeCommandListAppendSignalEvent,
               (commandListManager.getZeCommandList(), zeSignalEvent));
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

  auto zeSignalEvent =
      getSignalEvent(phEvent, UR_COMMAND_EVENTS_WAIT_WITH_BARRIER);
  auto [pWaitEvents, numWaitEvents] =
      getWaitListView(phEventWaitList, numEventsInWaitList);

  ZE2UR_CALL(zeCommandListAppendBarrier,
             (commandListManager.getZeCommandList(), zeSignalEvent,
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

ur_result_t ur_queue_immediate_in_order_t::enqueueMemBufferRead(
    ur_mem_handle_t hMem, bool blockingRead, size_t offset, size_t size,
    void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueMemBufferRead");
  UR_CALL(commandListManager.appendMemBufferRead(
      hMem, blockingRead, offset, size, pDst, numEventsInWaitList,
      phEventWaitList, phEvent));
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemBufferWrite(
    ur_mem_handle_t hMem, bool blockingWrite, size_t offset, size_t size,
    const void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueMemBufferWrite");
  UR_CALL(commandListManager.appendMemBufferWrite(
      hMem, blockingWrite, offset, size, pSrc, numEventsInWaitList,
      phEventWaitList, phEvent));
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemBufferReadRect(
    ur_mem_handle_t hMem, bool blockingRead, ur_rect_offset_t bufferOrigin,
    ur_rect_offset_t hostOrigin, ur_rect_region_t region, size_t bufferRowPitch,
    size_t bufferSlicePitch, size_t hostRowPitch, size_t hostSlicePitch,
    void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_in_order_t::enqueueMemBufferReadRect");

  UR_CALL(commandListManager.appendMemBufferReadRect(
      hMem, blockingRead, bufferOrigin, hostOrigin, region, bufferRowPitch,
      bufferSlicePitch, hostRowPitch, hostSlicePitch, pDst, numEventsInWaitList,
      phEventWaitList, phEvent));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemBufferWriteRect(
    ur_mem_handle_t hMem, bool blockingWrite, ur_rect_offset_t bufferOrigin,
    ur_rect_offset_t hostOrigin, ur_rect_region_t region, size_t bufferRowPitch,
    size_t bufferSlicePitch, size_t hostRowPitch, size_t hostSlicePitch,
    void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_in_order_t::enqueueMemBufferWriteRect");

  UR_CALL(commandListManager.appendMemBufferWriteRect(
      hMem, blockingWrite, bufferOrigin, hostOrigin, region, bufferRowPitch,
      bufferSlicePitch, hostRowPitch, hostSlicePitch, pSrc, numEventsInWaitList,
      phEventWaitList, phEvent));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemBufferCopy(
    ur_mem_handle_t hSrc, ur_mem_handle_t hDst, size_t srcOffset,
    size_t dstOffset, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueMemBufferCopy");

  UR_CALL(commandListManager.appendMemBufferCopy(
      hSrc, hDst, srcOffset, dstOffset, size, numEventsInWaitList,
      phEventWaitList, phEvent));
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemBufferCopyRect(
    ur_mem_handle_t hSrc, ur_mem_handle_t hDst, ur_rect_offset_t srcOrigin,
    ur_rect_offset_t dstOrigin, ur_rect_region_t region, size_t srcRowPitch,
    size_t srcSlicePitch, size_t dstRowPitch, size_t dstSlicePitch,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_in_order_t::enqueueMemBufferCopyRect");

  UR_CALL(commandListManager.appendMemBufferCopyRect(
      hSrc, hDst, srcOrigin, dstOrigin, region, srcRowPitch, srcSlicePitch,
      dstRowPitch, dstSlicePitch, numEventsInWaitList, phEventWaitList,
      phEvent));
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemBufferFill(
    ur_mem_handle_t hMem, const void *pPattern, size_t patternSize,
    size_t offset, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueMemBufferFill");

  UR_CALL(commandListManager.appendMemBufferFill(
      hMem, pPattern, patternSize, offset, size, numEventsInWaitList,
      phEventWaitList, phEvent));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemImageRead(
    ur_mem_handle_t hMem, bool blockingRead, ur_rect_offset_t origin,
    ur_rect_region_t region, size_t rowPitch, size_t slicePitch, void *pDst,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueMemImageRead");

  auto hImage = hMem->getImage();

  std::scoped_lock<ur_shared_mutex> lock(this->Mutex);

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_MEM_IMAGE_READ);
  auto waitListView = getWaitListView(phEventWaitList, numEventsInWaitList);

  auto [zeImage, zeRegion] =
      hImage->getRWRegion(origin, region, rowPitch, slicePitch);

  ZE2UR_CALL(zeCommandListAppendImageCopyToMemory,
             (commandListManager.getZeCommandList(), pDst, zeImage, &zeRegion,
              zeSignalEvent, waitListView.num, waitListView.handles));

  if (blockingRead) {
    ZE2UR_CALL(zeCommandListHostSynchronize,
               (commandListManager.getZeCommandList(), UINT64_MAX));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemImageWrite(
    ur_mem_handle_t hMem, bool blockingWrite, ur_rect_offset_t origin,
    ur_rect_region_t region, size_t rowPitch, size_t slicePitch, void *pSrc,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueMemImageWrite");

  auto hImage = hMem->getImage();

  std::scoped_lock<ur_shared_mutex> lock(this->Mutex);

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_MEM_IMAGE_WRITE);
  auto waitListView = getWaitListView(phEventWaitList, numEventsInWaitList);

  auto [zeImage, zeRegion] =
      hImage->getRWRegion(origin, region, rowPitch, slicePitch);

  ZE2UR_CALL(zeCommandListAppendImageCopyFromMemory,
             (commandListManager.getZeCommandList(), zeImage, pSrc, &zeRegion,
              zeSignalEvent, waitListView.num, waitListView.handles));

  if (blockingWrite) {
    ZE2UR_CALL(zeCommandListHostSynchronize,
               (commandListManager.getZeCommandList(), UINT64_MAX));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemImageCopy(
    ur_mem_handle_t hSrc, ur_mem_handle_t hDst, ur_rect_offset_t srcOrigin,
    ur_rect_offset_t dstOrigin, ur_rect_region_t region,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueMemImageWrite");

  auto hImageSrc = hSrc->getImage();
  auto hImageDst = hDst->getImage();

  std::scoped_lock<ur_shared_mutex> lock(this->Mutex);

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_MEM_IMAGE_COPY);
  auto waitListView = getWaitListView(phEventWaitList, numEventsInWaitList);

  auto desc = ur_mem_image_t::getCopyRegions(*hImageSrc, *hImageDst, srcOrigin,
                                             dstOrigin, region);

  auto [zeImageSrc, zeRegionSrc] = desc.src;
  auto [zeImageDst, zeRegionDst] = desc.dst;

  ZE2UR_CALL(zeCommandListAppendImageCopyRegion,
             (commandListManager.getZeCommandList(), zeImageDst, zeImageSrc,
              &zeRegionDst, &zeRegionSrc, zeSignalEvent, waitListView.num,
              waitListView.handles));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemBufferMap(
    ur_mem_handle_t hMem, bool blockingMap, ur_map_flags_t mapFlags,
    size_t offset, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent,
    void **ppRetMap) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueMemBufferMap");

  auto hBuffer = hMem->getBuffer();

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> lock(this->Mutex,
                                                          hBuffer->getMutex());

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_MEM_BUFFER_MAP);

  auto waitListView = getWaitListView(phEventWaitList, numEventsInWaitList);

  auto pDst = ur_cast<char *>(hBuffer->mapHostPtr(
      mapFlags, offset, size, [&](void *src, void *dst, size_t size) {
        ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                          (commandListManager.getZeCommandList(), dst, src,
                           size, nullptr, waitListView.num,
                           waitListView.handles));
        waitListView.clear();
      }));
  *ppRetMap = pDst;

  if (waitListView) {
    // If memory was not migrated, we need to wait on the events here.
    ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
               (commandListManager.getZeCommandList(), waitListView.num,
                waitListView.handles));
  }

  if (zeSignalEvent) {
    ZE2UR_CALL(zeCommandListAppendSignalEvent,
               (commandListManager.getZeCommandList(), zeSignalEvent));
  }

  if (blockingMap) {
    ZE2UR_CALL(zeCommandListHostSynchronize,
               (commandListManager.getZeCommandList(), UINT64_MAX));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemUnmap(
    ur_mem_handle_t hMem, void *pMappedPtr, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueMemUnmap");

  auto hBuffer = hMem->getBuffer();

  std::scoped_lock<ur_shared_mutex> lock(this->Mutex);

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_MEM_UNMAP);

  auto waitListView = getWaitListView(phEventWaitList, numEventsInWaitList);

  // TODO: currently unmapHostPtr deallocates memory immediately,
  // since the memory might be used by the user, we need to make sure
  // all dependencies are completed.
  ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
             (commandListManager.getZeCommandList(), waitListView.num,
              waitListView.handles));
  waitListView.clear();

  hBuffer->unmapHostPtr(pMappedPtr, [&](void *src, void *dst, size_t size) {
    ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                      (commandListManager.getZeCommandList(), dst, src, size,
                       nullptr, waitListView.num, waitListView.handles));
  });
  if (zeSignalEvent) {
    ZE2UR_CALL(zeCommandListAppendSignalEvent,
               (commandListManager.getZeCommandList(), zeSignalEvent));
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueUSMFill(
    void *pMem, size_t patternSize, const void *pPattern, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueUSMFill");

  UR_CALL(commandListManager.appendUSMFill(pMem, patternSize, pPattern, size,
                                           numEventsInWaitList, phEventWaitList,
                                           phEvent));
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueUSMMemcpy(
    bool blocking, void *pDst, const void *pSrc, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  // TODO: parametrize latency tracking with 'blocking'
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueUSMMemcpy");

  UR_CALL(commandListManager.appendUSMMemcpy(blocking, pDst, pSrc, size,
                                             numEventsInWaitList,
                                             phEventWaitList, phEvent));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueUSMPrefetch(
    const void *pMem, size_t size, ur_usm_migration_flags_t flags,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueUSMPrefetch");
  UR_CALL(commandListManager.appendUSMPrefetch(
      pMem, size, flags, numEventsInWaitList, phEventWaitList, phEvent));
  return UR_RESULT_SUCCESS;
}

ur_result_t
ur_queue_immediate_in_order_t::enqueueUSMAdvise(const void *pMem, size_t size,
                                                ur_usm_advice_flags_t advice,
                                                ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueUSMAdvise");

  UR_CALL(commandListManager.appendUSMAdvise(pMem, size, advice, phEvent));
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
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueUSMMemcpy2D");
  UR_CALL(commandListManager.appendUSMMemcpy2D(
      blocking, pDst, dstPitch, pSrc, srcPitch, width, height,
      numEventsInWaitList, phEventWaitList, phEvent));
  return UR_RESULT_SUCCESS;
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

ur_result_t ur_queue_immediate_in_order_t::enqueueUSMDeviceAllocExp(
    ur_usm_pool_handle_t, const size_t,
    const ur_exp_async_usm_alloc_properties_t *, uint32_t,
    const ur_event_handle_t *, void **, ur_event_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueUSMSharedAllocExp(
    ur_usm_pool_handle_t, const size_t,
    const ur_exp_async_usm_alloc_properties_t *, uint32_t,
    const ur_event_handle_t *, void **, ur_event_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueUSMHostAllocExp(
    ur_usm_pool_handle_t, const size_t,
    const ur_exp_async_usm_alloc_properties_t *, uint32_t,
    const ur_event_handle_t *, void **, ur_event_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueUSMFreeExp(
    ur_usm_pool_handle_t, void *, uint32_t, const ur_event_handle_t *,
    ur_event_handle_t *) {
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

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_KERNEL_LAUNCH);

  auto waitListView = getWaitListView(phEventWaitList, numEventsInWaitList);

  auto memoryMigrate = [&](void *src, void *dst, size_t size) {
    ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                      (commandListManager.getZeCommandList(), dst, src, size,
                       nullptr, waitListView.num, waitListView.handles));
    waitListView.clear();
  };

  UR_CALL(hKernel->prepareForSubmission(hContext, hDevice, pGlobalWorkOffset,
                                        workDim, WG[0], WG[1], WG[2],
                                        memoryMigrate));

  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::"
                      "zeCommandListAppendLaunchCooperativeKernel");
  ZE2UR_CALL(zeCommandListAppendLaunchCooperativeKernel,
             (commandListManager.getZeCommandList(), hZeKernel,
              &zeThreadGroupDimensions, zeSignalEvent, waitListView.num,
              waitListView.handles));

  recordSubmittedKernel(hKernel);

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueTimestampRecordingExp(
    bool blocking, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_in_order_t::enqueueTimestampRecordingExp");

  std::scoped_lock<ur_shared_mutex> lock(this->Mutex);

  if (!phEvent && !*phEvent) {
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  }
  getSignalEvent(phEvent, UR_COMMAND_TIMESTAMP_RECORDING_EXP);
  auto [pWaitEvents, numWaitEvents] =
      getWaitListView(phEventWaitList, numEventsInWaitList);

  (*phEvent)->recordStartTimestamp();

  auto [timestampPtr, zeSignalEvent] =
      (*phEvent)->getEventEndTimestampAndHandle();

  ZE2UR_CALL(zeCommandListAppendWriteGlobalTimestamp,
             (commandListManager.getZeCommandList(), timestampPtr,
              zeSignalEvent, numWaitEvents, pWaitEvents));

  if (blocking) {
    ZE2UR_CALL(zeCommandListHostSynchronize,
               (commandListManager.getZeCommandList(), UINT64_MAX));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueGenericCommandListsExp(
    uint32_t numCommandLists, ze_command_list_handle_t *phCommandLists,
    ur_event_handle_t *phEvent, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_command_t callerCommand) {
  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_in_order_t::enqueueGenericCommandListsExp");

  std::scoped_lock<ur_shared_mutex> Lock(this->Mutex);
  auto zeSignalEvent = getSignalEvent(phEvent, callerCommand);

  auto [pWaitEvents, numWaitEvents] =
      getWaitListView(phEventWaitList, numEventsInWaitList);
  // zeCommandListImmediateAppendCommandListsExp is not working with in-order
  // immediate lists what causes problems with synchronization
  // TODO: remove synchronization when it is not needed
  ZE_CALL_NOCHECK(zeCommandListHostSynchronize,
                  (commandListManager.getZeCommandList(), UINT64_MAX));
  ZE2UR_CALL(zeCommandListImmediateAppendCommandListsExp,
             (commandListManager.getZeCommandList(), numCommandLists,
              phCommandLists, zeSignalEvent, numWaitEvents, pWaitEvents));
  ZE_CALL_NOCHECK(zeCommandListHostSynchronize,
                  (commandListManager.getZeCommandList(), UINT64_MAX));
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueCommandBuffer(
    ze_command_list_handle_t commandBufferCommandList,
    ur_event_handle_t *phEvent, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList) {
  return enqueueGenericCommandListsExp(1, &commandBufferCommandList, phEvent,
                                       numEventsInWaitList, phEventWaitList,
                                       UR_COMMAND_COMMAND_BUFFER_ENQUEUE_EXP);
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
