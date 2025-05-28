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

namespace v2 {

wait_list_view ur_queue_immediate_in_order_t::getWaitListView(
    locked<ur_command_list_manager> &commandList,
    const ur_event_handle_t *phWaitEvents, uint32_t numWaitEvents,
    ur_event_handle_t additionalWaitEvent) {
  return commandList->getWaitListView(phWaitEvents, numWaitEvents,
                                      additionalWaitEvent);
}

static uint32_t getZeOrdinal(ur_device_handle_t hDevice) {
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
              hDevice->ZeDevice,
              {true, getZeOrdinal(hDevice),
               true /* always enable copy offload */},
              ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
              getZePriority(pProps ? pProps->flags : ur_queue_flags_t{}),
              getZeIndex(pProps)),
          eventFlagsFromQueueFlags(flags), this, PoolCacheType::Immediate) {}

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
                  if (checkL0LoaderTeardown()) {
                    ZE_CALL_NOCHECK(zeCommandListDestroy, (hZeCommandList));
                  }
                }
              }),
          eventFlagsFromQueueFlags(flags), this, PoolCacheType::Immediate) {}

ze_event_handle_t ur_queue_immediate_in_order_t::getSignalEvent(
    locked<ur_command_list_manager> &commandList, ur_event_handle_t *hUserEvent,
    ur_command_t commandType) {
  return commandList->getSignalEvent(hUserEvent, commandType);
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
      this->commandListManager.get_no_lock()->getZeCommandList());
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::queueFinish() {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::queueFinish");

  auto commandListLocked = commandListManager.lock();
  // TODO: use zeEventHostSynchronize instead?
  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_in_order_t::zeCommandListHostSynchronize");
  ZE2UR_CALL(zeCommandListHostSynchronize,
             (commandListLocked->getZeCommandList(), UINT64_MAX));

  hContext->getAsyncPool()->cleanupPoolsForQueue(this);

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
    // Ignore errors during destruction
  }
}

ur_result_t ur_queue_immediate_in_order_t::enqueueKernelLaunch(
    ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numPropsInLaunchPropList,
    const ur_kernel_launch_property_t *launchPropList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueKernelLaunch");

  for (uint32_t propIndex = 0; propIndex < numPropsInLaunchPropList;
       propIndex++) {
    if (launchPropList[propIndex].id ==
            UR_KERNEL_LAUNCH_PROPERTY_ID_COOPERATIVE &&
        launchPropList[propIndex].value.cooperative) {
      return enqueueCooperativeKernelLaunchHelper(
          hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize,
          numEventsInWaitList, phEventWaitList, phEvent);
    }
    if (launchPropList[propIndex].id != UR_KERNEL_LAUNCH_PROPERTY_ID_IGNORE &&
        launchPropList[propIndex].id !=
            UR_KERNEL_LAUNCH_PROPERTY_ID_COOPERATIVE) {
      // We don't support any other properties.
      return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }
  }

  auto commandListLocked = commandListManager.lock();
  UR_CALL(commandListLocked->appendKernelLaunch(
      hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize,
      numEventsInWaitList, phEventWaitList, phEvent));

  recordSubmittedKernel(hKernel);

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueEventsWait(
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueEventsWait");

  auto commandListLocked = commandListManager.lock();
  if (!numEventsInWaitList && !phEvent) {
    // nop
    return UR_RESULT_SUCCESS;
  }

  auto zeSignalEvent =
      getSignalEvent(commandListLocked, phEvent, UR_COMMAND_EVENTS_WAIT);
  auto [pWaitEvents, numWaitEvents] =
      getWaitListView(commandListLocked, phEventWaitList, numEventsInWaitList);

  if (numWaitEvents > 0) {
    ZE2UR_CALL(
        zeCommandListAppendWaitOnEvents,
        (commandListLocked->getZeCommandList(), numWaitEvents, pWaitEvents));
  }

  if (zeSignalEvent) {
    ZE2UR_CALL(zeCommandListAppendSignalEvent,
               (commandListLocked->getZeCommandList(), zeSignalEvent));
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueEventsWaitWithBarrierImpl(
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_in_order_t::enqueueEventsWaitWithBarrier");

  auto commandListLocked = commandListManager.lock();
  if (!numEventsInWaitList && !phEvent) {
    // nop
    return UR_RESULT_SUCCESS;
  }

  auto zeSignalEvent = getSignalEvent(commandListLocked, phEvent,
                                      UR_COMMAND_EVENTS_WAIT_WITH_BARRIER);
  auto [pWaitEvents, numWaitEvents] =
      getWaitListView(commandListLocked, phEventWaitList, numEventsInWaitList);

  ZE2UR_CALL(zeCommandListAppendBarrier,
             (commandListLocked->getZeCommandList(), zeSignalEvent,
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
  auto commandListLocked = commandListManager.lock();
  UR_CALL(commandListLocked->appendMemBufferRead(
      hMem, blockingRead, offset, size, pDst, numEventsInWaitList,
      phEventWaitList, phEvent));
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemBufferWrite(
    ur_mem_handle_t hMem, bool blockingWrite, size_t offset, size_t size,
    const void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueMemBufferWrite");
  auto commandListLocked = commandListManager.lock();
  UR_CALL(commandListLocked->appendMemBufferWrite(
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

  auto commandListLocked = commandListManager.lock();
  UR_CALL(commandListLocked->appendMemBufferReadRect(
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

  auto commandListLocked = commandListManager.lock();
  UR_CALL(commandListLocked->appendMemBufferWriteRect(
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

  auto commandListLocked = commandListManager.lock();
  UR_CALL(commandListLocked->appendMemBufferCopy(
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

  auto commandListLocked = commandListManager.lock();
  UR_CALL(commandListLocked->appendMemBufferCopyRect(
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

  auto commandListLocked = commandListManager.lock();
  UR_CALL(commandListLocked->appendMemBufferFill(
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

  auto commandListLocked = commandListManager.lock();

  auto zeSignalEvent =
      getSignalEvent(commandListLocked, phEvent, UR_COMMAND_MEM_IMAGE_READ);
  auto waitListView =
      getWaitListView(commandListLocked, phEventWaitList, numEventsInWaitList);

  auto [zeImage, zeRegion] =
      hImage->getRWRegion(origin, region, rowPitch, slicePitch);

  ZE2UR_CALL(zeCommandListAppendImageCopyToMemory,
             (commandListLocked->getZeCommandList(), pDst, zeImage, &zeRegion,
              zeSignalEvent, waitListView.num, waitListView.handles));

  if (blockingRead) {
    ZE2UR_CALL(zeCommandListHostSynchronize,
               (commandListLocked->getZeCommandList(), UINT64_MAX));
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

  auto commandListLocked = commandListManager.lock();

  auto zeSignalEvent =
      getSignalEvent(commandListLocked, phEvent, UR_COMMAND_MEM_IMAGE_WRITE);
  auto waitListView =
      getWaitListView(commandListLocked, phEventWaitList, numEventsInWaitList);

  auto [zeImage, zeRegion] =
      hImage->getRWRegion(origin, region, rowPitch, slicePitch);

  ZE2UR_CALL(zeCommandListAppendImageCopyFromMemory,
             (commandListLocked->getZeCommandList(), zeImage, pSrc, &zeRegion,
              zeSignalEvent, waitListView.num, waitListView.handles));

  if (blockingWrite) {
    ZE2UR_CALL(zeCommandListHostSynchronize,
               (commandListLocked->getZeCommandList(), UINT64_MAX));
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

  auto commandListLocked = commandListManager.lock();
  auto zeSignalEvent =
      getSignalEvent(commandListLocked, phEvent, UR_COMMAND_MEM_IMAGE_COPY);
  auto waitListView =
      getWaitListView(commandListLocked, phEventWaitList, numEventsInWaitList);

  auto desc = ur_mem_image_t::getCopyRegions(*hImageSrc, *hImageDst, srcOrigin,
                                             dstOrigin, region);

  auto [zeImageSrc, zeRegionSrc] = desc.src;
  auto [zeImageDst, zeRegionDst] = desc.dst;

  ZE2UR_CALL(zeCommandListAppendImageCopyRegion,
             (commandListLocked->getZeCommandList(), zeImageDst, zeImageSrc,
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

  std::scoped_lock<ur_shared_mutex> lock(hBuffer->getMutex());

  auto commandListLocked = commandListManager.lock();
  auto zeSignalEvent =
      getSignalEvent(commandListLocked, phEvent, UR_COMMAND_MEM_BUFFER_MAP);

  auto waitListView =
      getWaitListView(commandListLocked, phEventWaitList, numEventsInWaitList);

  auto pDst = ur_cast<char *>(hBuffer->mapHostPtr(
      mapFlags, offset, size, [&](void *src, void *dst, size_t size) {
        ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                          (commandListLocked->getZeCommandList(), dst, src,
                           size, nullptr, waitListView.num,
                           waitListView.handles));
        waitListView.clear();
      }));
  *ppRetMap = pDst;

  if (waitListView) {
    // If memory was not migrated, we need to wait on the events here.
    ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
               (commandListLocked->getZeCommandList(), waitListView.num,
                waitListView.handles));
  }

  if (zeSignalEvent) {
    ZE2UR_CALL(zeCommandListAppendSignalEvent,
               (commandListLocked->getZeCommandList(), zeSignalEvent));
  }

  if (blockingMap) {
    ZE2UR_CALL(zeCommandListHostSynchronize,
               (commandListLocked->getZeCommandList(), UINT64_MAX));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueMemUnmap(
    ur_mem_handle_t hMem, void *pMappedPtr, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueMemUnmap");

  auto hBuffer = hMem->getBuffer();

  auto commandListLocked = commandListManager.lock();

  auto zeSignalEvent =
      getSignalEvent(commandListLocked, phEvent, UR_COMMAND_MEM_UNMAP);

  auto waitListView =
      getWaitListView(commandListLocked, phEventWaitList, numEventsInWaitList);

  // TODO: currently unmapHostPtr deallocates memory immediately,
  // since the memory might be used by the user, we need to make sure
  // all dependencies are completed.
  ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
             (commandListLocked->getZeCommandList(), waitListView.num,
              waitListView.handles));
  waitListView.clear();

  hBuffer->unmapHostPtr(pMappedPtr, [&](void *src, void *dst, size_t size) {
    ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                      (commandListLocked->getZeCommandList(), dst, src, size,
                       nullptr, waitListView.num, waitListView.handles));
  });
  if (zeSignalEvent) {
    ZE2UR_CALL(zeCommandListAppendSignalEvent,
               (commandListLocked->getZeCommandList(), zeSignalEvent));
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueUSMFill(
    void *pMem, size_t patternSize, const void *pPattern, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueUSMFill");

  auto commandListLocked = commandListManager.lock();
  UR_CALL(commandListLocked->appendUSMFill(pMem, patternSize, pPattern, size,
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

  auto commandListLocked = commandListManager.lock();
  UR_CALL(commandListLocked->appendUSMMemcpy(blocking, pDst, pSrc, size,
                                             numEventsInWaitList,
                                             phEventWaitList, phEvent));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueUSMPrefetch(
    const void *pMem, size_t size, ur_usm_migration_flags_t flags,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueUSMPrefetch");
  auto commandListLocked = commandListManager.lock();
  UR_CALL(commandListLocked->appendUSMPrefetch(
      pMem, size, flags, numEventsInWaitList, phEventWaitList, phEvent));
  return UR_RESULT_SUCCESS;
}

ur_result_t
ur_queue_immediate_in_order_t::enqueueUSMAdvise(const void *pMem, size_t size,
                                                ur_usm_advice_flags_t advice,
                                                ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueUSMAdvise");

  auto commandListLocked = commandListManager.lock();
  UR_CALL(commandListLocked->appendUSMAdvise(pMem, size, advice, 0, nullptr,
                                             phEvent));
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueUSMFill2D(
    void * /*pMem*/, size_t /*pitch*/, size_t /*patternSize*/,
    const void * /*pPattern*/, size_t /*width*/, size_t /*height*/,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_event_handle_t * /*phEvent*/) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueUSMMemcpy2D(
    bool blocking, void *pDst, size_t dstPitch, const void *pSrc,
    size_t srcPitch, size_t width, size_t height, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueUSMMemcpy2D");
  auto commandListLocked = commandListManager.lock();
  UR_CALL(commandListLocked->appendUSMMemcpy2D(
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
    ur_program_handle_t /*hProgram*/, const char * /*pipe_symbol*/,
    bool /*blocking*/, void * /*pDst*/, size_t /*size*/,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_event_handle_t * /*phEvent*/) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueWriteHostPipe(
    ur_program_handle_t /*hProgram*/, const char * /*pipe_symbol*/,
    bool /*blocking*/, void * /*pSrc*/, size_t /*size*/,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_event_handle_t * /*phEvent*/) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueUSMAllocHelper(
    ur_usm_pool_handle_t pPool, const size_t size,
    const ur_exp_async_usm_alloc_properties_t *, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, void **ppMem,
    ur_event_handle_t *phEvent, ur_usm_type_t type) {
  auto commandListLocked = commandListManager.lock();

  if (!pPool) {
    pPool = hContext->getAsyncPool();
  }

  auto device = (type == UR_USM_TYPE_HOST) ? nullptr : hDevice;

  ur_event_handle_t originAllocEvent = nullptr;
  auto asyncAlloc = pPool->allocateEnqueued(hContext, this, true, device,
                                            nullptr, type, size);
  if (!asyncAlloc) {
    auto Ret = pPool->allocate(hContext, device, nullptr, type, size, ppMem);
    if (Ret) {
      return Ret;
    }
  } else {
    std::tie(*ppMem, originAllocEvent) = *asyncAlloc;
  }

  auto waitListView = getWaitListView(commandListLocked, phEventWaitList,
                                      numEventsInWaitList, originAllocEvent);

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

  auto zeSignalEvent = getSignalEvent(commandListLocked, phEvent, commandType);
  auto [pWaitEvents, numWaitEvents] = waitListView;

  if (numWaitEvents > 0) {
    ZE2UR_CALL(
        zeCommandListAppendWaitOnEvents,
        (commandListLocked->getZeCommandList(), numWaitEvents, pWaitEvents));
  }
  if (zeSignalEvent) {
    ZE2UR_CALL(zeCommandListAppendSignalEvent,
               (commandListLocked->getZeCommandList(), zeSignalEvent));
  }
  if (originAllocEvent) {
    originAllocEvent->release();
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueUSMDeviceAllocExp(
    ur_usm_pool_handle_t pPool, const size_t size,
    const ur_exp_async_usm_alloc_properties_t *pProperties,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    void **ppMem, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_in_order_t::enqueueUSMDeviceAllocExp");

  return enqueueUSMAllocHelper(pPool, size, pProperties, numEventsInWaitList,
                               phEventWaitList, ppMem, phEvent,
                               UR_USM_TYPE_DEVICE);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueUSMSharedAllocExp(
    ur_usm_pool_handle_t pPool, const size_t size,
    const ur_exp_async_usm_alloc_properties_t *pProperties,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    void **ppMem, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_in_order_t::enqueueUSMSharedAllocExp");

  return enqueueUSMAllocHelper(pPool, size, pProperties, numEventsInWaitList,
                               phEventWaitList, ppMem, phEvent,
                               UR_USM_TYPE_SHARED);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueUSMHostAllocExp(
    ur_usm_pool_handle_t pPool, const size_t size,
    const ur_exp_async_usm_alloc_properties_t *pProperties,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    void **ppMem, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueUSMHostAllocExp");

  return enqueueUSMAllocHelper(pPool, size, pProperties, numEventsInWaitList,
                               phEventWaitList, ppMem, phEvent,
                               UR_USM_TYPE_HOST);
}

ur_result_t ur_queue_immediate_in_order_t::enqueueUSMFreeExp(
    ur_usm_pool_handle_t, void *pMem, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::enqueueUSMFreeExp");
  auto commandListLocked = commandListManager.lock();
  ur_event_handle_t internalEvent = nullptr;
  if (phEvent == nullptr) {
    phEvent = &internalEvent;
  }

  auto zeSignalEvent = getSignalEvent(commandListLocked, phEvent,
                                      UR_COMMAND_ENQUEUE_USM_FREE_EXP);
  auto [pWaitEvents, numWaitEvents] =
      getWaitListView(commandListLocked, phEventWaitList, numEventsInWaitList);

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
  if (internalEvent == nullptr) {
    // When the output event is used instead of an internal event, we need to
    // increment the refcount.
    (*phEvent)->RefCount.increment();
  }

  if (numWaitEvents > 0) {
    ZE2UR_CALL(
        zeCommandListAppendWaitOnEvents,
        (commandListLocked->getZeCommandList(), numWaitEvents, pWaitEvents));
  }

  ZE2UR_CALL(zeCommandListAppendSignalEvent,
             (commandListLocked->getZeCommandList(), zeSignalEvent));

  // Insert must be done after the signal event is appended.
  usmPool->asyncPool.insert(pMem, size, *phEvent, this);

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::bindlessImagesImageCopyExp(
    const void *pSrc, void *pDst, const ur_image_desc_t *pSrcImageDesc,
    const ur_image_desc_t *pDstImageDesc,
    const ur_image_format_t *pSrcImageFormat,
    const ur_image_format_t *pDstImageFormat,
    ur_exp_image_copy_region_t *pCopyRegion,
    ur_exp_image_copy_flags_t imageCopyFlags, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {

  auto commandListMgr = commandListManager.lock();

  auto zeSignalEvent =
      getSignalEvent(commandListMgr, phEvent, UR_COMMAND_MEM_IMAGE_COPY);
  auto waitListView =
      getWaitListView(commandListMgr, phEventWaitList, numEventsInWaitList);

  return bindlessImagesHandleCopyFlags(
      pSrc, pDst, pSrcImageDesc, pDstImageDesc, pSrcImageFormat,
      pDstImageFormat, pCopyRegion, imageCopyFlags,
      commandListMgr->getZeCommandList(), zeSignalEvent, waitListView.num,
      waitListView.handles);
}

ur_result_t
ur_queue_immediate_in_order_t::bindlessImagesWaitExternalSemaphoreExp(
    ur_exp_external_semaphore_handle_t /*hSemaphore*/, bool /*hasWaitValue*/,
    uint64_t /*waitValue*/, uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_event_handle_t * /*phEvent*/) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t
ur_queue_immediate_in_order_t::bindlessImagesSignalExternalSemaphoreExp(
    ur_exp_external_semaphore_handle_t /*hSemaphore*/, bool /*hasSignalValue*/,
    uint64_t /*signalValue*/, uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_event_handle_t * /*phEvent*/) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueCooperativeKernelLaunchHelper(
    ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hKernel->getProgramHandle(), UR_RESULT_ERROR_INVALID_NULL_POINTER);

  UR_ASSERT(workDim > 0, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);
  UR_ASSERT(workDim < 4, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);

  ze_kernel_handle_t hZeKernel = hKernel->getZeHandle(hDevice);

  std::scoped_lock<ur_shared_mutex> Lock(hKernel->Mutex);

  auto commandListLocked = commandListManager.lock();
  ze_group_count_t zeThreadGroupDimensions{1, 1, 1};
  uint32_t WG[3]{};
  UR_CALL(calculateKernelWorkDimensions(hZeKernel, hDevice,
                                        zeThreadGroupDimensions, WG, workDim,
                                        pGlobalWorkSize, pLocalWorkSize));

  auto zeSignalEvent =
      getSignalEvent(commandListLocked, phEvent, UR_COMMAND_KERNEL_LAUNCH);

  auto waitListView =
      getWaitListView(commandListLocked, phEventWaitList, numEventsInWaitList);

  auto memoryMigrate = [&](void *src, void *dst, size_t size) {
    ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                      (commandListLocked->getZeCommandList(), dst, src, size,
                       nullptr, waitListView.num, waitListView.handles));
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

  UR_CALL(hKernel->prepareForSubmission(hContext, hDevice, pGlobalWorkOffset,
                                        workDim, WG[0], WG[1], WG[2],
                                        memoryMigrate));

  TRACK_SCOPE_LATENCY("ur_queue_immediate_in_order_t::"
                      "zeCommandListAppendLaunchCooperativeKernel");
  ZE2UR_CALL(zeCommandListAppendLaunchCooperativeKernel,
             (commandListLocked->getZeCommandList(), hZeKernel,
              &zeThreadGroupDimensions, zeSignalEvent, waitListView.num,
              waitListView.handles));

  recordSubmittedKernel(hKernel);

  postSubmit(hZeKernel, pGlobalWorkOffset);

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueTimestampRecordingExp(
    bool blocking, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_in_order_t::enqueueTimestampRecordingExp");

  auto commandListLocked = commandListManager.lock();
  if (!phEvent) {
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  }
  getSignalEvent(commandListLocked, phEvent,
                 UR_COMMAND_TIMESTAMP_RECORDING_EXP);
  auto [pWaitEvents, numWaitEvents] =
      getWaitListView(commandListLocked, phEventWaitList, numEventsInWaitList);

  (*phEvent)->recordStartTimestamp();

  auto [timestampPtr, zeSignalEvent] =
      (*phEvent)->getEventEndTimestampAndHandle();

  ZE2UR_CALL(zeCommandListAppendWriteGlobalTimestamp,
             (commandListLocked->getZeCommandList(), timestampPtr,
              zeSignalEvent, numWaitEvents, pWaitEvents));

  if (blocking) {
    ZE2UR_CALL(zeCommandListHostSynchronize,
               (commandListLocked->getZeCommandList(), UINT64_MAX));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueGenericCommandListsExp(
    uint32_t numCommandLists, ze_command_list_handle_t *phCommandLists,
    ur_event_handle_t *phEvent, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_command_t callerCommand,
    ur_event_handle_t additionalWaitEvent) {
  TRACK_SCOPE_LATENCY(
      "ur_queue_immediate_in_order_t::enqueueGenericCommandListsExp");

  auto commandListLocked = commandListManager.lock();

  auto zeSignalEvent =
      getSignalEvent(commandListLocked, phEvent, callerCommand);
  auto [pWaitEvents, numWaitEvents] =
      getWaitListView(commandListLocked, phEventWaitList, numEventsInWaitList,
                      additionalWaitEvent);

  ZE2UR_CALL(zeCommandListImmediateAppendCommandListsExp,
             (commandListLocked->getZeCommandList(), numCommandLists,
              phCommandLists, zeSignalEvent, numWaitEvents, pWaitEvents));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueCommandBufferExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {

  auto commandListLocked = hCommandBuffer->commandListManager.lock();
  ze_command_list_handle_t commandBufferCommandList =
      commandListLocked->getZeCommandList();
  ur_event_handle_t internalEvent = nullptr;
  if (phEvent == nullptr) {
    phEvent = &internalEvent;
  }
  ur_event_handle_t executionEvent =
      hCommandBuffer->getExecutionEventUnlocked();

  UR_CALL(enqueueGenericCommandListsExp(
      1, &commandBufferCommandList, phEvent, numEventsInWaitList,
      phEventWaitList, UR_COMMAND_ENQUEUE_COMMAND_BUFFER_EXP, executionEvent));
  UR_CALL(hCommandBuffer->registerExecutionEventUnlocked(*phEvent));
  if (internalEvent != nullptr) {
    internalEvent->release();
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_immediate_in_order_t::enqueueNativeCommandExp(
    ur_exp_enqueue_native_command_function_t, void *, uint32_t,
    const ur_mem_handle_t *, const ur_exp_enqueue_native_command_properties_t *,
    uint32_t, const ur_event_handle_t *, ur_event_handle_t *) {
  UR_LOG_LEGACY(
      ERR, logger::LegacyMessage("[UR][L0_v2] {} function not implemented!"),
      "{} function not implemented!", __FUNCTION__);

  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
} // namespace v2
