//===--------------- queue_batched.cpp - Level Zero Adapter ---------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "queue_batched.hpp"
#include "adapters/level_zero/common.hpp"
#include "adapters/level_zero/v2/command_list_cache.hpp"
#include "adapters/level_zero/v2/command_list_manager.hpp"
#include "adapters/level_zero/v2/lockable.hpp"
#include "command_buffer.hpp"
#include "kernel.hpp"
#include "memory.hpp"
#include "ur.hpp"

#include "../common/latency_tracker.hpp"
#include "../helpers/kernel_helpers.hpp"
#include "../image_common.hpp"

#include "../program.hpp"
#include "../ur_interface_loader.hpp"
#include "ur_api.h"
#include "ze_api.h"
#include <cstddef>
#include <cstdint>

namespace v2 {

// TODO move constants from hpp

// TODO if ordinal not provided: v2:: uint32_t getZeOrdinal(ur_device_handle_t
// hDevice)
ur_queue_batched_t::ur_queue_batched_t(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, uint32_t ordinal,
    ze_command_queue_priority_t priority, std::optional<int32_t> index,
    event_flags_t eventFlags, ur_queue_flags_t flags)
    // : hContext(hContext), hDevice(hDevice),
    : commandListManagerImmediate(
          hContext, hDevice,
          hContext->getCommandListCache().getImmediateCommandList(
              hDevice->ZeDevice,
              {true, ordinal, true /* always enable copy offload */},
              ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, priority, index)),
      // TODO initialize desc
      currentBatch(
          hContext, hDevice,
          /* regular command list*/
          hContext->getCommandListCache().getRegularCommandList(
              hDevice->ZeDevice,
              v2::command_list_desc_t{
                  true /* isInOrder*/,
                  (uint32_t)hDevice
                      ->QueueGroup[ur_device_handle_t_::queue_group_info_t::
                                       type::Compute]
                      .ZeOrdinal /* Ordinal*/,
                  true /* copyOffloadEnable*/, false /*isMutable*/}),
          /* command list immediate*/
          hContext->getCommandListCache().getImmediateCommandList(
              hDevice->ZeDevice,
              {true, ordinal, true /* always enable copy offload */},
              ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, priority, index)

      ) {
  // {
  // TODO common code?
  if (!hContext->getPlatform()->ZeCommandListImmediateAppendExt.Supported) {
    UR_LOG(ERR, "Adapter v2 is used but the current driver does not support "
                "the zeCommandListImmediateAppendCommandListsExp entrypoint.");
    throw UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  // TODO ordinal is already provided
  using queue_group_type = ur_device_handle_t_::queue_group_info_t::type;
  uint32_t queueGroupOrdinal =
      hDevice->QueueGroup[queue_group_type::Compute].ZeOrdinal;
  v2::command_list_desc_t listDesc;
  listDesc.IsInOrder = true;
  listDesc.Ordinal = queueGroupOrdinal;
  listDesc.CopyOffloadEnable = true;
  listDesc.Mutable = false;

  v2::raii::command_list_unique_handle zeCommandList =
      hContext->getCommandListCache().getRegularCommandList(hDevice->ZeDevice,
                                                            listDesc);

  this->hContext = hContext;
  this->hDevice = hDevice;
  commandListManagerCurrentRegular =
      std::make_unique<lockable<ur_command_list_manager>>(
          hContext, hDevice,
          std::forward<v2::raii::command_list_unique_handle>(zeCommandList));

  this->regularCmdListDesc = listDesc;

  runBatches = std::vector<ur_command_list_manager>();
  runBatches.reserve(default_num_batches);

  this->flags = flags;

  // eventPoolRegular(context->getEventPoolCache(PoolCacheType::Regular)
  //               .borrow(device->Id.value(),
  //                       isInOrder ? v2::EVENT_FLAGS_COUNTER : 0))
  // always in order
  // TODO remove immediate
  eventPoolImmediate = hContext->getEventPoolCache(PoolCacheType::Immediate)
                           .borrow(hDevice->Id.value(), eventFlags);
  eventPoolRegular = hContext->getEventPoolCache(PoolCacheType::Regular)
                         .borrow(hDevice->Id.value(), v2::EVENT_FLAGS_COUNTER);
  // TODO make const? always copy? - function needs const
}

// TODO use orginal create event if requested
ur_event_handle_t ur_queue_batched_t::createEventIfRequested(
    event_pool *eventPool, ur_event_handle_t *phEvent, ur_queue_t_ *queue,
    int64_t batch_generation) {
  if (phEvent == nullptr) {
    return nullptr;
  }

  (*phEvent) = eventPool->allocate();
  (*phEvent)->setQueue(queue);
  (*phEvent)->setBatch(batch_generation);

  return (*phEvent);
}

ur_event_handle_t
ur_queue_batched_t::createEventIfRequestedRegular(ur_event_handle_t *phEvent,
                                                  int64_t batch_generation) {
  if (phEvent == nullptr) {
    return nullptr;
  }

  (*phEvent) = eventPoolRegular->allocate();
  (*phEvent)->setQueue(this);
  (*phEvent)->setBatch(batch_generation);

  return (*phEvent);
}

locked<Batch> ur_queue_batched_t::renewRegular(locked<Batch> batchLocked) {
  batchLocked->generation++;

  // TODO replace with unlocked functions
  // save regular for execution
  // renew regular
  runBatches.push_back(std::move(batchLocked->regularBatch));
  batchLocked->regularBatch =
      ur_command_list_manager(hContext, hDevice, getNewRegularCmdList());

  return batchLocked;
}

ur_result_t
ur_queue_batched_t::runBatchIfCurrentBatch(int64_t batch_generation) {
  auto batchLocked = currentBatch.lock();

  if (batch_generation == batchLocked->generation) {
    auto cmdlist = batchLocked->regularBatch.getZeCommandList();

    // no syncpoints to synchronize
    ZE2UR_CALL(zeCommandListClose, (cmdlist));
    // run batch
    batchLocked->immediateList.appendRegular(&cmdlist);
  }

  // else: it must be older and already run

  batchLocked->generation++;

  // TODO change to only cmdlists, without command managers
  // TODO std::optional
  // save regular for execution
  // renew regular
  runBatches.push_back(std::move(batchLocked->regularBatch));
  batchLocked->regularBatch =
      ur_command_list_manager(hContext, hDevice, getNewRegularCmdList());

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_batched_t::enqueueKernelLaunch(
    ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numPropsInLaunchPropList,
    const ur_kernel_launch_property_t *launchPropList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {

  auto currentRegular = currentBatch.lock();
  UR_CALL(currentRegular->regularBatch.appendKernelLaunch(
      hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize,
      numPropsInLaunchPropList, launchPropList, numEventsInWaitList,
      phEventWaitList,
      ur_queue_batched_t::createEventIfRequested(
          eventPoolRegular.get(), phEvent, this,
          currentRegular->generation))); // nullptr));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_batched_t::queueFinish() {
  try {

    // finish current batch
    auto lockedBatches = currentBatch.lock();

    ze_command_list_handle_t cmdlist =
        lockedBatches->regularBatch.getZeCommandList();
    // finalize
    ZE2UR_CALL(zeCommandListClose, (cmdlist));

    // run current batch
    lockedBatches->immediateList.appendRegular(&cmdlist);

    // finish queue
    ZE2UR_CALL(zeCommandListHostSynchronize,
               (lockedBatches->immediateList.getZeCommandList(), UINT64_MAX));

    hContext->getAsyncPool()->cleanupPoolsForQueue(this);
    hContext->forEachUsmPool([this](ur_usm_pool_handle_t hPool) {
      hPool->cleanupPoolsForQueue(this);
      return true;
    });

    UR_CALL(lockedBatches->immediateList.releaseSubmittedKernels());

    // TODO removingf double lock but looks like trash - change to unlocked
    lockedBatches = renewRegular(std::move(lockedBatches));

    return UR_RESULT_SUCCESS;
  } catch (...) {
    return exceptionToResult(std::current_exception());
  }
}

ur_queue_batched_t::~ur_queue_batched_t() {
  try {
    UR_CALL_THROWS(queueFinish());
    // TODO regular is renewed, unnecessarily
  } catch (...) {
    // Ignore errors during destruction
  }
}

ur_result_t ur_queue_batched_t::enqueueMemBufferRead(
    ur_mem_handle_t hBuffer, bool blockingRead, size_t offset, size_t size,
    void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  try {
    // TODO remove double lock acquisition
    {
      auto lockedBatches = currentBatch.lock();
      UR_CALL(lockedBatches->regularBatch.appendMemBufferRead(
          hBuffer, false, offset, size, pDst, numEventsInWaitList,
          phEventWaitList,
          createEventIfRequestedRegular(
              phEvent, lockedBatches->generation))); // nullptr));
    }

    if (blockingRead) {
      UR_CALL_THROWS(queueFinish());
    }

    return UR_RESULT_SUCCESS;
  } catch (...) {
    return exceptionToResult(std::current_exception());
  }
}

ur_result_t ur_queue_batched_t::enqueueMemBufferWrite(
    ur_mem_handle_t hBuffer, bool blockingWrite, size_t offset, size_t size,
    const void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) try {

  // this is not my comment
  // the same issue as in urCommandBufferAppendKernelLaunchExp
  // sync mechanic can be ignored, because all lists are in-order
  // Responsibility of UMD to offload to copy engine
  // end of not my comment

  // TODO remove double lock acquisition - unlocked functions
  {

    auto lockedBatches = currentBatch.lock();

    // TODO create event if requested - pass only phEvent?
    UR_CALL(lockedBatches->regularBatch.appendMemBufferWrite(
        hBuffer, false, offset, size, pSrc, numEventsInWaitList,
        phEventWaitList,
        createEventIfRequested(eventPoolRegular.get(), phEvent, this,
                               lockedBatches->generation)));
  }

  if (blockingWrite) {
    UR_CALL_THROWS(queueFinish());
  }

  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t ur_queue_batched_t::enqueueMemBufferFill(
    ur_mem_handle_t hBuffer, const void *pPattern, size_t patternSize,
    size_t offset, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) try {

  auto lockedBatch = currentBatch.lock();
  UR_CALL(lockedBatch->regularBatch.appendMemBufferFill(
      hBuffer, pPattern, patternSize, offset, size, numEventsInWaitList,
      phEventWaitList,
      createEventIfRequestedRegular(phEvent, lockedBatch->generation)));

  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

// from in_order.cpp

ur_result_t ur_queue_batched_t::queueGetInfo(ur_queue_info_t propName,
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
    return ReturnValue(uint32_t{RefCount.getCount()});
  case UR_QUEUE_INFO_FLAGS:
    return ReturnValue(flags);
  case UR_QUEUE_INFO_SIZE:
  case UR_QUEUE_INFO_DEVICE_DEFAULT:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  case UR_QUEUE_INFO_EMPTY: {
    auto status = ZE_CALL_NOCHECK(
        zeCommandListHostSynchronize,
        (commandListManagerImmediate.get_no_lock()->getZeCommandList(), 0));
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

ur_result_t
ur_queue_batched_t::queueGetNativeHandle(ur_queue_native_desc_t * /*pDesc*/,
                                         ur_native_handle_t *phNativeQueue) {
  *phNativeQueue = reinterpret_cast<ur_native_handle_t>(
      commandListManagerImmediate.get_no_lock()->getZeCommandList());
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_batched_t::queueFlush() { return UR_RESULT_SUCCESS; }

} // namespace v2