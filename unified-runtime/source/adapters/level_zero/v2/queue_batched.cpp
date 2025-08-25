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
#include "command_buffer.hpp"
#include "command_list_cache.hpp"
#include "command_list_manager.hpp"
#include "event.hpp"
#include "event_pool.hpp"
#include "kernel.hpp"
#include "lockable.hpp"
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

ur_queue_batched_t::ur_queue_batched_t(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, uint32_t ordinal,
    ze_command_queue_priority_t priority, std::optional<int32_t> index,
    event_flags_t eventFlags, ur_queue_flags_t flags)
    : regularCmdListDesc(v2::command_list_desc_t{
          true /* isInOrder*/, ordinal /* Ordinal*/,
          true /* copyOffloadEnable*/, false /*isMutable*/}),
      currentCmdLists(
          hContext, hDevice,
          /* regular command list*/
          hContext->getCommandListCache().getRegularCommandList(
              hDevice->ZeDevice, regularCmdListDesc),
          /* command list immediate*/
          hContext->getCommandListCache().getImmediateCommandList(
              hDevice->ZeDevice,
              {true, ordinal, true /* always enable copy offload */},
              ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, priority, index)

      ) {
  TRACK_SCOPE_LATENCY("ur_queue_batched_t::constructor");

  // TODO common code?
  if (!hContext->getPlatform()->ZeCommandListImmediateAppendExt.Supported) {
    UR_LOG(ERR, "Adapter v2 is used but the current driver does not support "
                "the zeCommandListImmediateAppendCommandListsExp entrypoint.");
    throw UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  this->hContext = hContext;
  this->hDevice = hDevice;

  // runBatches.reserve(default_num_batches);

  this->flags = flags;

  eventPoolRegular = hContext->getEventPoolCache(PoolCacheType::Regular)
                         .borrow(hDevice->Id.value(), v2::EVENT_FLAGS_COUNTER);
}

ur_event_handle_t ur_queue_batched_t::createEventIfRequestedRegular(
    ur_event_handle_t *phEvent, ur_event_generation_t batch_generation) {
  TRACK_SCOPE_LATENCY("ur_queue_batched_t::createEventIfRequested");

  if (phEvent == nullptr) {
    return nullptr;
  }

  (*phEvent) = eventPoolRegular->allocate();
  (*phEvent)->setQueue(this);
  (*phEvent)->setBatch(batch_generation);

  return (*phEvent);
}

ur_result_t
ur_queue_batched_t::renewRegularUnlocked(locked<batch_manager> &batchLocked) {
  TRACK_SCOPE_LATENCY("ur_queue_batched_t::renewRegularUnlocked");

  batchLocked->regularGenerationNumber++;

  // save the previous regular for execution
  // renew regular
  batchLocked->runBatches.push_back(
      batchLocked->activeBatch
          .releaseCommandList()); // std::move(batchLocked->regularBatch));
  batchLocked->activeBatch.replaceCommandList(getNewRegularCmdList());

  return UR_RESULT_SUCCESS;
}

ur_result_t enqueueCurrentBatchUnlocked(ze_command_list_handle_t immediateList,
                                        ze_command_list_handle_t regularList) {
  TRACK_SCOPE_LATENCY("ur_queue_batched_t::enqueueCurrentBatchUnlocked");

  {
    TRACK_SCOPE_LATENCY(
        "ur_queue_batched_t::enqueueCurrentBatchUnlocked_finalize");
    // finalize
    ZE2UR_CALL(zeCommandListClose, (regularList));
  }
  {
    TRACK_SCOPE_LATENCY(
        "ur_queue_batched_t::enqueueCurrentBatchUnlocked_runBatchAppend");
    // run batch
    ZE2UR_CALL(zeCommandListImmediateAppendCommandListsExp,
               (immediateList, 1, &regularList, nullptr, 0, nullptr));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t
ur_queue_batched_t::runBatchIfActive(ur_event_generation_t batch_generation) {
  TRACK_SCOPE_LATENCY("ur_queue_batched_t::runBatchIfActive");

  auto batchLocked = currentCmdLists.lock();

  if (batch_generation != batchLocked->regularGenerationNumber) {
    // the batch must have been already run
    return UR_RESULT_SUCCESS;
  }

  // auto regularList = batchLocked->regularBatch.getZeCommandList();
  UR_CALL(
      enqueueCurrentBatchUnlocked(batchLocked->immediateList.getZeCommandList(),
                                  batchLocked->activeBatch.getZeCommandList()));

  return renewRegularUnlocked(batchLocked);
}

ur_result_t ur_queue_batched_t::enqueueKernelLaunch(
    ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numPropsInLaunchPropList,
    const ur_kernel_launch_property_t *launchPropList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {

  TRACK_SCOPE_LATENCY("ur_queue_batched_t::enqueueKernelLaunch");
  auto currentRegular = currentCmdLists.lock();
  UR_CALL(currentRegular->activeBatch.appendKernelLaunch(
      hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize,
      numPropsInLaunchPropList, launchPropList, numEventsInWaitList,
      phEventWaitList,
      createEventIfRequestedRegular(
          phEvent,
          currentRegular->regularGenerationNumber))); // nullptr));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_batched_t::queueFinishBatchAndPoolsUnlocked(
    ze_command_list_handle_t immediateList,
    ze_command_list_handle_t regularList) {
  TRACK_SCOPE_LATENCY("ur_queue_batched_t::queueFinishBatchAndPoolsUnlocked");

  enqueueCurrentBatchUnlocked(immediateList, regularList);

  {
    // TRACK_SCOPE_LATENCY(
    //     "ur_queue_batched_t::queueFinishBatchAndPoolsUnlocked_hostSynchronize");
    TRACK_SCOPE_LATENCY(
        "ur_queue_batched_t::hostSynchronize");
    // finish queue
    ZE2UR_CALL(zeCommandListHostSynchronize, (immediateList, UINT64_MAX));
  }

  {
    // TRACK_SCOPE_LATENCY(
    //     "ur_queue_batched_t::queueFinishBatchAndPoolsUnlocked_asyncPools");
    TRACK_SCOPE_LATENCY(
        "ur_queue_batched_t::asyncPools");
    hContext->getAsyncPool()->cleanupPoolsForQueue(this);
    hContext->forEachUsmPool([this](ur_usm_pool_handle_t hPool) {
      hPool->cleanupPoolsForQueue(this);
      return true;
    });
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t
ur_queue_batched_t::queueFinishUnlocked(locked<batch_manager> &batchLocked) {
  // auto regularCmdlist = (*batchLocked)->regularBatch.getZeCommandList();
  TRACK_SCOPE_LATENCY("ur_queue_batched_t::queueFinishUnlocked");

  UR_CALL(queueFinishBatchAndPoolsUnlocked(
      batchLocked->immediateList.getZeCommandList(),
      batchLocked->activeBatch.getZeCommandList()));

  {
    // TRACK_SCOPE_LATENCY(
    //     "ur_queue_batched_t::queueFinishUnlocked_releaseSubmittedKernels");
    TRACK_SCOPE_LATENCY(
        "ur_queue_batched_t::releaseSubmittedKernels");
    UR_CALL(batchLocked->immediateList.releaseSubmittedKernels());
  }

  // return renewRegularUnlocked(batchLocked);
  {
    TRACK_SCOPE_LATENCY(
        "ur_queue_batched_t::queueFinishUnlocked_resetRegCmdlist");
    ZE2UR_CALL(zeCommandListReset,
               (batchLocked->activeBatch.getZeCommandList()));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_batched_t::queueFinish() {
  try {
    TRACK_SCOPE_LATENCY("ur_queue_batched_t::queueFinish");
    // finish current batch
    auto lockedBatches = currentCmdLists.lock();
    return queueFinishUnlocked(lockedBatches);

    // ze_command_list_handle_t regularCmdlist =
    // TODO sth better than lvalue?
    // lockedBatches->regularBatch.getZeCommandList();

    // TODO UR_CALL_THROWS somewhere?

    ////////

    // UR_CALL(queueFinishBatchAndPoolsUnlocked(
    //     lockedBatches->immediateList.getZeCommandList(),
    //     lockedBatches->activeBatch.getZeCommandList()));

    // UR_CALL(lockedBatches->immediateList.releaseSubmittedKernels());

    // return renewRegularUnlocked(lockedBatches);
  } catch (...) {
    return exceptionToResult(std::current_exception());
  }
}

ur_queue_batched_t::~ur_queue_batched_t() {
  try {
    UR_CALL_THROWS(queueFinish());
  } catch (...) {
    // Ignore errors during destruction
  }
}

ur_result_t ur_queue_batched_t::enqueueMemBufferRead(
    ur_mem_handle_t hBuffer, bool blockingRead, size_t offset, size_t size,
    void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  try {
    TRACK_SCOPE_LATENCY("ur_queue_batched_t::enqueueMemBufferRead");

    auto lockedBatches = currentCmdLists.lock();
    UR_CALL(lockedBatches->activeBatch.appendMemBufferRead(
        hBuffer, false, offset, size, pDst, numEventsInWaitList,
        phEventWaitList,
        createEventIfRequestedRegular(
            phEvent, lockedBatches->regularGenerationNumber))); // nullptr));

    if (blockingRead) {
      UR_CALL_THROWS(queueFinishUnlocked(lockedBatches));
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

  // -------------- this is not my comment --------------------

  // the same issue as in urCommandBufferAppendKernelLaunchExp
  // sync mechanic can be ignored, because all lists are in-order
  // Responsibility of UMD to offload to copy engine

  // -------------- end of not my comment ---------------------
  TRACK_SCOPE_LATENCY("ur_queue_batched_t::enqueueMemBufferWrite");

  auto lockedBatches = currentCmdLists.lock();

  UR_CALL(lockedBatches->activeBatch.appendMemBufferWrite(
      hBuffer, false, offset, size, pSrc, numEventsInWaitList, phEventWaitList,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatches->regularGenerationNumber)));

  if (blockingWrite) {
    UR_CALL_THROWS(queueFinishUnlocked(lockedBatches));
  }

  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t ur_queue_batched_t::enqueueMemBufferFill(
    ur_mem_handle_t hBuffer, const void *pPattern, size_t patternSize,
    size_t offset, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) try {
  TRACK_SCOPE_LATENCY("ur_queue_batched_t::enqueueMemBufferFill");

  auto lockedBatch = currentCmdLists.lock();
  UR_CALL(lockedBatch->activeBatch.appendMemBufferFill(
      hBuffer, pPattern, patternSize, offset, size, numEventsInWaitList,
      phEventWaitList,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatch->regularGenerationNumber)));

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
        (currentCmdLists.get_no_lock()->immediateList.getZeCommandList(), 0));
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
      currentCmdLists.get_no_lock()->immediateList.getZeCommandList());
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_batched_t::queueFlush() { return UR_RESULT_SUCCESS; }

} // namespace v2
