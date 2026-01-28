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

#include "../common/latency_tracker.hpp"
#include "../helpers/kernel_helpers.hpp"
#include "../image_common.hpp"

#include "../program.hpp"
#include "../ur_interface_loader.hpp"
#include "ur.hpp"
#include "ur_api.h"
#include "ze_api.h"
#include <cstddef>
#include <cstdint>
#include <tuple>

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

  this->flags = flags;

  eventPoolRegular = hContext->getEventPoolCache(PoolCacheType::Regular)
                         .borrow(hDevice->Id.value(), v2::EVENT_FLAGS_COUNTER);

  eventPoolImmediate = hContext->getEventPoolCache(PoolCacheType::Immediate)
                           .borrow(hDevice->Id.value(), eventFlags);
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

ur_event_handle_t ur_queue_batched_t::createEventAndRetainRegular(
    ur_event_handle_t *phEvent, ur_event_generation_t batch_generation) {
  auto hEvent = eventPoolRegular->allocate();
  hEvent->setQueue(this);
  hEvent->setBatch(batch_generation);

  if (phEvent) {
    (*phEvent) = hEvent;
    hEvent->retain();
  }

  return hEvent;
}

ur_result_t batch_manager::renewRegularUnlocked(
    v2::raii::command_list_unique_handle &&newRegularBatch) {
  TRACK_SCOPE_LATENCY("batch_manager::renewRegularUnlocked");

  regularGenerationNumber++;

  // save the previous regular command list for execution
  runBatches.push_back(activeBatch.releaseCommandList());
  // renew the regular command list (current batch)
  activeBatch.replaceCommandList(
      std::forward<v2::raii::command_list_unique_handle>(newRegularBatch));

  setBatchEmpty();

  return UR_RESULT_SUCCESS;
}

ur_result_t
ur_queue_batched_t::renewBatchUnlocked(locked<batch_manager> &batchLocked) {
  if (batchLocked->isLimitOfUsedCommandListsReached()) {
    return queueFinishUnlocked(batchLocked);
  } else {
    return batchLocked->renewRegularUnlocked(getNewRegularCmdList());
  }
}

ur_result_t batch_manager::enqueueCurrentBatchUnlocked() {
  TRACK_SCOPE_LATENCY("ur_queue_batched_t::enqueueCurrentBatchUnlocked");

  ze_command_list_handle_t regularList = activeBatch.getZeCommandList();
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
               (immediateList.getZeCommandList(), 1, &regularList, nullptr, 0,
                nullptr));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t
ur_queue_batched_t::onEventWaitListUse(ur_event_generation_t batch_generation) {
  TRACK_SCOPE_LATENCY("ur_queue_batched_t::onEventWaitListUse");

  auto batchLocked = currentCmdLists.lock();
  if (batchLocked->isCurrentGeneration(batch_generation)) {
    return queueFlushUnlocked(batchLocked);
  } else {
    return UR_RESULT_SUCCESS;
  }
}

ur_result_t ur_queue_batched_t::markIssuedCommandInBatch(
    locked<batch_manager> &batchLocked) {
  if (batchLocked->isLimitOfEnqueuedCommandsReached()) {
    UR_CALL(queueFinishUnlocked(batchLocked));

    batchLocked->setBatchEmpty();
  }

  batchLocked->markNextIssuedCommand();

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_batched_t::enqueueKernelLaunch(
    ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize,
    const ur_kernel_launch_ext_properties_t *launchPropList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {

  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);

  TRACK_SCOPE_LATENCY("ur_queue_batched_t::enqueueKernelLaunch");
  auto currentRegular = currentCmdLists.lock();

  markIssuedCommandInBatch(currentRegular);

  UR_CALL(currentRegular->getActiveBatch().appendKernelLaunch(
      hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize,
      launchPropList, waitListView,
      createEventIfRequestedRegular(phEvent,
                                    currentRegular->getCurrentGeneration())));

  return UR_RESULT_SUCCESS;
}

ur_result_t batch_manager::hostSynchronize() {
  TRACK_SCOPE_LATENCY("ur_queue_batched_t::hostSynchronize");

  ZE2UR_CALL(zeCommandListHostSynchronize,
             (immediateList.getZeCommandList(), UINT64_MAX));

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_batched_t::queueFinishPoolsUnlocked() {
  TRACK_SCOPE_LATENCY("ur_queue_batched_t::asyncPools");

  hContext->getAsyncPool()->cleanupPoolsForQueue(this);
  hContext->forEachUsmPool([this](ur_usm_pool_handle_t hPool) {
    hPool->cleanupPoolsForQueue(this);
    return true;
  });

  return UR_RESULT_SUCCESS;
}

ur_result_t batch_manager::batchFinish() {
  TRACK_SCOPE_LATENCY("ur_queue_batched_t::batchFinish");

  UR_CALL(activeBatch.releaseSubmittedKernels());

  if (!isActiveBatchEmpty()) {
    // Should have been enqueued as part of queueFinishUnlocked
    TRACK_SCOPE_LATENCY("ur_queue_batched_t::resetRegCmdlist");
    ZE2UR_CALL(zeCommandListReset, (activeBatch.getZeCommandList()));

    setBatchEmpty();
    regularGenerationNumber++;
  }

  runBatches.clear();

  return UR_RESULT_SUCCESS;
}

ur_result_t
ur_queue_batched_t::queueFinishUnlocked(locked<batch_manager> &batchLocked) {
  if (!batchLocked->isActiveBatchEmpty()) {
    UR_CALL(batchLocked->enqueueCurrentBatchUnlocked());
  }

  UR_CALL(batchLocked->hostSynchronize());

  UR_CALL(queueFinishPoolsUnlocked());

  return batchLocked->batchFinish();
}

ur_result_t ur_queue_batched_t::queueFinish() {
  try {
    TRACK_SCOPE_LATENCY("ur_queue_batched_t::queueFinish");
    // finish current batch
    auto lockedBatches = currentCmdLists.lock();
    return queueFinishUnlocked(lockedBatches);

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

    wait_list_view waitListView =
        wait_list_view(phEventWaitList, numEventsInWaitList, this);

    auto lockedBatches = currentCmdLists.lock();

    markIssuedCommandInBatch(lockedBatches);

    UR_CALL(lockedBatches->getActiveBatch().appendMemBufferRead(
        hBuffer, false, offset, size, pDst, waitListView,
        createEventIfRequestedRegular(phEvent,
                                      lockedBatches->getCurrentGeneration())));

    if (blockingRead) {
      UR_CALL(queueFinishUnlocked(lockedBatches));
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
  TRACK_SCOPE_LATENCY("ur_queue_batched_t::enqueueMemBufferWrite");

  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);

  auto lockedBatches = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatches);

  UR_CALL(lockedBatches->getActiveBatch().appendMemBufferWrite(
      hBuffer, false, offset, size, pSrc, waitListView,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatches->getCurrentGeneration())));

  if (blockingWrite) {
    UR_CALL(queueFinishUnlocked(lockedBatches));
  }

  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t ur_queue_batched_t::enqueueDeviceGlobalVariableWrite(
    ur_program_handle_t hProgram, const char *name, bool blockingWrite,
    size_t count, size_t offset, const void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);

  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  UR_CALL(lockedBatch->getActiveBatch().appendDeviceGlobalVariableWrite(
      hProgram, name, false, count, offset, pSrc, waitListView,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatch->getCurrentGeneration())));

  if (blockingWrite) {
    UR_CALL(queueFinishUnlocked(lockedBatch));
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_batched_t::enqueueDeviceGlobalVariableRead(
    ur_program_handle_t hProgram, const char *name, bool blockingRead,
    size_t count, size_t offset, void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);

  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  UR_CALL(lockedBatch->getActiveBatch().appendDeviceGlobalVariableRead(
      hProgram, name, false, count, offset, pDst, waitListView,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatch->getCurrentGeneration())));

  if (blockingRead) {
    UR_CALL(queueFinishUnlocked(lockedBatch));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_batched_t::enqueueMemBufferFill(
    ur_mem_handle_t hBuffer, const void *pPattern, size_t patternSize,
    size_t offset, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) try {
  TRACK_SCOPE_LATENCY("ur_queue_batched_t::enqueueMemBufferFill");
  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);

  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  return lockedBatch->getActiveBatch().appendMemBufferFill(
      hBuffer, pPattern, patternSize, offset, size, waitListView,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatch->getCurrentGeneration()));

} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t ur_queue_batched_t::enqueueUSMMemcpy(
    bool blocking, void *pDst, const void *pSrc, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);
  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  UR_CALL(lockedBatch->getActiveBatch().appendUSMMemcpy(
      false, pDst, pSrc, size, waitListView,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatch->getCurrentGeneration())));

  if (blocking) {
    UR_CALL(queueFinishUnlocked(lockedBatch));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_batched_t::enqueueUSMFreeExp(
    ur_usm_pool_handle_t pPool, void *pMem, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);
  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  UR_CALL(lockedBatch->getActiveBatch().appendUSMFreeExp(
      this, pPool, pMem, waitListView,
      createEventAndRetainRegular(phEvent,
                                  lockedBatch->getCurrentGeneration())));

  return queueFlushUnlocked(lockedBatch);
}

ur_result_t ur_queue_batched_t::enqueueMemBufferMap(
    ur_mem_handle_t hBuffer, bool blockingMap, ur_map_flags_t mapFlags,
    size_t offset, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent,
    void **ppRetMap) {

  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);
  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  UR_CALL(lockedBatch->getActiveBatch().appendMemBufferMap(
      hBuffer, false, mapFlags, offset, size, waitListView,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatch->getCurrentGeneration()),
      ppRetMap));

  if (blockingMap) {
    UR_CALL(queueFinishUnlocked(lockedBatch));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_batched_t::enqueueMemUnmap(
    ur_mem_handle_t hMem, void *pMappedPtr, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);
  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  return lockedBatch->getActiveBatch().appendMemUnmap(
      hMem, pMappedPtr, waitListView,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatch->getCurrentGeneration()));
}

ur_result_t ur_queue_batched_t::enqueueMemBufferReadRect(
    ur_mem_handle_t hBuffer, bool blockingRead, ur_rect_offset_t bufferOrigin,
    ur_rect_offset_t hostOrigin, ur_rect_region_t region, size_t bufferRowPitch,
    size_t bufferSlicePitch, size_t hostRowPitch, size_t hostSlicePitch,
    void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);
  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  UR_CALL(lockedBatch->getActiveBatch().appendMemBufferReadRect(
      hBuffer, false, bufferOrigin, hostOrigin, region, bufferRowPitch,
      bufferSlicePitch, hostRowPitch, hostSlicePitch, pDst, waitListView,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatch->getCurrentGeneration())));

  if (blockingRead) {
    UR_CALL(queueFinishUnlocked(lockedBatch));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_batched_t::enqueueMemBufferWriteRect(
    ur_mem_handle_t hBuffer, bool blockingWrite, ur_rect_offset_t bufferOrigin,
    ur_rect_offset_t hostOrigin, ur_rect_region_t region, size_t bufferRowPitch,
    size_t bufferSlicePitch, size_t hostRowPitch, size_t hostSlicePitch,
    void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {

  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);
  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  UR_CALL(lockedBatch->getActiveBatch().appendMemBufferWriteRect(
      hBuffer, false, bufferOrigin, hostOrigin, region, bufferRowPitch,
      bufferSlicePitch, hostRowPitch, hostSlicePitch, pSrc, waitListView,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatch->getCurrentGeneration())));

  if (blockingWrite) {
    UR_CALL(queueFinishUnlocked(lockedBatch));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_batched_t::enqueueUSMAdvise(const void *pMem, size_t size,
                                                 ur_usm_advice_flags_t advice,
                                                 ur_event_handle_t *phEvent) {
  wait_list_view emptyWaitList = wait_list_view(nullptr, 0, this);

  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  return lockedBatch->getActiveBatch().appendUSMAdvise(
      pMem, size, advice, emptyWaitList,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatch->getCurrentGeneration()));
}

ur_result_t ur_queue_batched_t::enqueueUSMMemcpy2D(
    bool blocking, void *pDst, size_t dstPitch, const void *pSrc,
    size_t srcPitch, size_t width, size_t height, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);
  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  UR_CALL(lockedBatch->getActiveBatch().appendUSMMemcpy2D(
      false, pDst, dstPitch, pSrc, srcPitch, width, height, waitListView,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatch->getCurrentGeneration())));

  if (blocking) {
    UR_CALL(queueFinishUnlocked(lockedBatch));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_batched_t::enqueueUSMFill2D(
    void *pMem, size_t pitch, size_t patternSize, const void *pPattern,
    size_t width, size_t height, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);
  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  return lockedBatch->getActiveBatch().appendUSMFill2D(
      pMem, pitch, patternSize, pPattern, width, height, waitListView,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatch->getCurrentGeneration()));
}

ur_result_t ur_queue_batched_t::enqueueUSMPrefetch(
    const void *pMem, size_t size, ur_usm_migration_flags_t flags,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);
  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  return lockedBatch->getActiveBatch().appendUSMPrefetch(
      pMem, size, flags, waitListView,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatch->getCurrentGeneration()));
}

ur_result_t ur_queue_batched_t::enqueueMemBufferCopyRect(
    ur_mem_handle_t hBufferSrc, ur_mem_handle_t hBufferDst,
    ur_rect_offset_t srcOrigin, ur_rect_offset_t dstOrigin,
    ur_rect_region_t region, size_t srcRowPitch, size_t srcSlicePitch,
    size_t dstRowPitch, size_t dstSlicePitch, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {

  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);
  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  return lockedBatch->getActiveBatch().appendMemBufferCopyRect(
      hBufferSrc, hBufferDst, srcOrigin, dstOrigin, region, srcRowPitch,
      srcSlicePitch, dstRowPitch, dstSlicePitch, waitListView,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatch->getCurrentGeneration()));
}

ur_result_t ur_queue_batched_t::enqueueEventsWaitWithBarrier(
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);
  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  if ((flags & UR_QUEUE_FLAG_PROFILING_ENABLE) != 0) {
    UR_CALL(lockedBatch->getActiveBatch().appendEventsWaitWithBarrier(
        waitListView, createEventIfRequestedRegular(
                          phEvent, lockedBatch->getCurrentGeneration())));
  } else {
    UR_CALL(lockedBatch->getActiveBatch().appendEventsWait(
        waitListView, createEventIfRequestedRegular(
                          phEvent, lockedBatch->getCurrentGeneration())));
  }

  return queueFlushUnlocked(lockedBatch);
}

ur_result_t
ur_queue_batched_t::enqueueEventsWait(uint32_t numEventsInWaitList,
                                      const ur_event_handle_t *phEventWaitList,
                                      ur_event_handle_t *phEvent) {
  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);

  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  UR_CALL(lockedBatch->getActiveBatch().appendEventsWait(
      waitListView, createEventIfRequestedRegular(
                        phEvent, lockedBatch->getCurrentGeneration())));

  return queueFlushUnlocked(lockedBatch);
}

ur_result_t ur_queue_batched_t::enqueueMemBufferCopy(
    ur_mem_handle_t hBufferSrc, ur_mem_handle_t hBufferDst, size_t srcOffset,
    size_t dstOffset, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);

  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  return lockedBatch->getActiveBatch().appendMemBufferCopy(
      hBufferSrc, hBufferDst, srcOffset, dstOffset, size, waitListView,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatch->getCurrentGeneration()));
}

ur_result_t ur_queue_batched_t::enqueueUSMFill(
    void *pMem, size_t patternSize, const void *pPattern, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);

  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  return lockedBatch->getActiveBatch().appendUSMFill(
      pMem, patternSize, pPattern, size, waitListView,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatch->getCurrentGeneration()));
}

ur_result_t ur_queue_batched_t::enqueueMemImageRead(
    ur_mem_handle_t hImage, bool blockingRead, ur_rect_offset_t origin,
    ur_rect_region_t region, size_t rowPitch, size_t slicePitch, void *pDst,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);

  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  UR_CALL(lockedBatch->getActiveBatch().appendMemImageRead(
      hImage, false, origin, region, rowPitch, slicePitch, pDst, waitListView,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatch->getCurrentGeneration())));

  if (blockingRead) {
    UR_CALL(queueFinishUnlocked(lockedBatch));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_batched_t::enqueueMemImageWrite(
    ur_mem_handle_t hImage, bool blockingWrite, ur_rect_offset_t origin,
    ur_rect_region_t region, size_t rowPitch, size_t slicePitch, void *pSrc,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);

  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  UR_CALL(lockedBatch->getActiveBatch().appendMemImageWrite(
      hImage, false, origin, region, rowPitch, slicePitch, pSrc, waitListView,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatch->getCurrentGeneration())));

  if (blockingWrite) {
    UR_CALL(queueFinishUnlocked(lockedBatch));
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_batched_t::enqueueMemImageCopy(
    ur_mem_handle_t hImageSrc, ur_mem_handle_t hImageDst,
    ur_rect_offset_t srcOrigin, ur_rect_offset_t dstOrigin,
    ur_rect_region_t region, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);

  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  return lockedBatch->getActiveBatch().appendMemImageCopy(
      hImageSrc, hImageDst, srcOrigin, dstOrigin, region, waitListView,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatch->getCurrentGeneration()));
}

ur_result_t ur_queue_batched_t::enqueueReadHostPipe(
    ur_program_handle_t hProgram, const char *pipe_symbol, bool blocking,
    void *pDst, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);

  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  UR_CALL(lockedBatch->getActiveBatch().appendReadHostPipe(
      hProgram, pipe_symbol, false, pDst, size, waitListView,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatch->getCurrentGeneration())));

  if (blocking) {
    UR_CALL(queueFinishUnlocked(lockedBatch));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_batched_t::enqueueWriteHostPipe(
    ur_program_handle_t hProgram, const char *pipe_symbol, bool blocking,
    void *pSrc, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);

  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  UR_CALL(lockedBatch->getActiveBatch().appendWriteHostPipe(
      hProgram, pipe_symbol, false, pSrc, size, waitListView,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatch->getCurrentGeneration())));

  if (blocking) {
    UR_CALL(queueFinishUnlocked(lockedBatch));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_batched_t::enqueueUSMDeviceAllocExp(
    ur_usm_pool_handle_t pPool, const size_t size,
    const ur_exp_async_usm_alloc_properties_t *pProperties,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    void **ppMem, ur_event_handle_t *phEvent) {
  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);

  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  UR_CALL(lockedBatch->getActiveBatch().appendUSMAllocHelper(
      this, pPool, size, pProperties, waitListView, ppMem,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatch->getCurrentGeneration()),
      UR_USM_TYPE_DEVICE));

  return queueFlushUnlocked(lockedBatch);
}

ur_result_t ur_queue_batched_t::enqueueUSMSharedAllocExp(
    ur_usm_pool_handle_t pPool, const size_t size,
    const ur_exp_async_usm_alloc_properties_t *pProperties,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    void **ppMem, ur_event_handle_t *phEvent) {

  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);

  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  UR_CALL(lockedBatch->getActiveBatch().appendUSMAllocHelper(
      this, pPool, size, pProperties, waitListView, ppMem,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatch->getCurrentGeneration()),
      UR_USM_TYPE_SHARED));

  return queueFlushUnlocked(lockedBatch);
}

ur_result_t ur_queue_batched_t::enqueueUSMHostAllocExp(
    ur_usm_pool_handle_t pPool, const size_t size,
    const ur_exp_async_usm_alloc_properties_t *pProperties,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    void **ppMem, ur_event_handle_t *phEvent) {
  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);

  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  UR_CALL(lockedBatch->getActiveBatch().appendUSMAllocHelper(
      this, pPool, size, pProperties, waitListView, ppMem,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatch->getCurrentGeneration()),
      UR_USM_TYPE_HOST));

  return queueFlushUnlocked(lockedBatch);
}

ur_result_t ur_queue_batched_t::bindlessImagesImageCopyExp(
    const void *pSrc, void *pDst, const ur_image_desc_t *pSrcImageDesc,
    const ur_image_desc_t *pDstImageDesc,
    const ur_image_format_t *pSrcImageFormat,
    const ur_image_format_t *pDstImageFormat,
    ur_exp_image_copy_region_t *pCopyRegion,
    ur_exp_image_copy_flags_t imageCopyFlags,
    ur_exp_image_copy_input_types_t imageCopyInputTypes,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {

  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);

  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  return lockedBatch->getActiveBatch().bindlessImagesImageCopyExp(
      pSrc, pDst, pSrcImageDesc, pDstImageDesc, pSrcImageFormat,
      pDstImageFormat, pCopyRegion, imageCopyFlags, imageCopyInputTypes,
      waitListView,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatch->getCurrentGeneration()));
}

ur_result_t ur_queue_batched_t::bindlessImagesWaitExternalSemaphoreExp(
    ur_exp_external_semaphore_handle_t hSemaphore, bool hasWaitValue,
    uint64_t waitValue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);

  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  return lockedBatch->getActiveBatch().bindlessImagesWaitExternalSemaphoreExp(
      hSemaphore, hasWaitValue, waitValue, waitListView,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatch->getCurrentGeneration()));
}

ur_result_t ur_queue_batched_t::bindlessImagesSignalExternalSemaphoreExp(
    ur_exp_external_semaphore_handle_t hSemaphore, bool hasSignalValue,
    uint64_t signalValue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);

  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  return lockedBatch->getActiveBatch().bindlessImagesSignalExternalSemaphoreExp(
      hSemaphore, hasSignalValue, signalValue, waitListView,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatch->getCurrentGeneration()));
}

// In case of queues with batched submissions, which use regular command lists
// (similarly to command buffers), the start timestamp would be recorded as the
// operation is submitted (event.recordStartTimestamp() in
// appendTimestampRecordingExp does not use the queue but directly the device),
// but the end timestamp would wait for the submission of the given regular
// command list. The difference between the start and end timestamps would
// reflect the delay in the batch submission, the difference between end
// timestamps would reflect the actual time of execution.
//
// TODO
// The version of timestampRecording for batched queues should be adjusted in
// order to reflect the idea behind the original function

ur_result_t ur_queue_batched_t::enqueueTimestampRecordingExp(
    bool /* blocking */, uint32_t /* numEventsInWaitList */,
    const ur_event_handle_t * /* phEventWaitList */,
    ur_event_handle_t * /* phEvent */) {

  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  // wait_list_view waitListView =
  //     wait_list_view(phEventWaitList, numEventsInWaitList, this);

  // auto lockedBatch = currentCmdLists.lock();

  // lockedBatch->markNextIssuedCommand();

  // UR_CALL(lockedBatch->getActiveBatch().appendTimestampRecordingExp(
  //     false, waitListView,
  //     createEventIfRequestedRegular(phEvent,
  //                                   lockedBatch->getCurrentGeneration())));

  // if (blocking) {
  //   UR_CALL(queueFinishUnlocked(lockedBatch));
  // }

  // return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_batched_t::enqueueCommandBufferExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);

  auto lockedBatch = currentCmdLists.lock();

  // Firstly, enqueue the current batch (a regular list), then enqueue the
  // command buffer batch (also a regular list) to preserve the order of
  // operations
  if (!lockedBatch->isActiveBatchEmpty()) {
    UR_CALL(queueFlushUnlocked(lockedBatch));
  }

  // Regular lists cannot be appended to other regular lists for execution, only
  // to immediate lists
  return lockedBatch->getImmediateManager().appendCommandBufferExp(
      hCommandBuffer, waitListView,
      createEventAndRetain(eventPoolImmediate.get(), phEvent, this));
}

ur_result_t ur_queue_batched_t::enqueueNativeCommandExp(
    ur_exp_enqueue_native_command_function_t pfnNativeEnqueue, void *data,
    uint32_t numMemsInMemList, const ur_mem_handle_t *phMemList,
    const ur_exp_enqueue_native_command_properties_t *pProperties,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);

  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  return lockedBatch->getActiveBatch().appendNativeCommandExp(
      pfnNativeEnqueue, data, numMemsInMemList, phMemList, pProperties,
      waitListView,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatch->getCurrentGeneration()));
}

ur_result_t ur_queue_batched_t::enqueueKernelLaunchWithArgsExp(
    ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numArgs,
    const ur_exp_kernel_arg_properties_t *pArgs,
    const ur_kernel_launch_ext_properties_t *launchPropList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {

  wait_list_view waitListView =
      wait_list_view(phEventWaitList, numEventsInWaitList, this);

  auto lockedBatch = currentCmdLists.lock();

  markIssuedCommandInBatch(lockedBatch);

  return lockedBatch->getActiveBatch().appendKernelLaunchWithArgsExp(
      hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize,
      numArgs, pArgs, launchPropList, waitListView,
      createEventIfRequestedRegular(phEvent,
                                    lockedBatch->getCurrentGeneration()));
}

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
    bool isBatchEmpty = currentCmdLists.get_no_lock()->isActiveBatchEmpty();
    if (isBatchEmpty) {
      auto status = ZE_CALL_NOCHECK(
          zeCommandListHostSynchronize,
          (currentCmdLists.get_no_lock()->getImmediateListHandle(), 0));
      if (status == ZE_RESULT_SUCCESS) {
        return ReturnValue(true);
      } else if (status == ZE_RESULT_NOT_READY) {
        return ReturnValue(false);
      } else {
        return ze2urResult(status);
      }
    } else {
      return ReturnValue(false);
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
      currentCmdLists.get_no_lock()->getImmediateListHandle());
  return UR_RESULT_SUCCESS;
}

ur_result_t
ur_queue_batched_t::queueFlushUnlocked(locked<batch_manager> &batchLocked) {
  UR_CALL(batchLocked->enqueueCurrentBatchUnlocked());

  return renewBatchUnlocked(batchLocked);
}

ur_result_t ur_queue_batched_t::queueFlush() {
  auto batchLocked = currentCmdLists.lock();

  if (batchLocked->isActiveBatchEmpty()) {
    return UR_RESULT_SUCCESS;
  } else {
    return queueFlushUnlocked(batchLocked);
  }
}

} // namespace v2
