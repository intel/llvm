//===--------------- queue_batched.hpp - Level Zero Adapter ---------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "../common.hpp"
#include "../device.hpp"

#include "command_list_cache.hpp"
#include "common/ur_ref_count.hpp"
#include "context.hpp"
#include "event.hpp"
#include "event_pool_cache.hpp"
#include "memory.hpp"
#include "queue_api.hpp"

#include "ur/ur.hpp"

#include "command_buffer.hpp"
#include "command_list_manager.hpp"
#include "lockable.hpp"
#include "queue_immediate_in_order.hpp"
#include "ur_api.h"
#include "ze_api.h"

namespace v2 {

struct batch_manager {
private:
  ur_command_list_manager activeBatch;
  ur_command_list_manager immediateList;
  std::vector<v2::raii::command_list_unique_handle> runBatches;
  ur_event_generation_t regularGenerationNumber;
  static constexpr uint64_t initialSlotsForBatches = 10;
  bool isEmpty = true;

public:
  batch_manager(ur_context_handle_t context, ur_device_handle_t device,
                v2::raii::command_list_unique_handle &&commandListRegular,
                v2::raii::command_list_unique_handle &&commandListImmediate)
      : activeBatch(context, device,
                    std::forward<v2::raii::command_list_unique_handle>(
                        commandListRegular)),
        immediateList(context, device,
                      std::forward<v2::raii::command_list_unique_handle>(
                          commandListImmediate)),
        regularGenerationNumber(0) {
    runBatches.reserve(initialSlotsForBatches);
  }

  ur_result_t
  renewRegularUnlocked(v2::raii::command_list_unique_handle &&newRegularBatch);

  bool isCurrentGeneration(ur_event_generation_t batch_generation);

  // ur_result_t
  // runAndRenewBatch(v2::raii::command_list_unique_handle &&newRegularBatch);

  ur_command_list_manager &getActiveBatch() { return activeBatch; }

  ur_event_generation_t getCurrentGeneration() {
    return regularGenerationNumber;
  }

  ur_result_t batchFinish();

  ze_command_list_handle_t getImmediateListHandle() {
    return immediateList.getZeCommandList();
  }

  ze_command_list_handle_t getRegularListHandle() {
    return activeBatch.getZeCommandList();
  }

  bool isActiveBatchEmpty() { return isEmpty; }

  void markIssuedCommand() { isEmpty = false; }

  void setBatchEmpty() { isEmpty = true; }

  // void clearOldCommandLists();

  bool isLimitOfUsedCommandListsReached() {
    return initialSlotsForBatches <= runBatches.size();
  }
};

struct ur_queue_batched_t : ur_object, ur_queue_t_ {
private:
  ur_context_handle_t hContext;
  ur_device_handle_t hDevice;

  v2::command_list_desc_t regularCmdListDesc;
  lockable<batch_manager> currentCmdLists;

  ur_queue_flags_t flags;
  v2::raii::cache_borrowed_event_pool eventPoolRegular;

  v2::raii::command_list_unique_handle getNewRegularCmdList() {
    TRACK_SCOPE_LATENCY("ur_queue_batched_t::getNewRegularCmdList");

    return hContext->getCommandListCache().getRegularCommandList(
        hDevice->ZeDevice, regularCmdListDesc);
  }

  void runOldBatchRenewBatch();

  // ur_result_t renewRegular();
  ur_result_t renewBatchUnlocked(locked<batch_manager> &batchLocked);

  ur_event_handle_t
  createEventIfRequestedRegular(ur_event_handle_t *phEvent,
                                ur_event_generation_t generation_number);

  ur_event_handle_t
  createEventAndRetainRegular(ur_event_handle_t *phEvent,
                              ur_event_generation_t batch_generation);

  ur_result_t
  queueFinishBatchAndPoolsUnlocked(ze_command_list_handle_t immediateList,
                                   ze_command_list_handle_t regularList);

  ur_result_t queueFinishUnlocked(locked<batch_manager> &batchLocked);

  ur_result_t queueFlushUnlocked(locked<batch_manager> &batchLocked);

public:
  ur_queue_batched_t(ur_context_handle_t, ur_device_handle_t, uint32_t ordinal,
                     ze_command_queue_priority_t priority,
                     std::optional<int32_t> index, event_flags_t eventFlags,
                     ur_queue_flags_t flags);

  ur_result_t
  onEventWaitListUse(ur_event_generation_t batch_generation) override;

  ~ur_queue_batched_t();

  ur_result_t queueGetInfo(ur_queue_info_t propName, size_t propSize,
                           void *pPropValue, size_t *pPropSizeRet) override;
  ur_result_t queueGetNativeHandle(ur_queue_native_desc_t *pDesc,
                                   ur_native_handle_t *phNativeQueue) override;
  ur_result_t queueFinish() override;
  ur_result_t queueFlush() override;
  ur_result_t enqueueKernelLaunch(
      ur_kernel_handle_t hKernel, uint32_t workDim,
      const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
      const size_t *pLocalWorkSize, uint32_t numPropsInLaunchPropList,
      const ur_kernel_launch_property_t *launchPropList,
      uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
      ur_event_handle_t *phEvent) override;

  ur_result_t
  enqueueEventsWaitWithBarrier(uint32_t numEventsInWaitList,
                               const ur_event_handle_t *phEventWaitList,
                               ur_event_handle_t *phEvent) override;

  ur_result_t enqueueEventsWait(uint32_t numEventsInWaitList,
                                const ur_event_handle_t *phEventWaitList,
                                ur_event_handle_t *phEvent) override;

  ur_result_t
  enqueueEventsWaitWithBarrierExt(const ur_exp_enqueue_ext_properties_t *,
                                  uint32_t numEventsInWaitList,
                                  const ur_event_handle_t *phEventWaitList,
                                  ur_event_handle_t *phEvent) override {
    return enqueueEventsWaitWithBarrier(numEventsInWaitList, phEventWaitList,
                                        phEvent);
  }

  ur_result_t enqueueMemBufferRead(ur_mem_handle_t hBuffer, bool blockingRead,
                                   size_t offset, size_t size, void *pDst,
                                   uint32_t numEventsInWaitList,
                                   const ur_event_handle_t *phEventWaitList,
                                   ur_event_handle_t *phEvent) override;

  ur_result_t enqueueMemBufferWrite(ur_mem_handle_t hBuffer, bool blockingWrite,
                                    size_t offset, size_t size,
                                    const void *pSrc,
                                    uint32_t numEventsInWaitList,
                                    const ur_event_handle_t *phEventWaitList,
                                    ur_event_handle_t *phEvent) override;

  ur_result_t enqueueMemBufferReadRect(
      ur_mem_handle_t hBuffer, bool blockingRead, ur_rect_offset_t bufferOrigin,
      ur_rect_offset_t hostOrigin, ur_rect_region_t region,
      size_t bufferRowPitch, size_t bufferSlicePitch, size_t hostRowPitch,
      size_t hostSlicePitch, void *pDst, uint32_t numEventsInWaitList,
      const ur_event_handle_t *phEventWaitList,
      ur_event_handle_t *phEvent) override;

  ur_result_t enqueueMemBufferWriteRect(
      ur_mem_handle_t hBuffer, bool blockingWrite,
      ur_rect_offset_t bufferOrigin, ur_rect_offset_t hostOrigin,
      ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
      size_t hostRowPitch, size_t hostSlicePitch, void *pSrc,
      uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
      ur_event_handle_t *phEvent) override;

  ur_result_t enqueueMemBufferCopy(ur_mem_handle_t hBufferSrc,
                                   ur_mem_handle_t hBufferDst, size_t srcOffset,
                                   size_t dstOffset, size_t size,
                                   uint32_t numEventsInWaitList,
                                   const ur_event_handle_t *phEventWaitList,
                                   ur_event_handle_t *phEvent) override;

  ur_result_t enqueueMemBufferCopyRect(
      ur_mem_handle_t hBufferSrc, ur_mem_handle_t hBufferDst,
      ur_rect_offset_t srcOrigin, ur_rect_offset_t dstOrigin,
      ur_rect_region_t region, size_t srcRowPitch, size_t srcSlicePitch,
      size_t dstRowPitch, size_t dstSlicePitch, uint32_t numEventsInWaitList,
      const ur_event_handle_t *phEventWaitList,
      ur_event_handle_t *phEvent) override;

  ur_result_t enqueueMemBufferFill(ur_mem_handle_t hBuffer,
                                   const void *pPattern, size_t patternSize,
                                   size_t offset, size_t size,
                                   uint32_t numEventsInWaitList,
                                   const ur_event_handle_t *phEventWaitList,
                                   ur_event_handle_t *phEvent) override;

  ur_result_t enqueueMemImageRead(ur_mem_handle_t hImage, bool blockingRead,
                                  ur_rect_offset_t origin,
                                  ur_rect_region_t region, size_t rowPitch,
                                  size_t slicePitch, void *pDst,
                                  uint32_t numEventsInWaitList,
                                  const ur_event_handle_t *phEventWaitList,
                                  ur_event_handle_t *phEvent) override;

  ur_result_t enqueueMemImageWrite(ur_mem_handle_t hImage, bool blockingWrite,
                                   ur_rect_offset_t origin,
                                   ur_rect_region_t region, size_t rowPitch,
                                   size_t slicePitch, void *pSrc,
                                   uint32_t numEventsInWaitList,
                                   const ur_event_handle_t *phEventWaitList,
                                   ur_event_handle_t *phEvent) override;

  ur_result_t
  enqueueMemImageCopy(ur_mem_handle_t hImageSrc, ur_mem_handle_t hImageDst,
                      ur_rect_offset_t srcOrigin, ur_rect_offset_t dstOrigin,
                      ur_rect_region_t region, uint32_t numEventsInWaitList,
                      const ur_event_handle_t *phEventWaitList,
                      ur_event_handle_t *phEvent) override;

  ur_result_t enqueueMemBufferMap(ur_mem_handle_t hBuffer, bool blockingMap,
                                  ur_map_flags_t mapFlags, size_t offset,
                                  size_t size, uint32_t numEventsInWaitList,
                                  const ur_event_handle_t *phEventWaitList,
                                  ur_event_handle_t *phEvent,
                                  void **ppRetMap) override;

  ur_result_t enqueueMemUnmap(ur_mem_handle_t hMem, void *pMappedPtr,
                              uint32_t numEventsInWaitList,
                              const ur_event_handle_t *phEventWaitList,
                              ur_event_handle_t *phEvent) override;

  ur_result_t enqueueUSMFill(void *pMem, size_t patternSize,
                             const void *pPattern, size_t size,
                             uint32_t numEventsInWaitList,
                             const ur_event_handle_t *phEventWaitList,
                             ur_event_handle_t *phEvent) override;

  ur_result_t enqueueUSMMemcpy(bool blocking, void *pDst, const void *pSrc,
                               size_t size, uint32_t numEventsInWaitList,
                               const ur_event_handle_t *phEventWaitList,
                               ur_event_handle_t *phEvent) override;

  ur_result_t enqueueUSMFill2D(void *pMem, size_t pitch, size_t patternSize,
                               const void *pPattern, size_t width,
                               size_t height, uint32_t numEventsInWaitList,
                               const ur_event_handle_t *phEventWaitList,
                               ur_event_handle_t *phEvent) override;

  ur_result_t enqueueUSMMemcpy2D(bool blocking, void *pDst, size_t dstPitch,
                                 const void *pSrc, size_t srcPitch,
                                 size_t width, size_t height,
                                 uint32_t numEventsInWaitList,
                                 const ur_event_handle_t *phEventWaitList,
                                 ur_event_handle_t *phEvent) override;

  ur_result_t enqueueUSMPrefetch(const void *pMem, size_t size,
                                 ur_usm_migration_flags_t flags,
                                 uint32_t numEventsInWaitList,
                                 const ur_event_handle_t *phEventWaitList,
                                 ur_event_handle_t *phEvent) override;

  ur_result_t enqueueUSMAdvise(const void *pMem, size_t size,
                               ur_usm_advice_flags_t advice,
                               ur_event_handle_t *phEvent) override;

  ur_result_t enqueueDeviceGlobalVariableWrite(
      ur_program_handle_t hProgram, const char *name, bool blockingWrite,
      size_t count, size_t offset, const void *pSrc,
      uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
      ur_event_handle_t *phEvent) override;

  ur_result_t enqueueDeviceGlobalVariableRead(
      ur_program_handle_t hProgram, const char *name, bool blockingRead,
      size_t count, size_t offset, void *pDst, uint32_t numEventsInWaitList,
      const ur_event_handle_t *phEventWaitList,
      ur_event_handle_t *phEvent) override;

  ur_result_t enqueueReadHostPipe(ur_program_handle_t hProgram,
                                  const char *pipe_symbol, bool blocking,
                                  void *pDst, size_t size,
                                  uint32_t numEventsInWaitList,
                                  const ur_event_handle_t *phEventWaitList,
                                  ur_event_handle_t *phEvent) override;

  ur_result_t enqueueWriteHostPipe(ur_program_handle_t hProgram,
                                   const char *pipe_symbol, bool blocking,
                                   void *pSrc, size_t size,
                                   uint32_t numEventsInWaitList,
                                   const ur_event_handle_t *phEventWaitList,
                                   ur_event_handle_t *phEvent) override;

  ur_result_t enqueueUSMDeviceAllocExp(
      ur_usm_pool_handle_t pPool, const size_t size,
      const ur_exp_async_usm_alloc_properties_t *pProperties,
      uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
      void **ppMem, ur_event_handle_t *phEvent) override;

  ur_result_t enqueueUSMSharedAllocExp(
      ur_usm_pool_handle_t pPool, const size_t size,
      const ur_exp_async_usm_alloc_properties_t *pProperties,
      uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
      void **ppMem, ur_event_handle_t *phEvent) override;

  ur_result_t
  enqueueUSMHostAllocExp(ur_usm_pool_handle_t pPool, const size_t size,
                         const ur_exp_async_usm_alloc_properties_t *pProperties,
                         uint32_t numEventsInWaitList,
                         const ur_event_handle_t *phEventWaitList, void **ppMem,
                         ur_event_handle_t *phEvent) override;

  ur_result_t enqueueUSMFreeExp(ur_usm_pool_handle_t pPool, void *pMem,
                                uint32_t numEventsInWaitList,
                                const ur_event_handle_t *phEventWaitList,
                                ur_event_handle_t *phEvent) override;

  ur_result_t bindlessImagesImageCopyExp(
      const void *pSrc, void *pDst, const ur_image_desc_t *pSrcImageDesc,
      const ur_image_desc_t *pDstImageDesc,
      const ur_image_format_t *pSrcImageFormat,
      const ur_image_format_t *pDstImageFormat,
      ur_exp_image_copy_region_t *pCopyRegion,
      ur_exp_image_copy_flags_t imageCopyFlags,
      ur_exp_image_copy_input_types_t imageCopyInputTypes,
      uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
      ur_event_handle_t *phEvent) override;

  ur_result_t bindlessImagesWaitExternalSemaphoreExp(
      ur_exp_external_semaphore_handle_t hSemaphore, bool hasWaitValue,
      uint64_t waitValue, uint32_t numEventsInWaitList,
      const ur_event_handle_t *phEventWaitList,
      ur_event_handle_t *phEvent) override;

  ur_result_t bindlessImagesSignalExternalSemaphoreExp(
      ur_exp_external_semaphore_handle_t hSemaphore, bool hasSignalValue,
      uint64_t signalValue, uint32_t numEventsInWaitList,
      const ur_event_handle_t *phEventWaitList,
      ur_event_handle_t *phEvent) override;

  ur_result_t
  enqueueTimestampRecordingExp(bool blocking, uint32_t numEventsInWaitList,
                               const ur_event_handle_t *phEventWaitList,
                               ur_event_handle_t *phEvent) override;

  ur_result_t
  enqueueCommandBufferExp(ur_exp_command_buffer_handle_t hCommandBuffer,
                          uint32_t numEventsInWaitList,
                          const ur_event_handle_t *phEventWaitList,
                          ur_event_handle_t *phEvent) override;

  ur_result_t enqueueNativeCommandExp(
      ur_exp_enqueue_native_command_function_t pfnNativeEnqueue, void *data,
      uint32_t numMemsInMemList, const ur_mem_handle_t *phMemList,
      const ur_exp_enqueue_native_command_properties_t *pProperties,
      uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
      ur_event_handle_t *phEvent) override;

  ur::RefCount RefCount;
};

} // namespace v2
