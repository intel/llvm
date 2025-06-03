//===--------- queue_immediate_in_order.hpp - Level Zero Adapter ---------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "../common.hpp"
#include "../device.hpp"

#include "context.hpp"
#include "event.hpp"
#include "event_pool_cache.hpp"
#include "memory.hpp"
#include "queue_api.hpp"

#include "ur/ur.hpp"

#include "command_list_manager.hpp"
#include "lockable.hpp"

namespace v2 {

using queue_group_type = ur_device_handle_t_::queue_group_info_t::type;

struct ur_queue_immediate_in_order_t : ur_object, public ur_queue_t_ {
private:
  ur_context_handle_t hContext;
  ur_device_handle_t hDevice;
  ur_queue_flags_t flags;

  lockable<ur_command_list_manager> commandListManager;
  std::vector<ur_kernel_handle_t> submittedKernels;

  wait_list_view
  getWaitListView(locked<ur_command_list_manager> &commandList,
                  const ur_event_handle_t *phWaitEvents, uint32_t numWaitEvents,
                  ur_event_handle_t additionalWaitEvent = nullptr);

  ze_event_handle_t getSignalEvent(locked<ur_command_list_manager> &commandList,
                                   ur_event_handle_t *hUserEvent,
                                   ur_command_t commandType);

  ur_result_t enqueueGenericFillUnlocked(
      ur_mem_buffer_t *hBuffer, size_t offset, size_t patternSize,
      const void *pPattern, size_t size, uint32_t numEventsInWaitList,
      const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent,
      ur_command_t commandType);

  ur_result_t enqueueGenericCommandListsExp(
      uint32_t numCommandLists, ze_command_list_handle_t *phCommandLists,
      ur_event_handle_t *phEvent, uint32_t numEventsInWaitList,
      const ur_event_handle_t *phEventWaitList, ur_command_t callerCommand,
      ur_event_handle_t additionalWaitEvent);

  ur_result_t
  enqueueEventsWaitWithBarrierImpl(uint32_t numEventsInWaitList,
                                   const ur_event_handle_t *phEventWaitList,
                                   ur_event_handle_t *phEvent);

  void recordSubmittedKernel(ur_kernel_handle_t hKernel);

  inline ur_result_t enqueueCooperativeKernelLaunchHelper(
      ur_kernel_handle_t hKernel, uint32_t workDim,
      const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
      const size_t *pLocalWorkSize, uint32_t numEventsInWaitList,
      const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent);

  ur_result_t
  enqueueUSMAllocHelper(ur_usm_pool_handle_t pPool, const size_t size,
                        const ur_exp_async_usm_alloc_properties_t *pProperties,
                        uint32_t numEventsInWaitList,
                        const ur_event_handle_t *phEventWaitList, void **ppMem,
                        ur_event_handle_t *phEvent, ur_usm_type_t Type);

public:
  ur_queue_immediate_in_order_t(ur_context_handle_t, ur_device_handle_t,
                                const ur_queue_properties_t *);
  ur_queue_immediate_in_order_t(ur_context_handle_t, ur_device_handle_t,
                                ur_native_handle_t, ur_queue_flags_t,
                                bool ownZeQueue);

  ~ur_queue_immediate_in_order_t();

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
  ur_result_t enqueueEventsWait(uint32_t numEventsInWaitList,
                                const ur_event_handle_t *phEventWaitList,
                                ur_event_handle_t *phEvent) override;
  ur_result_t
  enqueueEventsWaitWithBarrier(uint32_t numEventsInWaitList,
                               const ur_event_handle_t *phEventWaitList,
                               ur_event_handle_t *phEvent) override;
  ur_result_t enqueueEventsWaitWithBarrierExt(
      const ur_exp_enqueue_ext_properties_t *pProperties,
      uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
      ur_event_handle_t *phEvent) override;
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
  ur_result_t enqueueUSMFill2D(void *, size_t, size_t, const void *, size_t,
                               size_t, uint32_t, const ur_event_handle_t *,
                               ur_event_handle_t *) override;
  ur_result_t enqueueUSMMemcpy2D(bool, void *, size_t, const void *, size_t,
                                 size_t, size_t, uint32_t,
                                 const ur_event_handle_t *,
                                 ur_event_handle_t *) override;
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
      ur_exp_image_copy_flags_t imageCopyFlags, uint32_t numEventsInWaitList,
      const ur_event_handle_t *phEventWaitList,
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
  ur_result_t
  enqueueNativeCommandExp(ur_exp_enqueue_native_command_function_t, void *,
                          uint32_t, const ur_mem_handle_t *,
                          const ur_exp_enqueue_native_command_properties_t *,
                          uint32_t, const ur_event_handle_t *,
                          ur_event_handle_t *) override;
};

} // namespace v2
