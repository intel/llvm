//===--------- command_list_manager.hpp - Level Zero Adapter --------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "command_list_cache.hpp"
#include "common.hpp"
#include "context.hpp"
#include "event_pool_cache.hpp"
#include "queue_api.hpp"
#include "ur_api.h"
#include <ze_api.h>

struct ur_mem_buffer_t;

struct wait_list_view {
  ze_event_handle_t *handles;
  uint32_t num;
  uint32_t max_size;

  wait_list_view(const ur_event_handle_t *phWaitEvents, uint32_t numWaitEvents);
  wait_list_view(const ur_event_handle_t *phWaitEvents, uint32_t numWaitEvents,
                 ur_queue_t_ *currentBatchedQueue);

  void init(uint32_t numWaitEvents);

  void setHandles(const ur_event_handle_t *phWaitEvents);

  void addEvent(ur_event_handle_t Event);

  operator bool() const {
    assert((handles != nullptr) == (num > 0));
    return num > 0;
  }

  void clear() {
    handles = nullptr;
    num = 0;
  }
};

struct ur_command_list_manager {
  ur_command_list_manager(ur_context_handle_t context,
                          ur_device_handle_t device,
                          v2::raii::command_list_unique_handle &&commandList);
  ur_command_list_manager(const ur_command_list_manager &src) = delete;
  ur_command_list_manager(ur_command_list_manager &&src) = default;

  ur_command_list_manager &
  operator=(const ur_command_list_manager &src) = delete;
  ur_command_list_manager &operator=(ur_command_list_manager &&src) = default;

  ~ur_command_list_manager() = default;

  ze_command_list_handle_t getZeCommandList();

  ur_result_t releaseSubmittedKernels();

  /************ Generic queue methods *************/
  ur_result_t appendEventsWait(wait_list_view &waitListView,
                               ur_event_handle_t phEvent);
  ur_result_t appendEventsWaitWithBarrier(wait_list_view &waitList,
                                          ur_event_handle_t phEvent);
  ur_result_t appendMemBufferRead(ur_mem_handle_t hBuffer, bool blockingRead,
                                  size_t offset, size_t size, void *pDst,
                                  wait_list_view &waitListView,
                                  ur_event_handle_t phEvent);
  ur_result_t appendMemBufferWrite(ur_mem_handle_t hBuffer, bool blockingWrite,
                                   size_t offset, size_t size, const void *pSrc,
                                   wait_list_view &waitListView,
                                   ur_event_handle_t phEvent);
  ur_result_t appendMemBufferReadRect(
      ur_mem_handle_t hBuffer, bool blockingRead, ur_rect_offset_t bufferOrigin,
      ur_rect_offset_t hostOrigin, ur_rect_region_t region,
      size_t bufferRowPitch, size_t bufferSlicePitch, size_t hostRowPitch,
      size_t hostSlicePitch, void *pDst, wait_list_view &waitListView,
      ur_event_handle_t phEvent);
  ur_result_t appendMemBufferWriteRect(
      ur_mem_handle_t hBuffer, bool blockingWrite,
      ur_rect_offset_t bufferOrigin, ur_rect_offset_t hostOrigin,
      ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
      size_t hostRowPitch, size_t hostSlicePitch, void *pSrc,
      wait_list_view &waitListView, ur_event_handle_t phEvent);
  ur_result_t appendMemBufferCopy(ur_mem_handle_t hBufferSrc,
                                  ur_mem_handle_t hBufferDst, size_t srcOffset,
                                  size_t dstOffset, size_t size,
                                  wait_list_view &waitListView,
                                  ur_event_handle_t phEvent);
  ur_result_t appendMemBufferCopyRect(
      ur_mem_handle_t hBufferSrc, ur_mem_handle_t hBufferDst,
      ur_rect_offset_t srcOrigin, ur_rect_offset_t dstOrigin,
      ur_rect_region_t region, size_t srcRowPitch, size_t srcSlicePitch,
      size_t dstRowPitch, size_t dstSlicePitch, wait_list_view &waitListView,
      ur_event_handle_t phEvent);
  ur_result_t appendMemBufferFill(ur_mem_handle_t hBuffer, const void *pPattern,
                                  size_t patternSize, size_t offset,
                                  size_t size, wait_list_view &waitListView,
                                  ur_event_handle_t phEvent);
  ur_result_t appendMemImageRead(ur_mem_handle_t hImage, bool blockingRead,
                                 ur_rect_offset_t origin,
                                 ur_rect_region_t region, size_t rowPitch,
                                 size_t slicePitch, void *pDst,
                                 wait_list_view &waitListView,
                                 ur_event_handle_t phEvent);
  ur_result_t appendMemImageWrite(ur_mem_handle_t hImage, bool blockingWrite,
                                  ur_rect_offset_t origin,
                                  ur_rect_region_t region, size_t rowPitch,
                                  size_t slicePitch, void *pSrc,
                                  wait_list_view &waitListView,
                                  ur_event_handle_t phEvent);
  ur_result_t
  appendMemImageCopy(ur_mem_handle_t hImageSrc, ur_mem_handle_t hImageDst,
                     ur_rect_offset_t srcOrigin, ur_rect_offset_t dstOrigin,
                     ur_rect_region_t region, wait_list_view &waitListView,
                     ur_event_handle_t phEvent);
  ur_result_t appendMemBufferMap(ur_mem_handle_t hBuffer, bool blockingMap,
                                 ur_map_flags_t mapFlags, size_t offset,
                                 size_t size, wait_list_view &waitListView,
                                 ur_event_handle_t phEvent, void **ppRetMap);
  ur_result_t appendMemUnmap(ur_mem_handle_t hMem, void *pMappedPtr,
                             wait_list_view &waitListView,
                             ur_event_handle_t phEvent);
  ur_result_t appendUSMFill(void *pMem, size_t patternSize,
                            const void *pPattern, size_t size,
                            wait_list_view &waitListView,
                            ur_event_handle_t phEvent);
  ur_result_t appendUSMMemcpy(bool blocking, void *pDst, const void *pSrc,
                              size_t size, wait_list_view &waitListView,
                              ur_event_handle_t phEvent);
  ur_result_t appendUSMFill2D(void *, size_t, size_t, const void *, size_t,
                              size_t, wait_list_view &, ur_event_handle_t);
  ur_result_t appendUSMMemcpy2D(bool, void *, size_t, const void *, size_t,
                                size_t, size_t, wait_list_view &,
                                ur_event_handle_t);
  ur_result_t appendUSMPrefetch(const void *pMem, size_t size,
                                ur_usm_migration_flags_t flags,
                                wait_list_view &waitListView,
                                ur_event_handle_t phEvent);
  ur_result_t appendUSMAdvise(const void *pMem, size_t size,
                              ur_usm_advice_flags_t advice,
                              wait_list_view &waitListView,
                              ur_event_handle_t phEvent);
  ur_result_t appendDeviceGlobalVariableWrite(ur_program_handle_t hProgram,
                                              const char *name,
                                              bool blockingWrite, size_t count,
                                              size_t offset, const void *pSrc,
                                              wait_list_view &waitListView,
                                              ur_event_handle_t phEvent);
  ur_result_t appendDeviceGlobalVariableRead(ur_program_handle_t hProgram,
                                             const char *name,
                                             bool blockingRead, size_t count,
                                             size_t offset, void *pDst,
                                             wait_list_view &waitListView,
                                             ur_event_handle_t phEvent);
  ur_result_t appendReadHostPipe(ur_program_handle_t hProgram,
                                 const char *pipe_symbol, bool blocking,
                                 void *pDst, size_t size,
                                 wait_list_view &waitListView,
                                 ur_event_handle_t phEvent);
  ur_result_t appendWriteHostPipe(ur_program_handle_t hProgram,
                                  const char *pipe_symbol, bool blocking,
                                  void *pSrc, size_t size,
                                  wait_list_view &waitListView,
                                  ur_event_handle_t phEvent);
  ur_result_t bindlessImagesImageCopyExp(
      const void *pSrc, void *pDst, const ur_image_desc_t *pSrcImageDesc,
      const ur_image_desc_t *pDstImageDesc,
      const ur_image_format_t *pSrcImageFormat,
      const ur_image_format_t *pDstImageFormat,
      ur_exp_image_copy_region_t *pCopyRegion,
      ur_exp_image_copy_flags_t imageCopyFlags,
      ur_exp_image_copy_input_types_t imageCopyInputTypes,
      wait_list_view &waitListView, ur_event_handle_t phEvent);
  ur_result_t bindlessImagesWaitExternalSemaphoreExp(
      ur_exp_external_semaphore_handle_t hSemaphore, bool hasWaitValue,
      uint64_t waitValue, wait_list_view &waitListView,
      ur_event_handle_t phEvent);
  ur_result_t bindlessImagesSignalExternalSemaphoreExp(
      ur_exp_external_semaphore_handle_t hSemaphore, bool hasSignalValue,
      uint64_t signalValue, wait_list_view &waitListView,
      ur_event_handle_t phEvent);
  ur_result_t appendCooperativeKernelLaunchExp(
      ur_kernel_handle_t hKernel, uint32_t workDim,
      const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
      const size_t *pLocalWorkSize, uint32_t numEventsInWaitList,
      const ur_event_handle_t *phEventWaitList, ur_event_handle_t phEvent);
  ur_result_t appendTimestampRecordingExp(bool blocking,
                                          wait_list_view &waitListView,
                                          ur_event_handle_t phEvent);
  ur_result_t
  appendCommandBufferExp(ur_exp_command_buffer_handle_t hCommandBuffer,
                         wait_list_view &waitListView,
                         ur_event_handle_t phEvent);
  ur_result_t
  appendKernelLaunch(ur_kernel_handle_t hKernel, uint32_t workDim,
                     const size_t *pGlobalWorkOffset,
                     const size_t *pGlobalWorkSize,
                     const size_t *pLocalWorkSize,
                     const ur_kernel_launch_ext_properties_t *launchPropList,
                     wait_list_view &waitListView, ur_event_handle_t phEvent);
  ur_result_t
  appendNativeCommandExp(ur_exp_enqueue_native_command_function_t, void *,
                         uint32_t, const ur_mem_handle_t *,
                         const ur_exp_enqueue_native_command_properties_t *,
                         wait_list_view &, ur_event_handle_t);

  ur_result_t appendUSMAllocHelper(
      ur_queue_t_ *Queue, ur_usm_pool_handle_t pPool, const size_t size,
      const ur_exp_async_usm_alloc_properties_t *, wait_list_view &waitListView,
      void **ppMem, ur_event_handle_t phEvent, ur_usm_type_t type);

  ur_result_t appendUSMFreeExp(ur_queue_t_ *Queue, ur_usm_pool_handle_t,
                               void *pMem, wait_list_view &waitListView,
                               ur_event_handle_t phEvent);

  ur_result_t appendKernelLaunchWithArgsExp(
      ur_kernel_handle_t hKernel, uint32_t workDim,
      const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
      const size_t *pLocalWorkSize, uint32_t numArgs,
      const ur_exp_kernel_arg_properties_t *pArgs,
      const ur_kernel_launch_ext_properties_t *launchPropList,
      wait_list_view &waitListView, ur_event_handle_t phEvent);

  v2::raii::command_list_unique_handle &&releaseCommandList();

  void replaceCommandList(v2::raii::command_list_unique_handle &&cmdlist);

private:
  ur_result_t appendKernelLaunchWithArgsExpOld(
      ur_kernel_handle_t hKernel, uint32_t workDim,
      const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
      const size_t *pLocalWorkSize, uint32_t numArgs,
      const ur_exp_kernel_arg_properties_t *pArgs,
      const ur_kernel_launch_ext_properties_t *launchPropList,
      wait_list_view &waitListView, ur_event_handle_t phEvent);

  ur_result_t appendKernelLaunchWithArgsExpNew(
      ur_kernel_handle_t hKernel, uint32_t workDim,
      const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
      const size_t *pLocalWorkSize, uint32_t numArgs,
      const ur_exp_kernel_arg_properties_t *pArgs,
      const ur_kernel_launch_ext_properties_t *launchPropList,
      wait_list_view &waitListView, ur_event_handle_t phEvent);

  ur_result_t appendGenericCommandListsExp(
      uint32_t numCommandLists, ze_command_list_handle_t *phCommandLists,
      ur_event_handle_t phEvent, wait_list_view &waitListView,
      ur_command_t callerCommand);

  void recordSubmittedKernel(ur_kernel_handle_t hKernel);

  ze_event_handle_t getSignalEvent(ur_event_handle_t hUserEvent,
                                   ur_command_t commandType);

  ur_result_t appendKernelLaunchLocked(
      ur_kernel_handle_t hKernel, ze_kernel_handle_t hZeKernel,
      uint32_t workDim, const size_t *pGlobalWorkOffset,
      const size_t *pGlobalWorkSize, const size_t *pLocalWorkSize,
      wait_list_view &waitListView, ur_event_handle_t phEvent, bool cooperative,
      bool callWithArgs = false, void *pNext = nullptr);

  ur_result_t appendKernelLaunchUnlocked(
      ur_kernel_handle_t hKernel, uint32_t workDim,
      const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
      const size_t *pLocalWorkSize, wait_list_view &waitListView,
      ur_event_handle_t phEvent, bool cooperative);

  ur_result_t appendGenericFillUnlocked(ur_mem_buffer_t *hBuffer, size_t offset,
                                        size_t patternSize,
                                        const void *pPattern, size_t size,
                                        wait_list_view &waitListView,
                                        ur_event_handle_t phEvent,
                                        ur_command_t commandType);

  ur_result_t appendGenericCopyUnlocked(ur_mem_buffer_t *src,
                                        ur_mem_buffer_t *dst, bool blocking,
                                        size_t srcOffset, size_t dstOffset,
                                        size_t size,
                                        wait_list_view &waitListView,
                                        ur_event_handle_t phEvent,
                                        ur_command_t commandType);

  ur_result_t appendRegionCopyUnlocked(
      ur_mem_buffer_t *src, ur_mem_buffer_t *dst, bool blocking,
      ur_rect_offset_t srcOrigin, ur_rect_offset_t dstOrigin,
      ur_rect_region_t region, size_t srcRowPitch, size_t srcSlicePitch,
      size_t dstRowPitch, size_t dstSlicePitch, wait_list_view &waitListView,
      ur_event_handle_t phEvent, ur_command_t commandType);

  // Context needs to be a first member - it needs to be alive
  // until all other members are destroyed.
  v2::raii::ur_context_handle_t hContext;
  v2::raii::ur_device_handle_t hDevice;

  std::vector<ur_kernel_handle_t> submittedKernels;
  v2::raii::command_list_unique_handle zeCommandList;
  std::vector<ze_event_handle_t> waitList;
};
