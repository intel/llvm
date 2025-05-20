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
#include "event_pool_cache.hpp"
#include "memory.hpp"
#include "queue_api.hpp"
#include <ze_api.h>

struct wait_list_view {
  ze_event_handle_t *handles;
  uint32_t num;

  wait_list_view(ze_event_handle_t *handles, uint32_t num)
      : handles(num > 0 ? handles : nullptr), num(num) {}

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
                          v2::raii::command_list_unique_handle &&commandList,
                          v2::event_flags_t flags, ur_queue_t_ *queue,
                          bool isImmediate);
  ur_command_list_manager(const ur_command_list_manager &src) = delete;
  ur_command_list_manager(ur_command_list_manager &&src) = default;

  ur_command_list_manager &
  operator=(const ur_command_list_manager &src) = delete;
  ur_command_list_manager &operator=(ur_command_list_manager &&src) = default;

  ~ur_command_list_manager();

  ur_result_t appendKernelLaunch(ur_kernel_handle_t hKernel, uint32_t workDim,
                                 const size_t *pGlobalWorkOffset,
                                 const size_t *pGlobalWorkSize,
                                 const size_t *pLocalWorkSize,
                                 uint32_t numEventsInWaitList,
                                 const ur_event_handle_t *phEventWaitList,
                                 ur_event_handle_t *phEvent);

  ur_result_t appendUSMMemcpy(bool blocking, void *pDst, const void *pSrc,
                              size_t size, uint32_t numEventsInWaitList,
                              const ur_event_handle_t *phEventWaitList,
                              ur_event_handle_t *phEvent);
  ur_result_t appendMemBufferRead(ur_mem_handle_t hBuffer, bool blockingRead,
                                  size_t offset, size_t size, void *pDst,
                                  uint32_t numEventsInWaitList,
                                  const ur_event_handle_t *phEventWaitList,
                                  ur_event_handle_t *phEvent);

  ur_result_t appendMemBufferWrite(ur_mem_handle_t hBuffer, bool blockingWrite,
                                   size_t offset, size_t size, const void *pSrc,
                                   uint32_t numEventsInWaitList,
                                   const ur_event_handle_t *phEventWaitList,
                                   ur_event_handle_t *phEvent);

  ur_result_t appendMemBufferCopy(ur_mem_handle_t hBufferSrc,
                                  ur_mem_handle_t hBufferDst, size_t srcOffset,
                                  size_t dstOffset, size_t size,
                                  uint32_t numEventsInWaitList,
                                  const ur_event_handle_t *phEventWaitList,
                                  ur_event_handle_t *phEvent);

  ur_result_t appendMemBufferReadRect(
      ur_mem_handle_t hBuffer, bool blockingRead, ur_rect_offset_t bufferOrigin,
      ur_rect_offset_t hostOrigin, ur_rect_region_t region,
      size_t bufferRowPitch, size_t bufferSlicePitch, size_t hostRowPitch,
      size_t hostSlicePitch, void *pDst, uint32_t numEventsInWaitList,
      const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent);

  ur_result_t appendMemBufferWriteRect(
      ur_mem_handle_t hBuffer, bool blockingWrite,
      ur_rect_offset_t bufferOrigin, ur_rect_offset_t hostOrigin,
      ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
      size_t hostRowPitch, size_t hostSlicePitch, void *pSrc,
      uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
      ur_event_handle_t *phEvent);

  ur_result_t appendMemBufferCopyRect(
      ur_mem_handle_t hBufferSrc, ur_mem_handle_t hBufferDst,
      ur_rect_offset_t srcOrigin, ur_rect_offset_t dstOrigin,
      ur_rect_region_t region, size_t srcRowPitch, size_t srcSlicePitch,
      size_t dstRowPitch, size_t dstSlicePitch, uint32_t numEventsInWaitList,
      const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent);

  ur_result_t appendUSMMemcpy2D(bool blocking, void *pDst, size_t dstPitch,
                                const void *pSrc, size_t srcPitch, size_t width,
                                size_t height, uint32_t numEventsInWaitList,
                                const ur_event_handle_t *phEventWaitList,
                                ur_event_handle_t *phEvent);

  ur_result_t appendMemBufferFill(ur_mem_handle_t hBuffer, const void *pPattern,
                                  size_t patternSize, size_t offset,
                                  size_t size, uint32_t numEventsInWaitList,
                                  const ur_event_handle_t *phEventWaitList,
                                  ur_event_handle_t *phEvent);

  ur_result_t appendUSMFill(void *pMem, size_t patternSize,
                            const void *pPattern, size_t size,
                            uint32_t numEventsInWaitList,
                            const ur_event_handle_t *phEventWaitList,
                            ur_event_handle_t *phEvent);

  ur_result_t appendUSMPrefetch(const void *pMem, size_t size,
                                ur_usm_migration_flags_t flags,
                                uint32_t numEventsInWaitList,
                                const ur_event_handle_t *phEventWaitList,
                                ur_event_handle_t *phEvent);

  ur_result_t appendUSMAdvise(const void *pMem, size_t size,
                              ur_usm_advice_flags_t advice,
                              uint32_t numEventsInWaitList,
                              const ur_event_handle_t *phEventWaitList,
                              ur_event_handle_t *phEvent);

  ur_result_t appendBarrier(uint32_t numEventsInWaitList,
                            const ur_event_handle_t *phEventWaitList,
                            ur_event_handle_t *phEvent);

  ze_command_list_handle_t getZeCommandList();

  wait_list_view
  getWaitListView(const ur_event_handle_t *phWaitEvents, uint32_t numWaitEvents,
                  ur_event_handle_t additionalWaitEvent = nullptr);
  ze_event_handle_t getSignalEvent(ur_event_handle_t *hUserEvent,
                                   ur_command_t commandType);

private:
  ur_result_t appendGenericFillUnlocked(
      ur_mem_buffer_t *hBuffer, size_t offset, size_t patternSize,
      const void *pPattern, size_t size, uint32_t numEventsInWaitList,
      const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent,
      ur_command_t commandType);

  ur_result_t appendGenericCopyUnlocked(
      ur_mem_buffer_t *src, ur_mem_buffer_t *dst, bool blocking,
      size_t srcOffset, size_t dstOffset, size_t size,
      uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
      ur_event_handle_t *phEvent, ur_command_t commandType);

  ur_result_t appendRegionCopyUnlocked(
      ur_mem_buffer_t *src, ur_mem_buffer_t *dst, bool blocking,
      ur_rect_offset_t srcOrigin, ur_rect_offset_t dstOrigin,
      ur_rect_region_t region, size_t srcRowPitch, size_t srcSlicePitch,
      size_t dstRowPitch, size_t dstSlicePitch, uint32_t numEventsInWaitList,
      const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent,
      ur_command_t commandType);
  // UR context associated with this command-buffer
  ur_context_handle_t context;
  // Device associated with this command-buffer
  ur_device_handle_t device;
  v2::raii::cache_borrowed_event_pool eventPool;
  v2::raii::command_list_unique_handle zeCommandList;
  ur_queue_t_ *queue;
  std::vector<ze_event_handle_t> waitList;
};
