/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file queue_api.hpp
 *
 */

#pragma once

#include <ur_api.h>

struct ur_queue_handle_t_ {
  virtual ~ur_queue_handle_t_();

  virtual void deferEventFree(ur_event_handle_t hEvent) = 0;

  virtual ur_result_t queueGetInfo(ur_queue_info_t, size_t, void *,
                                   size_t *) = 0;
  virtual ur_result_t queueRetain() = 0;
  virtual ur_result_t queueRelease() = 0;
  virtual ur_result_t queueGetNativeHandle(ur_queue_native_desc_t *,
                                           ur_native_handle_t *) = 0;
  virtual ur_result_t queueFinish() = 0;
  virtual ur_result_t queueFlush() = 0;
  virtual ur_result_t enqueueKernelLaunch(ur_kernel_handle_t, uint32_t,
                                          const size_t *, const size_t *,
                                          const size_t *, uint32_t,
                                          const ur_event_handle_t *,
                                          ur_event_handle_t *) = 0;
  virtual ur_result_t enqueueEventsWait(uint32_t, const ur_event_handle_t *,
                                        ur_event_handle_t *) = 0;
  virtual ur_result_t enqueueEventsWaitWithBarrier(uint32_t,
                                                   const ur_event_handle_t *,
                                                   ur_event_handle_t *) = 0;
  virtual ur_result_t enqueueMemBufferRead(ur_mem_handle_t, bool, size_t,
                                           size_t, void *, uint32_t,
                                           const ur_event_handle_t *,
                                           ur_event_handle_t *) = 0;
  virtual ur_result_t enqueueMemBufferWrite(ur_mem_handle_t, bool, size_t,
                                            size_t, const void *, uint32_t,
                                            const ur_event_handle_t *,
                                            ur_event_handle_t *) = 0;
  virtual ur_result_t
  enqueueMemBufferReadRect(ur_mem_handle_t, bool, ur_rect_offset_t,
                           ur_rect_offset_t, ur_rect_region_t, size_t, size_t,
                           size_t, size_t, void *, uint32_t,
                           const ur_event_handle_t *, ur_event_handle_t *) = 0;
  virtual ur_result_t
  enqueueMemBufferWriteRect(ur_mem_handle_t, bool, ur_rect_offset_t,
                            ur_rect_offset_t, ur_rect_region_t, size_t, size_t,
                            size_t, size_t, void *, uint32_t,
                            const ur_event_handle_t *, ur_event_handle_t *) = 0;
  virtual ur_result_t enqueueMemBufferCopy(ur_mem_handle_t, ur_mem_handle_t,
                                           size_t, size_t, size_t, uint32_t,
                                           const ur_event_handle_t *,
                                           ur_event_handle_t *) = 0;
  virtual ur_result_t
  enqueueMemBufferCopyRect(ur_mem_handle_t, ur_mem_handle_t, ur_rect_offset_t,
                           ur_rect_offset_t, ur_rect_region_t, size_t, size_t,
                           size_t, size_t, uint32_t, const ur_event_handle_t *,
                           ur_event_handle_t *) = 0;
  virtual ur_result_t enqueueMemBufferFill(ur_mem_handle_t, const void *,
                                           size_t, size_t, size_t, uint32_t,
                                           const ur_event_handle_t *,
                                           ur_event_handle_t *) = 0;
  virtual ur_result_t enqueueMemImageRead(ur_mem_handle_t, bool,
                                          ur_rect_offset_t, ur_rect_region_t,
                                          size_t, size_t, void *, uint32_t,
                                          const ur_event_handle_t *,
                                          ur_event_handle_t *) = 0;
  virtual ur_result_t enqueueMemImageWrite(ur_mem_handle_t, bool,
                                           ur_rect_offset_t, ur_rect_region_t,
                                           size_t, size_t, void *, uint32_t,
                                           const ur_event_handle_t *,
                                           ur_event_handle_t *) = 0;
  virtual ur_result_t enqueueMemImageCopy(ur_mem_handle_t, ur_mem_handle_t,
                                          ur_rect_offset_t, ur_rect_offset_t,
                                          ur_rect_region_t, uint32_t,
                                          const ur_event_handle_t *,
                                          ur_event_handle_t *) = 0;
  virtual ur_result_t enqueueMemBufferMap(ur_mem_handle_t, bool, ur_map_flags_t,
                                          size_t, size_t, uint32_t,
                                          const ur_event_handle_t *,
                                          ur_event_handle_t *, void **) = 0;
  virtual ur_result_t enqueueMemUnmap(ur_mem_handle_t, void *, uint32_t,
                                      const ur_event_handle_t *,
                                      ur_event_handle_t *) = 0;
  virtual ur_result_t enqueueUSMFill(void *, size_t, const void *, size_t,
                                     uint32_t, const ur_event_handle_t *,
                                     ur_event_handle_t *) = 0;
  virtual ur_result_t enqueueUSMMemcpy(bool, void *, const void *, size_t,
                                       uint32_t, const ur_event_handle_t *,
                                       ur_event_handle_t *) = 0;
  virtual ur_result_t enqueueUSMPrefetch(const void *, size_t,
                                         ur_usm_migration_flags_t, uint32_t,
                                         const ur_event_handle_t *,
                                         ur_event_handle_t *) = 0;
  virtual ur_result_t enqueueUSMAdvise(const void *, size_t,
                                       ur_usm_advice_flags_t,
                                       ur_event_handle_t *) = 0;
  virtual ur_result_t enqueueUSMFill2D(void *, size_t, size_t, const void *,
                                       size_t, size_t, uint32_t,
                                       const ur_event_handle_t *,
                                       ur_event_handle_t *) = 0;
  virtual ur_result_t enqueueUSMMemcpy2D(bool, void *, size_t, const void *,
                                         size_t, size_t, size_t, uint32_t,
                                         const ur_event_handle_t *,
                                         ur_event_handle_t *) = 0;
  virtual ur_result_t enqueueDeviceGlobalVariableWrite(
      ur_program_handle_t, const char *, bool, size_t, size_t, const void *,
      uint32_t, const ur_event_handle_t *, ur_event_handle_t *) = 0;
  virtual ur_result_t enqueueDeviceGlobalVariableRead(
      ur_program_handle_t, const char *, bool, size_t, size_t, void *, uint32_t,
      const ur_event_handle_t *, ur_event_handle_t *) = 0;
  virtual ur_result_t enqueueReadHostPipe(ur_program_handle_t, const char *,
                                          bool, void *, size_t, uint32_t,
                                          const ur_event_handle_t *,
                                          ur_event_handle_t *) = 0;
  virtual ur_result_t enqueueWriteHostPipe(ur_program_handle_t, const char *,
                                           bool, void *, size_t, uint32_t,
                                           const ur_event_handle_t *,
                                           ur_event_handle_t *) = 0;
  virtual ur_result_t bindlessImagesImageCopyExp(
      const void *, void *, const ur_image_desc_t *, const ur_image_desc_t *,
      const ur_image_format_t *, const ur_image_format_t *,
      ur_exp_image_copy_region_t *, ur_exp_image_copy_flags_t, uint32_t,
      const ur_event_handle_t *, ur_event_handle_t *) = 0;
  virtual ur_result_t bindlessImagesWaitExternalSemaphoreExp(
      ur_exp_external_semaphore_handle_t, bool, uint64_t, uint32_t,
      const ur_event_handle_t *, ur_event_handle_t *) = 0;
  virtual ur_result_t bindlessImagesSignalExternalSemaphoreExp(
      ur_exp_external_semaphore_handle_t, bool, uint64_t, uint32_t,
      const ur_event_handle_t *, ur_event_handle_t *) = 0;
  virtual ur_result_t enqueueCooperativeKernelLaunchExp(
      ur_kernel_handle_t, uint32_t, const size_t *, const size_t *,
      const size_t *, uint32_t, const ur_event_handle_t *,
      ur_event_handle_t *) = 0;
  virtual ur_result_t enqueueTimestampRecordingExp(bool, uint32_t,
                                                   const ur_event_handle_t *,
                                                   ur_event_handle_t *) = 0;
  virtual ur_result_t enqueueKernelLaunchCustomExp(
      ur_kernel_handle_t, uint32_t, const size_t *, const size_t *,
      const size_t *, uint32_t, const ur_exp_launch_property_t *, uint32_t,
      const ur_event_handle_t *, ur_event_handle_t *) = 0;
  virtual ur_result_t
  enqueueEventsWaitWithBarrierExt(const ur_exp_enqueue_ext_properties_t *,
                                  uint32_t, const ur_event_handle_t *,
                                  ur_event_handle_t *) = 0;
  virtual ur_result_t
  enqueueNativeCommandExp(ur_exp_enqueue_native_command_function_t, void *,
                          uint32_t, const ur_mem_handle_t *,
                          const ur_exp_enqueue_native_command_properties_t *,
                          uint32_t, const ur_event_handle_t *,
                          ur_event_handle_t *) = 0;
};
