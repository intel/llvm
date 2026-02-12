/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file msan_ddi.hpp
 *
 */

#include "ur_ddi.h"

namespace ur_sanitizer_layer {

namespace msan {

ur_result_t urUSMDeviceAlloc(ur_context_handle_t hContext,
                             ur_device_handle_t hDevice,
                             const ur_usm_desc_t *pUSMDesc,
                             ur_usm_pool_handle_t pool, size_t size,
                             void **ppMem);

ur_result_t urUSMHostAlloc(ur_context_handle_t hContext,
                           const ur_usm_desc_t *pUSMDesc,
                           ur_usm_pool_handle_t pool, size_t size,
                           void **ppMem);

ur_result_t urUSMFree(ur_context_handle_t hContext, void *pMem);

ur_result_t urEnqueueUSMMemcpy(ur_queue_handle_t hQueue, bool blocking,
                               void *pDst, const void *pSrc, size_t size,
                               uint32_t numEventsInWaitList,
                               const ur_event_handle_t *phEventWaitList,
                               ur_event_handle_t *phEvent);

ur_result_t urEnqueueUSMMemcpy2D(ur_queue_handle_t hQueue, bool blocking,
                                 void *pDst, size_t dstPitch, const void *pSrc,
                                 size_t srcPitch, size_t width, size_t height,
                                 uint32_t numEventsInWaitList,
                                 const ur_event_handle_t *phEventWaitList,
                                 ur_event_handle_t *phEvent);

ur_result_t urEnqueueUSMFill(ur_queue_handle_t hQueue, void *pMem,
                             size_t patternSize, const void *pPattern,
                             size_t size, uint32_t numEventsInWaitList,
                             const ur_event_handle_t *phEventWaitList,
                             ur_event_handle_t *phEvent);

ur_result_t urEnqueueUSMFill2D(ur_queue_handle_t hQueue, void *pMem,
                               size_t pitch, size_t patternSize,
                               const void *pPattern, size_t width,
                               size_t height, uint32_t numEventsInWaitList,
                               const ur_event_handle_t *phEventWaitList,
                               ur_event_handle_t *phEvent);

} // namespace msan

void initMsanInterceptor();
void destroyMsanInterceptor();

ur_result_t initMsanDDITable(ur_dditable_t *dditable);

} // namespace ur_sanitizer_layer
