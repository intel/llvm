/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file queue_api.cpp
 *
 */

#include "queue_api.hpp"
#include "ur_util.hpp"

ur_queue_handle_t_::~ur_queue_handle_t_() {}

namespace ur::level_zero {
ur_result_t urQueueGetInfo(ur_queue_handle_t hQueue, ur_queue_info_t propName,
                           size_t propSize, void *pPropValue,
                           size_t *pPropSizeRet) try {
  return hQueue->queueGetInfo(propName, propSize, pPropValue, pPropSizeRet);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urQueueRetain(ur_queue_handle_t hQueue) try {
  return hQueue->queueRetain();
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urQueueRelease(ur_queue_handle_t hQueue) try {
  return hQueue->queueRelease();
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urQueueGetNativeHandle(ur_queue_handle_t hQueue,
                                   ur_queue_native_desc_t *pDesc,
                                   ur_native_handle_t *phNativeQueue) try {
  return hQueue->queueGetNativeHandle(pDesc, phNativeQueue);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urQueueFinish(ur_queue_handle_t hQueue) try {
  return hQueue->queueFinish();
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urQueueFlush(ur_queue_handle_t hQueue) try {
  return hQueue->queueFlush();
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urEnqueueKernelLaunch(
    ur_queue_handle_t hQueue, ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) try {
  return hQueue->enqueueKernelLaunch(
      hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize,
      numEventsInWaitList, phEventWaitList, phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urEnqueueEventsWait(ur_queue_handle_t hQueue,
                                uint32_t numEventsInWaitList,
                                const ur_event_handle_t *phEventWaitList,
                                ur_event_handle_t *phEvent) try {
  return hQueue->enqueueEventsWait(numEventsInWaitList, phEventWaitList,
                                   phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urEnqueueEventsWaitWithBarrier(
    ur_queue_handle_t hQueue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) try {
  return hQueue->enqueueEventsWaitWithBarrier(numEventsInWaitList,
                                              phEventWaitList, phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urEnqueueMemBufferRead(ur_queue_handle_t hQueue,
                                   ur_mem_handle_t hBuffer, bool blockingRead,
                                   size_t offset, size_t size, void *pDst,
                                   uint32_t numEventsInWaitList,
                                   const ur_event_handle_t *phEventWaitList,
                                   ur_event_handle_t *phEvent) try {
  return hQueue->enqueueMemBufferRead(hBuffer, blockingRead, offset, size, pDst,
                                      numEventsInWaitList, phEventWaitList,
                                      phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urEnqueueMemBufferWrite(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingWrite,
    size_t offset, size_t size, const void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) try {
  return hQueue->enqueueMemBufferWrite(hBuffer, blockingWrite, offset, size,
                                       pSrc, numEventsInWaitList,
                                       phEventWaitList, phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urEnqueueMemBufferReadRect(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingRead,
    ur_rect_offset_t bufferOrigin, ur_rect_offset_t hostOrigin,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pDst,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) try {
  return hQueue->enqueueMemBufferReadRect(
      hBuffer, blockingRead, bufferOrigin, hostOrigin, region, bufferRowPitch,
      bufferSlicePitch, hostRowPitch, hostSlicePitch, pDst, numEventsInWaitList,
      phEventWaitList, phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urEnqueueMemBufferWriteRect(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingWrite,
    ur_rect_offset_t bufferOrigin, ur_rect_offset_t hostOrigin,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pSrc,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) try {
  return hQueue->enqueueMemBufferWriteRect(
      hBuffer, blockingWrite, bufferOrigin, hostOrigin, region, bufferRowPitch,
      bufferSlicePitch, hostRowPitch, hostSlicePitch, pSrc, numEventsInWaitList,
      phEventWaitList, phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urEnqueueMemBufferCopy(ur_queue_handle_t hQueue,
                                   ur_mem_handle_t hBufferSrc,
                                   ur_mem_handle_t hBufferDst, size_t srcOffset,
                                   size_t dstOffset, size_t size,
                                   uint32_t numEventsInWaitList,
                                   const ur_event_handle_t *phEventWaitList,
                                   ur_event_handle_t *phEvent) try {
  return hQueue->enqueueMemBufferCopy(hBufferSrc, hBufferDst, srcOffset,
                                      dstOffset, size, numEventsInWaitList,
                                      phEventWaitList, phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urEnqueueMemBufferCopyRect(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBufferSrc,
    ur_mem_handle_t hBufferDst, ur_rect_offset_t srcOrigin,
    ur_rect_offset_t dstOrigin, ur_rect_region_t region, size_t srcRowPitch,
    size_t srcSlicePitch, size_t dstRowPitch, size_t dstSlicePitch,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) try {
  return hQueue->enqueueMemBufferCopyRect(
      hBufferSrc, hBufferDst, srcOrigin, dstOrigin, region, srcRowPitch,
      srcSlicePitch, dstRowPitch, dstSlicePitch, numEventsInWaitList,
      phEventWaitList, phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urEnqueueMemBufferFill(ur_queue_handle_t hQueue,
                                   ur_mem_handle_t hBuffer,
                                   const void *pPattern, size_t patternSize,
                                   size_t offset, size_t size,
                                   uint32_t numEventsInWaitList,
                                   const ur_event_handle_t *phEventWaitList,
                                   ur_event_handle_t *phEvent) try {
  return hQueue->enqueueMemBufferFill(hBuffer, pPattern, patternSize, offset,
                                      size, numEventsInWaitList,
                                      phEventWaitList, phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urEnqueueMemImageRead(
    ur_queue_handle_t hQueue, ur_mem_handle_t hImage, bool blockingRead,
    ur_rect_offset_t origin, ur_rect_region_t region, size_t rowPitch,
    size_t slicePitch, void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) try {
  return hQueue->enqueueMemImageRead(
      hImage, blockingRead, origin, region, rowPitch, slicePitch, pDst,
      numEventsInWaitList, phEventWaitList, phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urEnqueueMemImageWrite(
    ur_queue_handle_t hQueue, ur_mem_handle_t hImage, bool blockingWrite,
    ur_rect_offset_t origin, ur_rect_region_t region, size_t rowPitch,
    size_t slicePitch, void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) try {
  return hQueue->enqueueMemImageWrite(
      hImage, blockingWrite, origin, region, rowPitch, slicePitch, pSrc,
      numEventsInWaitList, phEventWaitList, phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t
urEnqueueMemImageCopy(ur_queue_handle_t hQueue, ur_mem_handle_t hImageSrc,
                      ur_mem_handle_t hImageDst, ur_rect_offset_t srcOrigin,
                      ur_rect_offset_t dstOrigin, ur_rect_region_t region,
                      uint32_t numEventsInWaitList,
                      const ur_event_handle_t *phEventWaitList,
                      ur_event_handle_t *phEvent) try {
  return hQueue->enqueueMemImageCopy(hImageSrc, hImageDst, srcOrigin, dstOrigin,
                                     region, numEventsInWaitList,
                                     phEventWaitList, phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urEnqueueMemBufferMap(ur_queue_handle_t hQueue,
                                  ur_mem_handle_t hBuffer, bool blockingMap,
                                  ur_map_flags_t mapFlags, size_t offset,
                                  size_t size, uint32_t numEventsInWaitList,
                                  const ur_event_handle_t *phEventWaitList,
                                  ur_event_handle_t *phEvent,
                                  void **ppRetMap) try {
  return hQueue->enqueueMemBufferMap(hBuffer, blockingMap, mapFlags, offset,
                                     size, numEventsInWaitList, phEventWaitList,
                                     phEvent, ppRetMap);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urEnqueueMemUnmap(ur_queue_handle_t hQueue, ur_mem_handle_t hMem,
                              void *pMappedPtr, uint32_t numEventsInWaitList,
                              const ur_event_handle_t *phEventWaitList,
                              ur_event_handle_t *phEvent) try {
  return hQueue->enqueueMemUnmap(hMem, pMappedPtr, numEventsInWaitList,
                                 phEventWaitList, phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urEnqueueUSMFill(ur_queue_handle_t hQueue, void *pMem,
                             size_t patternSize, const void *pPattern,
                             size_t size, uint32_t numEventsInWaitList,
                             const ur_event_handle_t *phEventWaitList,
                             ur_event_handle_t *phEvent) try {
  return hQueue->enqueueUSMFill(pMem, patternSize, pPattern, size,
                                numEventsInWaitList, phEventWaitList, phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urEnqueueUSMMemcpy(ur_queue_handle_t hQueue, bool blocking,
                               void *pDst, const void *pSrc, size_t size,
                               uint32_t numEventsInWaitList,
                               const ur_event_handle_t *phEventWaitList,
                               ur_event_handle_t *phEvent) try {
  return hQueue->enqueueUSMMemcpy(blocking, pDst, pSrc, size,
                                  numEventsInWaitList, phEventWaitList,
                                  phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urEnqueueUSMPrefetch(ur_queue_handle_t hQueue, const void *pMem,
                                 size_t size, ur_usm_migration_flags_t flags,
                                 uint32_t numEventsInWaitList,
                                 const ur_event_handle_t *phEventWaitList,
                                 ur_event_handle_t *phEvent) try {
  return hQueue->enqueueUSMPrefetch(pMem, size, flags, numEventsInWaitList,
                                    phEventWaitList, phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urEnqueueUSMAdvise(ur_queue_handle_t hQueue, const void *pMem,
                               size_t size, ur_usm_advice_flags_t advice,
                               ur_event_handle_t *phEvent) try {
  return hQueue->enqueueUSMAdvise(pMem, size, advice, phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urEnqueueUSMFill2D(ur_queue_handle_t hQueue, void *pMem,
                               size_t pitch, size_t patternSize,
                               const void *pPattern, size_t width,
                               size_t height, uint32_t numEventsInWaitList,
                               const ur_event_handle_t *phEventWaitList,
                               ur_event_handle_t *phEvent) try {
  return hQueue->enqueueUSMFill2D(pMem, pitch, patternSize, pPattern, width,
                                  height, numEventsInWaitList, phEventWaitList,
                                  phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urEnqueueUSMMemcpy2D(ur_queue_handle_t hQueue, bool blocking,
                                 void *pDst, size_t dstPitch, const void *pSrc,
                                 size_t srcPitch, size_t width, size_t height,
                                 uint32_t numEventsInWaitList,
                                 const ur_event_handle_t *phEventWaitList,
                                 ur_event_handle_t *phEvent) try {
  return hQueue->enqueueUSMMemcpy2D(blocking, pDst, dstPitch, pSrc, srcPitch,
                                    width, height, numEventsInWaitList,
                                    phEventWaitList, phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urEnqueueDeviceGlobalVariableWrite(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram, const char *name,
    bool blockingWrite, size_t count, size_t offset, const void *pSrc,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) try {
  return hQueue->enqueueDeviceGlobalVariableWrite(
      hProgram, name, blockingWrite, count, offset, pSrc, numEventsInWaitList,
      phEventWaitList, phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urEnqueueDeviceGlobalVariableRead(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram, const char *name,
    bool blockingRead, size_t count, size_t offset, void *pDst,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) try {
  return hQueue->enqueueDeviceGlobalVariableRead(
      hProgram, name, blockingRead, count, offset, pDst, numEventsInWaitList,
      phEventWaitList, phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urEnqueueReadHostPipe(ur_queue_handle_t hQueue,
                                  ur_program_handle_t hProgram,
                                  const char *pipe_symbol, bool blocking,
                                  void *pDst, size_t size,
                                  uint32_t numEventsInWaitList,
                                  const ur_event_handle_t *phEventWaitList,
                                  ur_event_handle_t *phEvent) try {
  return hQueue->enqueueReadHostPipe(hProgram, pipe_symbol, blocking, pDst,
                                     size, numEventsInWaitList, phEventWaitList,
                                     phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urEnqueueWriteHostPipe(ur_queue_handle_t hQueue,
                                   ur_program_handle_t hProgram,
                                   const char *pipe_symbol, bool blocking,
                                   void *pSrc, size_t size,
                                   uint32_t numEventsInWaitList,
                                   const ur_event_handle_t *phEventWaitList,
                                   ur_event_handle_t *phEvent) try {
  return hQueue->enqueueWriteHostPipe(hProgram, pipe_symbol, blocking, pSrc,
                                      size, numEventsInWaitList,
                                      phEventWaitList, phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urBindlessImagesImageCopyExp(
    ur_queue_handle_t hQueue, const void *pSrc, void *pDst,
    const ur_image_desc_t *pSrcImageDesc, const ur_image_desc_t *pDstImageDesc,
    const ur_image_format_t *pSrcImageFormat,
    const ur_image_format_t *pDstImageFormat,
    ur_exp_image_copy_region_t *pCopyRegion,
    ur_exp_image_copy_flags_t imageCopyFlags, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) try {
  return hQueue->bindlessImagesImageCopyExp(
      pSrc, pDst, pSrcImageDesc, pDstImageDesc, pSrcImageFormat,
      pDstImageFormat, pCopyRegion, imageCopyFlags, numEventsInWaitList,
      phEventWaitList, phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urBindlessImagesWaitExternalSemaphoreExp(
    ur_queue_handle_t hQueue, ur_exp_external_semaphore_handle_t hSemaphore,
    bool hasWaitValue, uint64_t waitValue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) try {
  return hQueue->bindlessImagesWaitExternalSemaphoreExp(
      hSemaphore, hasWaitValue, waitValue, numEventsInWaitList, phEventWaitList,
      phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urBindlessImagesSignalExternalSemaphoreExp(
    ur_queue_handle_t hQueue, ur_exp_external_semaphore_handle_t hSemaphore,
    bool hasSignalValue, uint64_t signalValue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) try {
  return hQueue->bindlessImagesSignalExternalSemaphoreExp(
      hSemaphore, hasSignalValue, signalValue, numEventsInWaitList,
      phEventWaitList, phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urEnqueueCooperativeKernelLaunchExp(
    ur_queue_handle_t hQueue, ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) try {
  return hQueue->enqueueCooperativeKernelLaunchExp(
      hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize,
      numEventsInWaitList, phEventWaitList, phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urEnqueueTimestampRecordingExp(
    ur_queue_handle_t hQueue, bool blocking, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) try {
  return hQueue->enqueueTimestampRecordingExp(blocking, numEventsInWaitList,
                                              phEventWaitList, phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urEnqueueKernelLaunchCustomExp(
    ur_queue_handle_t hQueue, ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numPropsInLaunchPropList,
    const ur_exp_launch_property_t *launchPropList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) try {
  return hQueue->enqueueKernelLaunchCustomExp(
      hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize,
      numPropsInLaunchPropList, launchPropList, numEventsInWaitList,
      phEventWaitList, phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urEnqueueEventsWaitWithBarrierExt(
    ur_queue_handle_t hQueue,
    const ur_exp_enqueue_ext_properties_t *pProperties,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) try {
  return hQueue->enqueueEventsWaitWithBarrierExt(
      pProperties, numEventsInWaitList, phEventWaitList, phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urEnqueueNativeCommandExp(
    ur_queue_handle_t hQueue,
    ur_exp_enqueue_native_command_function_t pfnNativeEnqueue, void *data,
    uint32_t numMemsInMemList, const ur_mem_handle_t *phMemList,
    const ur_exp_enqueue_native_command_properties_t *pProperties,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) try {
  return hQueue->enqueueNativeCommandExp(
      pfnNativeEnqueue, data, numMemsInMemList, phMemList, pProperties,
      numEventsInWaitList, phEventWaitList, phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
} // namespace ur::level_zero