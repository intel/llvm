//===--------- ur_interface_loader.hpp - Level Zero v2 Adapter ---------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <ur_api.h>
#include <ur_ddi.h>

// Shared function declarations live in ur::level_zero namespace.
// Include the v1 header which declares all of them.
#include "../ur_interface_loader.hpp"

// v2-specific function declarations
namespace ur::level_zero_v2 {
ur_result_t urContextCreate(uint32_t DeviceCount,
                            const ur_device_handle_t *phDevices,
                            const ur_context_properties_t *pProperties,
                            ur_context_handle_t *phContext);
ur_result_t urContextRetain(ur_context_handle_t hContext);
ur_result_t urContextRelease(ur_context_handle_t hContext);
ur_result_t urContextGetInfo(ur_context_handle_t hContext,
                             ur_context_info_t propName, size_t propSize,
                             void *pPropValue, size_t *pPropSizeRet);
ur_result_t urContextGetNativeHandle(ur_context_handle_t hContext,
                                     ur_native_handle_t *phNativeContext);
ur_result_t urContextCreateWithNativeHandle(
    ur_native_handle_t hNativeContext, ur_adapter_handle_t hAdapter,
    uint32_t numDevices, const ur_device_handle_t *phDevices,
    const ur_context_native_properties_t *pProperties,
    ur_context_handle_t *phContext);
ur_result_t
urContextSetExtendedDeleter(ur_context_handle_t hContext,
                            ur_context_extended_deleter_t pfnDeleter,
                            void *pUserData);
ur_result_t urMemImageCreate(ur_context_handle_t hContext, ur_mem_flags_t flags,
                             const ur_image_format_t *pImageFormat,
                             const ur_image_desc_t *pImageDesc, void *pHost,
                             ur_mem_handle_t *phMem);
ur_result_t urMemBufferCreate(ur_context_handle_t hContext,
                              ur_mem_flags_t flags, size_t size,
                              const ur_buffer_properties_t *pProperties,
                              ur_mem_handle_t *phBuffer);
ur_result_t urMemRetain(ur_mem_handle_t hMem);
ur_result_t urMemRelease(ur_mem_handle_t hMem);
ur_result_t urMemBufferPartition(ur_mem_handle_t hBuffer, ur_mem_flags_t flags,
                                 ur_buffer_create_type_t bufferCreateType,
                                 const ur_buffer_region_t *pRegion,
                                 ur_mem_handle_t *phMem);
ur_result_t urMemGetNativeHandle(ur_mem_handle_t hMem,
                                 ur_device_handle_t hDevice,
                                 ur_native_handle_t *phNativeMem);
ur_result_t urMemBufferCreateWithNativeHandle(
    ur_native_handle_t hNativeMem, ur_context_handle_t hContext,
    const ur_mem_native_properties_t *pProperties, ur_mem_handle_t *phMem);
ur_result_t urMemImageCreateWithNativeHandle(
    ur_native_handle_t hNativeMem, ur_context_handle_t hContext,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    const ur_mem_native_properties_t *pProperties, ur_mem_handle_t *phMem);
ur_result_t urMemGetInfo(ur_mem_handle_t hMemory, ur_mem_info_t propName,
                         size_t propSize, void *pPropValue,
                         size_t *pPropSizeRet);
ur_result_t urMemImageGetInfo(ur_mem_handle_t hMemory, ur_image_info_t propName,
                              size_t propSize, void *pPropValue,
                              size_t *pPropSizeRet);
ur_result_t urUSMHostAlloc(ur_context_handle_t hContext,
                           const ur_usm_desc_t *pUSMDesc,
                           ur_usm_pool_handle_t pool, size_t size,
                           void **ppMem);
ur_result_t urUSMDeviceAlloc(ur_context_handle_t hContext,
                             ur_device_handle_t hDevice,
                             const ur_usm_desc_t *pUSMDesc,
                             ur_usm_pool_handle_t pool, size_t size,
                             void **ppMem);
ur_result_t urUSMSharedAlloc(ur_context_handle_t hContext,
                             ur_device_handle_t hDevice,
                             const ur_usm_desc_t *pUSMDesc,
                             ur_usm_pool_handle_t pool, size_t size,
                             void **ppMem);
ur_result_t urUSMFree(ur_context_handle_t hContext, void *pMem);
ur_result_t urUSMGetMemAllocInfo(ur_context_handle_t hContext, const void *pMem,
                                 ur_usm_alloc_info_t propName, size_t propSize,
                                 void *pPropValue, size_t *pPropSizeRet);
ur_result_t urUSMPoolCreate(ur_context_handle_t hContext,
                            ur_usm_pool_desc_t *pPoolDesc,
                            ur_usm_pool_handle_t *ppPool);
ur_result_t urUSMPoolRetain(ur_usm_pool_handle_t pPool);
ur_result_t urUSMPoolRelease(ur_usm_pool_handle_t pPool);
ur_result_t urUSMPoolGetInfo(ur_usm_pool_handle_t hPool,
                             ur_usm_pool_info_t propName, size_t propSize,
                             void *pPropValue, size_t *pPropSizeRet);
ur_result_t urKernelCreate(ur_program_handle_t hProgram,
                           const char *pKernelName,
                           ur_kernel_handle_t *phKernel);
ur_result_t urKernelSetArgValue(
    ur_kernel_handle_t hKernel, uint32_t argIndex, size_t argSize,
    const ur_kernel_arg_value_properties_t *pProperties, const void *pArgValue);
ur_result_t
urKernelSetArgLocal(ur_kernel_handle_t hKernel, uint32_t argIndex,
                    size_t argSize,
                    const ur_kernel_arg_local_properties_t *pProperties);
ur_result_t urKernelGetInfo(ur_kernel_handle_t hKernel,
                            ur_kernel_info_t propName, size_t propSize,
                            void *pPropValue, size_t *pPropSizeRet);
ur_result_t urKernelGetGroupInfo(ur_kernel_handle_t hKernel,
                                 ur_device_handle_t hDevice,
                                 ur_kernel_group_info_t propName,
                                 size_t propSize, void *pPropValue,
                                 size_t *pPropSizeRet);
ur_result_t urKernelGetSubGroupInfo(ur_kernel_handle_t hKernel,
                                    ur_device_handle_t hDevice,
                                    ur_kernel_sub_group_info_t propName,
                                    size_t propSize, void *pPropValue,
                                    size_t *pPropSizeRet);
ur_result_t urKernelRetain(ur_kernel_handle_t hKernel);
ur_result_t urKernelRelease(ur_kernel_handle_t hKernel);
ur_result_t
urKernelSetArgPointer(ur_kernel_handle_t hKernel, uint32_t argIndex,
                      const ur_kernel_arg_pointer_properties_t *pProperties,
                      const void *pArgValue);
ur_result_t
urKernelSetExecInfo(ur_kernel_handle_t hKernel, ur_kernel_exec_info_t propName,
                    size_t propSize,
                    const ur_kernel_exec_info_properties_t *pProperties,
                    const void *pPropValue);
ur_result_t
urKernelSetArgSampler(ur_kernel_handle_t hKernel, uint32_t argIndex,
                      const ur_kernel_arg_sampler_properties_t *pProperties,
                      ur_sampler_handle_t hArgValue);
ur_result_t
urKernelSetArgMemObj(ur_kernel_handle_t hKernel, uint32_t argIndex,
                     const ur_kernel_arg_mem_obj_properties_t *pProperties,
                     ur_mem_handle_t hArgValue);
ur_result_t urKernelSetSpecializationConstants(
    ur_kernel_handle_t hKernel, uint32_t count,
    const ur_specialization_constant_info_t *pSpecConstants);
ur_result_t urKernelGetNativeHandle(ur_kernel_handle_t hKernel,
                                    ur_native_handle_t *phNativeKernel);
ur_result_t
urKernelCreateWithNativeHandle(ur_native_handle_t hNativeKernel,
                               ur_context_handle_t hContext,
                               ur_program_handle_t hProgram,
                               const ur_kernel_native_properties_t *pProperties,
                               ur_kernel_handle_t *phKernel);
ur_result_t urKernelGetSuggestedLocalWorkSize(ur_kernel_handle_t hKernel,
                                              ur_queue_handle_t hQueue,
                                              uint32_t numWorkDim,
                                              const size_t *pGlobalWorkOffset,
                                              const size_t *pGlobalWorkSize,
                                              size_t *pSuggestedLocalWorkSize);
ur_result_t urKernelSuggestMaxCooperativeGroupCount(
    ur_kernel_handle_t hKernel, ur_device_handle_t hDevice, uint32_t workDim,
    const size_t *pLocalWorkSize, size_t dynamicSharedMemorySize,
    uint32_t *pGroupCountRet);
ur_result_t urQueueGetInfo(ur_queue_handle_t hQueue, ur_queue_info_t propName,
                           size_t propSize, void *pPropValue,
                           size_t *pPropSizeRet);
ur_result_t urQueueCreate(ur_context_handle_t hContext,
                          ur_device_handle_t hDevice,
                          const ur_queue_properties_t *pProperties,
                          ur_queue_handle_t *phQueue);
ur_result_t urQueueRetain(ur_queue_handle_t hQueue);
ur_result_t urQueueRelease(ur_queue_handle_t hQueue);
ur_result_t urQueueGetNativeHandle(ur_queue_handle_t hQueue,
                                   ur_queue_native_desc_t *pDesc,
                                   ur_native_handle_t *phNativeQueue);
ur_result_t urQueueCreateWithNativeHandle(
    ur_native_handle_t hNativeQueue, ur_context_handle_t hContext,
    ur_device_handle_t hDevice, const ur_queue_native_properties_t *pProperties,
    ur_queue_handle_t *phQueue);
ur_result_t urQueueFinish(ur_queue_handle_t hQueue);
ur_result_t urQueueFlush(ur_queue_handle_t hQueue);
ur_result_t urEventGetInfo(ur_event_handle_t hEvent, ur_event_info_t propName,
                           size_t propSize, void *pPropValue,
                           size_t *pPropSizeRet);
ur_result_t urEventGetProfilingInfo(ur_event_handle_t hEvent,
                                    ur_profiling_info_t propName,
                                    size_t propSize, void *pPropValue,
                                    size_t *pPropSizeRet);
ur_result_t urEventWait(uint32_t numEvents,
                        const ur_event_handle_t *phEventWaitList);
ur_result_t urEventRetain(ur_event_handle_t hEvent);
ur_result_t urEventRelease(ur_event_handle_t hEvent);
ur_result_t urEventGetNativeHandle(ur_event_handle_t hEvent,
                                   ur_native_handle_t *phNativeEvent);
ur_result_t
urEventCreateWithNativeHandle(ur_native_handle_t hNativeEvent,
                              ur_context_handle_t hContext,
                              const ur_event_native_properties_t *pProperties,
                              ur_event_handle_t *phEvent);
ur_result_t urEventSetCallback(ur_event_handle_t hEvent,
                               ur_execution_info_t execStatus,
                               ur_event_callback_t pfnNotify, void *pUserData);
ur_result_t urEnqueueKernelLaunch(
    ur_queue_handle_t hQueue, ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize,
    const ur_kernel_launch_ext_properties_t *launchPropList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent);
ur_result_t urEnqueueEventsWait(ur_queue_handle_t hQueue,
                                uint32_t numEventsInWaitList,
                                const ur_event_handle_t *phEventWaitList,
                                ur_event_handle_t *phEvent);
ur_result_t urEnqueueEventsWaitWithBarrier(
    ur_queue_handle_t hQueue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent);
ur_result_t urEnqueueMemBufferRead(ur_queue_handle_t hQueue,
                                   ur_mem_handle_t hBuffer, bool blockingRead,
                                   size_t offset, size_t size, void *pDst,
                                   uint32_t numEventsInWaitList,
                                   const ur_event_handle_t *phEventWaitList,
                                   ur_event_handle_t *phEvent);
ur_result_t urEnqueueMemBufferWrite(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingWrite,
    size_t offset, size_t size, const void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent);
ur_result_t urEnqueueMemBufferReadRect(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingRead,
    ur_rect_offset_t bufferOrigin, ur_rect_offset_t hostOrigin,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pDst,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent);
ur_result_t urEnqueueMemBufferWriteRect(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingWrite,
    ur_rect_offset_t bufferOrigin, ur_rect_offset_t hostOrigin,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pSrc,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent);
ur_result_t urEnqueueMemBufferCopy(ur_queue_handle_t hQueue,
                                   ur_mem_handle_t hBufferSrc,
                                   ur_mem_handle_t hBufferDst, size_t srcOffset,
                                   size_t dstOffset, size_t size,
                                   uint32_t numEventsInWaitList,
                                   const ur_event_handle_t *phEventWaitList,
                                   ur_event_handle_t *phEvent);
ur_result_t urEnqueueMemBufferCopyRect(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBufferSrc,
    ur_mem_handle_t hBufferDst, ur_rect_offset_t srcOrigin,
    ur_rect_offset_t dstOrigin, ur_rect_region_t region, size_t srcRowPitch,
    size_t srcSlicePitch, size_t dstRowPitch, size_t dstSlicePitch,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent);
ur_result_t urEnqueueMemBufferFill(ur_queue_handle_t hQueue,
                                   ur_mem_handle_t hBuffer,
                                   const void *pPattern, size_t patternSize,
                                   size_t offset, size_t size,
                                   uint32_t numEventsInWaitList,
                                   const ur_event_handle_t *phEventWaitList,
                                   ur_event_handle_t *phEvent);
ur_result_t urEnqueueMemImageRead(
    ur_queue_handle_t hQueue, ur_mem_handle_t hImage, bool blockingRead,
    ur_rect_offset_t origin, ur_rect_region_t region, size_t rowPitch,
    size_t slicePitch, void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent);
ur_result_t urEnqueueMemImageWrite(
    ur_queue_handle_t hQueue, ur_mem_handle_t hImage, bool blockingWrite,
    ur_rect_offset_t origin, ur_rect_region_t region, size_t rowPitch,
    size_t slicePitch, void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent);
ur_result_t
urEnqueueMemImageCopy(ur_queue_handle_t hQueue, ur_mem_handle_t hImageSrc,
                      ur_mem_handle_t hImageDst, ur_rect_offset_t srcOrigin,
                      ur_rect_offset_t dstOrigin, ur_rect_region_t region,
                      uint32_t numEventsInWaitList,
                      const ur_event_handle_t *phEventWaitList,
                      ur_event_handle_t *phEvent);
ur_result_t urEnqueueMemBufferMap(ur_queue_handle_t hQueue,
                                  ur_mem_handle_t hBuffer, bool blockingMap,
                                  ur_map_flags_t mapFlags, size_t offset,
                                  size_t size, uint32_t numEventsInWaitList,
                                  const ur_event_handle_t *phEventWaitList,
                                  ur_event_handle_t *phEvent, void **ppRetMap);
ur_result_t urEnqueueMemUnmap(ur_queue_handle_t hQueue, ur_mem_handle_t hMem,
                              void *pMappedPtr, uint32_t numEventsInWaitList,
                              const ur_event_handle_t *phEventWaitList,
                              ur_event_handle_t *phEvent);
ur_result_t urEnqueueUSMFill(ur_queue_handle_t hQueue, void *pMem,
                             size_t patternSize, const void *pPattern,
                             size_t size, uint32_t numEventsInWaitList,
                             const ur_event_handle_t *phEventWaitList,
                             ur_event_handle_t *phEvent);
ur_result_t urEnqueueUSMMemcpy(ur_queue_handle_t hQueue, bool blocking,
                               void *pDst, const void *pSrc, size_t size,
                               uint32_t numEventsInWaitList,
                               const ur_event_handle_t *phEventWaitList,
                               ur_event_handle_t *phEvent);
ur_result_t urEnqueueUSMPrefetch(ur_queue_handle_t hQueue, const void *pMem,
                                 size_t size, ur_usm_migration_flags_t flags,
                                 uint32_t numEventsInWaitList,
                                 const ur_event_handle_t *phEventWaitList,
                                 ur_event_handle_t *phEvent);
ur_result_t urEnqueueUSMAdvise(ur_queue_handle_t hQueue, const void *pMem,
                               size_t size, ur_usm_advice_flags_t advice,
                               ur_event_handle_t *phEvent);
ur_result_t urEnqueueUSMFill2D(ur_queue_handle_t hQueue, void *pMem,
                               size_t pitch, size_t patternSize,
                               const void *pPattern, size_t width,
                               size_t height, uint32_t numEventsInWaitList,
                               const ur_event_handle_t *phEventWaitList,
                               ur_event_handle_t *phEvent);
ur_result_t urEnqueueUSMMemcpy2D(ur_queue_handle_t hQueue, bool blocking,
                                 void *pDst, size_t dstPitch, const void *pSrc,
                                 size_t srcPitch, size_t width, size_t height,
                                 uint32_t numEventsInWaitList,
                                 const ur_event_handle_t *phEventWaitList,
                                 ur_event_handle_t *phEvent);
ur_result_t urEnqueueDeviceGlobalVariableWrite(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram, const char *name,
    bool blockingWrite, size_t count, size_t offset, const void *pSrc,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent);
ur_result_t urEnqueueDeviceGlobalVariableRead(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram, const char *name,
    bool blockingRead, size_t count, size_t offset, void *pDst,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent);
ur_result_t urEnqueueReadHostPipe(ur_queue_handle_t hQueue,
                                  ur_program_handle_t hProgram,
                                  const char *pipe_symbol, bool blocking,
                                  void *pDst, size_t size,
                                  uint32_t numEventsInWaitList,
                                  const ur_event_handle_t *phEventWaitList,
                                  ur_event_handle_t *phEvent);
ur_result_t urEnqueueWriteHostPipe(ur_queue_handle_t hQueue,
                                   ur_program_handle_t hProgram,
                                   const char *pipe_symbol, bool blocking,
                                   void *pSrc, size_t size,
                                   uint32_t numEventsInWaitList,
                                   const ur_event_handle_t *phEventWaitList,
                                   ur_event_handle_t *phEvent);
ur_result_t urEnqueueUSMDeviceAllocExp(
    ur_queue_handle_t hQueue, ur_usm_pool_handle_t pPool, const size_t size,
    const ur_exp_async_usm_alloc_properties_t *pProperties,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    void **ppMem, ur_event_handle_t *phEvent);
ur_result_t urEnqueueUSMSharedAllocExp(
    ur_queue_handle_t hQueue, ur_usm_pool_handle_t pPool, const size_t size,
    const ur_exp_async_usm_alloc_properties_t *pProperties,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    void **ppMem, ur_event_handle_t *phEvent);
ur_result_t urEnqueueUSMHostAllocExp(
    ur_queue_handle_t hQueue, ur_usm_pool_handle_t pPool, const size_t size,
    const ur_exp_async_usm_alloc_properties_t *pProperties,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    void **ppMem, ur_event_handle_t *phEvent);
ur_result_t urEnqueueUSMFreeExp(ur_queue_handle_t hQueue,
                                ur_usm_pool_handle_t pPool, void *pMem,
                                uint32_t numEventsInWaitList,
                                const ur_event_handle_t *phEventWaitList,
                                ur_event_handle_t *phEvent);
ur_result_t urUSMPoolCreateExp(ur_context_handle_t hContext,
                               ur_device_handle_t hDevice,
                               ur_usm_pool_desc_t *pPoolDesc,
                               ur_usm_pool_handle_t *pPool);
ur_result_t urUSMPoolDestroyExp(ur_context_handle_t hContext,
                                ur_device_handle_t hDevice,
                                ur_usm_pool_handle_t hPool);
ur_result_t urUSMPoolGetDefaultDevicePoolExp(ur_context_handle_t hContext,
                                             ur_device_handle_t hDevice,
                                             ur_usm_pool_handle_t *pPool);
ur_result_t urUSMPoolGetInfoExp(ur_usm_pool_handle_t hPool,
                                ur_usm_pool_info_t propName, void *pPropValue,
                                size_t *pPropSizeRet);
ur_result_t urUSMPoolSetInfoExp(ur_usm_pool_handle_t hPool,
                                ur_usm_pool_info_t propName, void *pPropValue,
                                size_t propSize);
ur_result_t urUSMPoolSetDevicePoolExp(ur_context_handle_t hContext,
                                      ur_device_handle_t hDevice,
                                      ur_usm_pool_handle_t hPool);
ur_result_t urUSMPoolGetDevicePoolExp(ur_context_handle_t hContext,
                                      ur_device_handle_t hDevice,
                                      ur_usm_pool_handle_t *pPool);
ur_result_t urUSMPoolTrimToExp(ur_context_handle_t hContext,
                               ur_device_handle_t hDevice,
                               ur_usm_pool_handle_t hPool,
                               size_t minBytesToKeep);
ur_result_t urUSMContextMemcpyExp(ur_context_handle_t hContext, void *pDst,
                                  const void *pSrc, size_t size);
ur_result_t urUSMImportExp(ur_context_handle_t hContext, void *pMem,
                           size_t size);
ur_result_t urUSMReleaseExp(ur_context_handle_t hContext, void *pMem);
ur_result_t urIPCGetMemHandleExp(ur_context_handle_t hContext, void *pMem,
                                 void **ppIPCMemHandleData,
                                 size_t *pIPCMemHandleDataSizeRet);
ur_result_t urIPCPutMemHandleExp(ur_context_handle_t hContext,
                                 void *pIPCMemHandleData);
ur_result_t urIPCOpenMemHandleExp(ur_context_handle_t hContext,
                                  ur_device_handle_t hDevice,
                                  void *pIPCMemHandleData,
                                  size_t ipcMemHandleDataSize, void **ppMem);
ur_result_t urIPCCloseMemHandleExp(ur_context_handle_t hContext, void *pMem);
ur_result_t
urCommandBufferCreateExp(ur_context_handle_t hContext,
                         ur_device_handle_t hDevice,
                         const ur_exp_command_buffer_desc_t *pCommandBufferDesc,
                         ur_exp_command_buffer_handle_t *phCommandBuffer);
ur_result_t
urCommandBufferRetainExp(ur_exp_command_buffer_handle_t hCommandBuffer);
ur_result_t
urCommandBufferReleaseExp(ur_exp_command_buffer_handle_t hCommandBuffer);
ur_result_t
urCommandBufferFinalizeExp(ur_exp_command_buffer_handle_t hCommandBuffer);
ur_result_t urCommandBufferAppendKernelLaunchExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_kernel_handle_t hKernel,
    uint32_t workDim, const size_t *pGlobalWorkOffset,
    const size_t *pGlobalWorkSize, const size_t *pLocalWorkSize,
    uint32_t numKernelAlternatives, ur_kernel_handle_t *phKernelAlternatives,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint, ur_event_handle_t *phEvent,
    ur_exp_command_buffer_command_handle_t *phCommand);
ur_result_t urCommandBufferAppendUSMMemcpyExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, void *pDst, const void *pSrc,
    size_t size, uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint, ur_event_handle_t *phEvent,
    ur_exp_command_buffer_command_handle_t *phCommand);
ur_result_t urCommandBufferAppendUSMFillExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, void *pMemory,
    const void *pPattern, size_t patternSize, size_t size,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint, ur_event_handle_t *phEvent,
    ur_exp_command_buffer_command_handle_t *phCommand);
ur_result_t urCommandBufferAppendMemBufferCopyExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hSrcMem,
    ur_mem_handle_t hDstMem, size_t srcOffset, size_t dstOffset, size_t size,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint, ur_event_handle_t *phEvent,
    ur_exp_command_buffer_command_handle_t *phCommand);
ur_result_t urCommandBufferAppendMemBufferWriteExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    size_t offset, size_t size, const void *pSrc,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint, ur_event_handle_t *phEvent,
    ur_exp_command_buffer_command_handle_t *phCommand);
ur_result_t urCommandBufferAppendMemBufferReadExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    size_t offset, size_t size, void *pDst, uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint, ur_event_handle_t *phEvent,
    ur_exp_command_buffer_command_handle_t *phCommand);
ur_result_t urCommandBufferAppendMemBufferCopyRectExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hSrcMem,
    ur_mem_handle_t hDstMem, ur_rect_offset_t srcOrigin,
    ur_rect_offset_t dstOrigin, ur_rect_region_t region, size_t srcRowPitch,
    size_t srcSlicePitch, size_t dstRowPitch, size_t dstSlicePitch,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint, ur_event_handle_t *phEvent,
    ur_exp_command_buffer_command_handle_t *phCommand);
ur_result_t urCommandBufferAppendMemBufferWriteRectExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    ur_rect_offset_t bufferOffset, ur_rect_offset_t hostOffset,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pSrc,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint, ur_event_handle_t *phEvent,
    ur_exp_command_buffer_command_handle_t *phCommand);
ur_result_t urCommandBufferAppendMemBufferReadRectExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    ur_rect_offset_t bufferOffset, ur_rect_offset_t hostOffset,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pDst,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint, ur_event_handle_t *phEvent,
    ur_exp_command_buffer_command_handle_t *phCommand);
ur_result_t urCommandBufferAppendMemBufferFillExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    const void *pPattern, size_t patternSize, size_t offset, size_t size,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint, ur_event_handle_t *phEvent,
    ur_exp_command_buffer_command_handle_t *phCommand);
ur_result_t urCommandBufferAppendUSMPrefetchExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, const void *pMemory,
    size_t size, ur_usm_migration_flags_t flags,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint, ur_event_handle_t *phEvent,
    ur_exp_command_buffer_command_handle_t *phCommand);
ur_result_t urCommandBufferAppendUSMAdviseExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, const void *pMemory,
    size_t size, ur_usm_advice_flags_t advice, uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint, ur_event_handle_t *phEvent,
    ur_exp_command_buffer_command_handle_t *phCommand);
ur_result_t urCommandBufferAppendNativeCommandExp(
    ur_exp_command_buffer_handle_t hCommandBuffer,
    ur_exp_command_buffer_native_command_function_t pfnNativeCommand,
    void *pData, ur_exp_command_buffer_handle_t hChildCommandBuffer,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint);
ur_result_t urEnqueueCommandBufferExp(
    ur_queue_handle_t hQueue, ur_exp_command_buffer_handle_t hCommandBuffer,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent);
ur_result_t urCommandBufferUpdateKernelLaunchExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, uint32_t numKernelUpdates,
    const ur_exp_command_buffer_update_kernel_launch_desc_t
        *pUpdateKernelLaunch);
ur_result_t urCommandBufferUpdateSignalEventExp(
    ur_exp_command_buffer_command_handle_t hCommand,
    ur_event_handle_t *phSignalEvent);
ur_result_t urCommandBufferUpdateWaitEventsExp(
    ur_exp_command_buffer_command_handle_t hCommand,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList);
ur_result_t
urCommandBufferGetInfoExp(ur_exp_command_buffer_handle_t hCommandBuffer,
                          ur_exp_command_buffer_info_t propName,
                          size_t propSize, void *pPropValue,
                          size_t *pPropSizeRet);
ur_result_t
urCommandBufferGetNativeHandleExp(ur_exp_command_buffer_handle_t hCommandBuffer,
                                  ur_native_handle_t *phNativeCommandBuffer);
ur_result_t urEnqueueTimestampRecordingExp(
    ur_queue_handle_t hQueue, bool blocking, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent);
ur_result_t urEnqueueKernelLaunchWithArgsExp(
    ur_queue_handle_t hQueue, ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numArgs,
    const ur_exp_kernel_arg_properties_t *pArgs,
    const ur_kernel_launch_ext_properties_t *launchPropList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent);
ur_result_t urEnqueueHostTaskExp(
    ur_queue_handle_t hQueue, ur_exp_host_task_function_t pfnHostTask,
    void *data, const ur_exp_host_task_properties_t *pProperties,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent);
ur_result_t urEnqueueEventsWaitWithBarrierExt(
    ur_queue_handle_t hQueue,
    const ur_exp_enqueue_ext_properties_t *pProperties,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent);
ur_result_t urEnqueueNativeCommandExp(
    ur_queue_handle_t hQueue,
    ur_exp_enqueue_native_command_function_t pfnNativeEnqueue, void *data,
    uint32_t numMemsInMemList, const ur_mem_handle_t *phMemList,
    const ur_exp_enqueue_native_command_properties_t *pProperties,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent);
ur_result_t urGraphCreateExp(ur_context_handle_t hContext,
                             ur_exp_graph_handle_t *phGraph);
ur_result_t urQueueBeginGraphCaptureExp(ur_queue_handle_t hQueue);
ur_result_t urQueueBeginCaptureIntoGraphExp(ur_queue_handle_t hQueue,
                                            ur_exp_graph_handle_t hGraph);
ur_result_t urQueueEndGraphCaptureExp(ur_queue_handle_t hQueue,
                                      ur_exp_graph_handle_t *phGraph);
ur_result_t
urGraphInstantiateGraphExp(ur_exp_graph_handle_t hGraph,
                           ur_exp_executable_graph_handle_t *phExecGraph);
ur_result_t urEnqueueGraphExp(ur_queue_handle_t hQueue,
                              ur_exp_executable_graph_handle_t hGraph,
                              uint32_t numEventsInWaitList,
                              const ur_event_handle_t *phEventWaitList,
                              ur_event_handle_t *phEvent);
ur_result_t urGraphDestroyExp(ur_exp_graph_handle_t hGraph);
ur_result_t urGraphExecutableGraphDestroyExp(
    ur_exp_executable_graph_handle_t hExecutableGraph);
ur_result_t urQueueIsGraphCaptureEnabledExp(ur_queue_handle_t hQueue,
                                            bool *pResult);
ur_result_t urGraphIsEmptyExp(ur_exp_graph_handle_t hGraph, bool *pResult);
ur_result_t urGraphDumpContentsExp(ur_exp_graph_handle_t hGraph,
                                   const char *filePath);
#ifdef UR_STATIC_ADAPTER_LEVEL_ZERO_V2
ur_result_t urAdapterGetDdiTables(ur_dditable_t *ddi);
#endif

struct ddi_getter {
  const static ur_dditable_t *value();
};
} // namespace ur::level_zero_v2
