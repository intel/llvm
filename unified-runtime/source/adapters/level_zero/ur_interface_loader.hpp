//===--------- ur_interface_loader.hpp - Level Zero Adapter ------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <ur_api.h>
#include <ur_ddi.h>

namespace ur::level_zero {
ur_result_t urAdapterGet(uint32_t NumEntries, ur_adapter_handle_t *phAdapters,
                         uint32_t *pNumAdapters);
ur_result_t urAdapterRelease(ur_adapter_handle_t hAdapter);
ur_result_t urAdapterRetain(ur_adapter_handle_t hAdapter);
ur_result_t urAdapterGetLastError(ur_adapter_handle_t hAdapter,
                                  const char **ppMessage, int32_t *pError);
ur_result_t urAdapterGetInfo(ur_adapter_handle_t hAdapter,
                             ur_adapter_info_t propName, size_t propSize,
                             void *pPropValue, size_t *pPropSizeRet);
ur_result_t urPlatformGet(ur_adapter_handle_t *phAdapters, uint32_t NumAdapters,
                          uint32_t NumEntries,
                          ur_platform_handle_t *phPlatforms,
                          uint32_t *pNumPlatforms);
ur_result_t urPlatformGetInfo(ur_platform_handle_t hPlatform,
                              ur_platform_info_t propName, size_t propSize,
                              void *pPropValue, size_t *pPropSizeRet);
ur_result_t urPlatformGetApiVersion(ur_platform_handle_t hPlatform,
                                    ur_api_version_t *pVersion);
ur_result_t urPlatformGetNativeHandle(ur_platform_handle_t hPlatform,
                                      ur_native_handle_t *phNativePlatform);
ur_result_t urPlatformCreateWithNativeHandle(
    ur_native_handle_t hNativePlatform, ur_adapter_handle_t hAdapter,
    const ur_platform_native_properties_t *pProperties,
    ur_platform_handle_t *phPlatform);
ur_result_t urPlatformGetBackendOption(ur_platform_handle_t hPlatform,
                                       const char *pFrontendOption,
                                       const char **ppPlatformOption);
ur_result_t urDeviceGet(ur_platform_handle_t hPlatform,
                        ur_device_type_t DeviceType, uint32_t NumEntries,
                        ur_device_handle_t *phDevices, uint32_t *pNumDevices);
ur_result_t urDeviceGetInfo(ur_device_handle_t hDevice,
                            ur_device_info_t propName, size_t propSize,
                            void *pPropValue, size_t *pPropSizeRet);
ur_result_t urDeviceRetain(ur_device_handle_t hDevice);
ur_result_t urDeviceRelease(ur_device_handle_t hDevice);
ur_result_t
urDevicePartition(ur_device_handle_t hDevice,
                  const ur_device_partition_properties_t *pProperties,
                  uint32_t NumDevices, ur_device_handle_t *phSubDevices,
                  uint32_t *pNumDevicesRet);
ur_result_t urDeviceSelectBinary(ur_device_handle_t hDevice,
                                 const ur_device_binary_t *pBinaries,
                                 uint32_t NumBinaries,
                                 uint32_t *pSelectedBinary);
ur_result_t urDeviceGetNativeHandle(ur_device_handle_t hDevice,
                                    ur_native_handle_t *phNativeDevice);
ur_result_t
urDeviceCreateWithNativeHandle(ur_native_handle_t hNativeDevice,
                               ur_adapter_handle_t hAdapter,
                               const ur_device_native_properties_t *pProperties,
                               ur_device_handle_t *phDevice);
ur_result_t urDeviceGetGlobalTimestamps(ur_device_handle_t hDevice,
                                        uint64_t *pDeviceTimestamp,
                                        uint64_t *pHostTimestamp);
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
ur_result_t urSamplerCreate(ur_context_handle_t hContext,
                            const ur_sampler_desc_t *pDesc,
                            ur_sampler_handle_t *phSampler);
ur_result_t urSamplerRetain(ur_sampler_handle_t hSampler);
ur_result_t urSamplerRelease(ur_sampler_handle_t hSampler);
ur_result_t urSamplerGetInfo(ur_sampler_handle_t hSampler,
                             ur_sampler_info_t propName, size_t propSize,
                             void *pPropValue, size_t *pPropSizeRet);
ur_result_t urSamplerGetNativeHandle(ur_sampler_handle_t hSampler,
                                     ur_native_handle_t *phNativeSampler);
ur_result_t urSamplerCreateWithNativeHandle(
    ur_native_handle_t hNativeSampler, ur_context_handle_t hContext,
    const ur_sampler_native_properties_t *pProperties,
    ur_sampler_handle_t *phSampler);
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
ur_result_t urVirtualMemGranularityGetInfo(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_virtual_mem_granularity_info_t propName, size_t propSize,
    void *pPropValue, size_t *pPropSizeRet);
ur_result_t urVirtualMemReserve(ur_context_handle_t hContext,
                                const void *pStart, size_t size,
                                void **ppStart);
ur_result_t urVirtualMemFree(ur_context_handle_t hContext, const void *pStart,
                             size_t size);
ur_result_t urVirtualMemMap(ur_context_handle_t hContext, const void *pStart,
                            size_t size, ur_physical_mem_handle_t hPhysicalMem,
                            size_t offset, ur_virtual_mem_access_flags_t flags);
ur_result_t urVirtualMemUnmap(ur_context_handle_t hContext, const void *pStart,
                              size_t size);
ur_result_t urVirtualMemSetAccess(ur_context_handle_t hContext,
                                  const void *pStart, size_t size,
                                  ur_virtual_mem_access_flags_t flags);
ur_result_t urVirtualMemGetInfo(ur_context_handle_t hContext,
                                const void *pStart, size_t size,
                                ur_virtual_mem_info_t propName, size_t propSize,
                                void *pPropValue, size_t *pPropSizeRet);
ur_result_t urPhysicalMemCreate(ur_context_handle_t hContext,
                                ur_device_handle_t hDevice, size_t size,
                                const ur_physical_mem_properties_t *pProperties,
                                ur_physical_mem_handle_t *phPhysicalMem);
ur_result_t urPhysicalMemRetain(ur_physical_mem_handle_t hPhysicalMem);
ur_result_t urPhysicalMemRelease(ur_physical_mem_handle_t hPhysicalMem);
ur_result_t urPhysicalMemGetInfo(ur_physical_mem_handle_t hPhysicalMem,
                                 ur_physical_mem_info_t propName,
                                 size_t propSize, void *pPropValue,
                                 size_t *pPropSizeRet);
ur_result_t urProgramCreateWithIL(ur_context_handle_t hContext, const void *pIL,
                                  size_t length,
                                  const ur_program_properties_t *pProperties,
                                  ur_program_handle_t *phProgram);
ur_result_t urProgramCreateWithBinary(
    ur_context_handle_t hContext, uint32_t numDevices,
    ur_device_handle_t *phDevices, size_t *pLengths, const uint8_t **ppBinaries,
    const ur_program_properties_t *pProperties, ur_program_handle_t *phProgram);
ur_result_t urProgramBuild(ur_context_handle_t hContext,
                           ur_program_handle_t hProgram, const char *pOptions);
ur_result_t urProgramCompile(ur_context_handle_t hContext,
                             ur_program_handle_t hProgram,
                             const char *pOptions);
ur_result_t urProgramLink(ur_context_handle_t hContext, uint32_t count,
                          const ur_program_handle_t *phPrograms,
                          const char *pOptions, ur_program_handle_t *phProgram);
ur_result_t urProgramRetain(ur_program_handle_t hProgram);
ur_result_t urProgramRelease(ur_program_handle_t hProgram);
ur_result_t urProgramGetFunctionPointer(ur_device_handle_t hDevice,
                                        ur_program_handle_t hProgram,
                                        const char *pFunctionName,
                                        void **ppFunctionPointer);
ur_result_t urProgramGetGlobalVariablePointer(
    ur_device_handle_t hDevice, ur_program_handle_t hProgram,
    const char *pGlobalVariableName, size_t *pGlobalVariableSizeRet,
    void **ppGlobalVariablePointerRet);
ur_result_t urProgramGetInfo(ur_program_handle_t hProgram,
                             ur_program_info_t propName, size_t propSize,
                             void *pPropValue, size_t *pPropSizeRet);
ur_result_t urProgramGetBuildInfo(ur_program_handle_t hProgram,
                                  ur_device_handle_t hDevice,
                                  ur_program_build_info_t propName,
                                  size_t propSize, void *pPropValue,
                                  size_t *pPropSizeRet);
ur_result_t urProgramSetSpecializationConstants(
    ur_program_handle_t hProgram, uint32_t count,
    const ur_specialization_constant_info_t *pSpecConstants);
ur_result_t urProgramGetNativeHandle(ur_program_handle_t hProgram,
                                     ur_native_handle_t *phNativeProgram);
ur_result_t urProgramCreateWithNativeHandle(
    ur_native_handle_t hNativeProgram, ur_context_handle_t hContext,
    const ur_program_native_properties_t *pProperties,
    ur_program_handle_t *phProgram);
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
    const size_t *pLocalWorkSize, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent);
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
ur_result_t urUSMPoolSetThresholdExp(ur_context_handle_t hContext,
                                     ur_device_handle_t hDevice,
                                     ur_usm_pool_handle_t hPool,
                                     size_t newThreshold);
ur_result_t urUSMPoolGetDefaultDevicePoolExp(ur_context_handle_t hContext,
                                             ur_device_handle_t hDevice,
                                             ur_usm_pool_handle_t *pPool);
ur_result_t urUSMPoolGetInfoExp(ur_usm_pool_handle_t hPool,
                                ur_usm_pool_info_t propName, void *pPropValue,
                                size_t *pPropSizeRet);
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
ur_result_t urUSMPitchedAllocExp(ur_context_handle_t hContext,
                                 ur_device_handle_t hDevice,
                                 const ur_usm_desc_t *pUSMDesc,
                                 ur_usm_pool_handle_t pool, size_t widthInBytes,
                                 size_t height, size_t elementSizeBytes,
                                 void **ppMem, size_t *pResultPitch);
ur_result_t urBindlessImagesUnsampledImageHandleDestroyExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_native_handle_t hImage);
ur_result_t urBindlessImagesSampledImageHandleDestroyExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_native_handle_t hImage);
ur_result_t urBindlessImagesImageAllocateExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    ur_exp_image_mem_native_handle_t *phImageMem);
ur_result_t
urBindlessImagesImageFreeExp(ur_context_handle_t hContext,
                             ur_device_handle_t hDevice,
                             ur_exp_image_mem_native_handle_t hImageMem);
ur_result_t urBindlessImagesUnsampledImageCreateExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_mem_native_handle_t hImageMem,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    ur_exp_image_native_handle_t *phImage);
ur_result_t urBindlessImagesSampledImageCreateExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_mem_native_handle_t hImageMem,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    ur_sampler_handle_t hSampler, ur_exp_image_native_handle_t *phImage);
ur_result_t urBindlessImagesImageCopyExp(
    ur_queue_handle_t hQueue, const void *pSrc, void *pDst,
    const ur_image_desc_t *pSrcImageDesc, const ur_image_desc_t *pDstImageDesc,
    const ur_image_format_t *pSrcImageFormat,
    const ur_image_format_t *pDstImageFormat,
    ur_exp_image_copy_region_t *pCopyRegion,
    ur_exp_image_copy_flags_t imageCopyFlags, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent);
ur_result_t urBindlessImagesImageGetInfoExp(
    ur_context_handle_t hContext, ur_exp_image_mem_native_handle_t hImageMem,
    ur_image_info_t propName, void *pPropValue, size_t *pPropSizeRet);
ur_result_t urBindlessImagesMipmapGetLevelExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_mem_native_handle_t hImageMem, uint32_t mipmapLevel,
    ur_exp_image_mem_native_handle_t *phImageMem);
ur_result_t
urBindlessImagesMipmapFreeExp(ur_context_handle_t hContext,
                              ur_device_handle_t hDevice,
                              ur_exp_image_mem_native_handle_t hMem);
ur_result_t urBindlessImagesImportExternalMemoryExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, size_t size,
    ur_exp_external_mem_type_t memHandleType,
    ur_exp_external_mem_desc_t *pExternalMemDesc,
    ur_exp_external_mem_handle_t *phExternalMem);
ur_result_t urBindlessImagesMapExternalArrayExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    ur_exp_external_mem_handle_t hExternalMem,
    ur_exp_image_mem_native_handle_t *phImageMem);
ur_result_t urBindlessImagesMapExternalLinearMemoryExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, uint64_t offset,
    uint64_t size, ur_exp_external_mem_handle_t hExternalMem, void **ppRetMem);
ur_result_t urBindlessImagesReleaseExternalMemoryExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_external_mem_handle_t hExternalMem);
ur_result_t urBindlessImagesImportExternalSemaphoreExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_external_semaphore_type_t semHandleType,
    ur_exp_external_semaphore_desc_t *pExternalSemaphoreDesc,
    ur_exp_external_semaphore_handle_t *phExternalSemaphore);
ur_result_t urBindlessImagesReleaseExternalSemaphoreExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_external_semaphore_handle_t hExternalSemaphore);
ur_result_t urBindlessImagesWaitExternalSemaphoreExp(
    ur_queue_handle_t hQueue, ur_exp_external_semaphore_handle_t hSemaphore,
    bool hasWaitValue, uint64_t waitValue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent);
ur_result_t urBindlessImagesSignalExternalSemaphoreExp(
    ur_queue_handle_t hQueue, ur_exp_external_semaphore_handle_t hSemaphore,
    bool hasSignalValue, uint64_t signalValue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent);
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
ur_result_t urCommandBufferEnqueueExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_queue_handle_t hQueue,
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
ur_result_t urEnqueueCooperativeKernelLaunchExp(
    ur_queue_handle_t hQueue, ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent);
ur_result_t urKernelSuggestMaxCooperativeGroupCountExp(
    ur_kernel_handle_t hKernel, ur_device_handle_t hDevice, uint32_t workDim,
    const size_t *pLocalWorkSize, size_t dynamicSharedMemorySize,
    uint32_t *pGroupCountRet);
ur_result_t urEnqueueTimestampRecordingExp(
    ur_queue_handle_t hQueue, bool blocking, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent);
ur_result_t urEnqueueKernelLaunchCustomExp(
    ur_queue_handle_t hQueue, ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numPropsInLaunchPropList,
    const ur_exp_launch_property_t *launchPropList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent);
ur_result_t urProgramBuildExp(ur_program_handle_t hProgram, uint32_t numDevices,
                              ur_device_handle_t *phDevices,
                              const char *pOptions);
ur_result_t urProgramCompileExp(ur_program_handle_t hProgram,
                                uint32_t numDevices,
                                ur_device_handle_t *phDevices,
                                const char *pOptions);
ur_result_t urProgramLinkExp(ur_context_handle_t hContext, uint32_t numDevices,
                             ur_device_handle_t *phDevices, uint32_t count,
                             const ur_program_handle_t *phPrograms,
                             const char *pOptions,
                             ur_program_handle_t *phProgram);
ur_result_t urUSMImportExp(ur_context_handle_t hContext, void *pMem,
                           size_t size);
ur_result_t urUSMReleaseExp(ur_context_handle_t hContext, void *pMem);
ur_result_t urUsmP2PEnablePeerAccessExp(ur_device_handle_t commandDevice,
                                        ur_device_handle_t peerDevice);
ur_result_t urUsmP2PDisablePeerAccessExp(ur_device_handle_t commandDevice,
                                         ur_device_handle_t peerDevice);
ur_result_t urUsmP2PPeerAccessGetInfoExp(ur_device_handle_t commandDevice,
                                         ur_device_handle_t peerDevice,
                                         ur_exp_peer_info_t propName,
                                         size_t propSize, void *pPropValue,
                                         size_t *pPropSizeRet);
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
#ifdef UR_STATIC_ADAPTER_LEVEL_ZERO
ur_result_t urAdapterGetDdiTables(ur_dditable_t *ddi);
#endif
} // namespace ur::level_zero
