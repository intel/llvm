//===--------- api.hpp - Level Zero Adapter ------------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <unified-runtime/ur_api.h>

// Declarations of every official UR entry point implemented in the common.

namespace ur::level_zero {

// Device
ur_result_t urDeviceGet(::ur_platform_handle_t hPlatformOpque,
                        ::ur_device_type_t DeviceType, uint32_t NumEntries,
                        ::ur_device_handle_t *phDevicesOpque,
                        uint32_t *pNumDevices);
ur_result_t urDeviceGetInfo(::ur_device_handle_t hDeviceOpque,
                            ::ur_device_info_t propName, size_t propSize,
                            void *pPropValue, size_t *pPropSizeRet);
ur_result_t urDeviceRetain(::ur_device_handle_t hDeviceOpque);
ur_result_t urDeviceRelease(::ur_device_handle_t hDeviceOpque);
ur_result_t
urDevicePartition(::ur_device_handle_t hDeviceOpque,
                  const ::ur_device_partition_properties_t *pProperties,
                  uint32_t NumDevices, ::ur_device_handle_t *phSubDevicesOpque,
                  uint32_t *pNumDevicesRet);
ur_result_t urDeviceSelectBinary(::ur_device_handle_t hDeviceOpque,
                                 const ::ur_device_binary_t *pBinaries,
                                 uint32_t NumBinaries,
                                 uint32_t *pSelectedBinary);
ur_result_t urDeviceGetNativeHandle(::ur_device_handle_t hDeviceOpque,
                                    ::ur_native_handle_t *phNativeDevice);
ur_result_t urDeviceCreateWithNativeHandle(
    ::ur_native_handle_t hNativeDevice, ::ur_adapter_handle_t hAdapterOpque,
    const ::ur_device_native_properties_t *pProperties,
    ::ur_device_handle_t *phDeviceOpque);
ur_result_t urDeviceGetGlobalTimestamps(::ur_device_handle_t hDeviceOpque,
                                        uint64_t *pDeviceTimestamp,
                                        uint64_t *pHostTimestamp);
ur_result_t urDeviceWaitExp(::ur_device_handle_t hDeviceOpque);

// Platform
ur_result_t urPlatformGet(::ur_adapter_handle_t hAdapterOpque,
                          uint32_t NumEntries,
                          ::ur_platform_handle_t *phPlatformsOpque,
                          uint32_t *pNumPlatforms);
ur_result_t urPlatformGetInfo(::ur_platform_handle_t hPlatformOpque,
                              ur_platform_info_t propName, size_t propSize,
                              void *pPropValue, size_t *pPropSizeRet);
ur_result_t urPlatformGetApiVersion(::ur_platform_handle_t hPlatformOpque,
                                    ur_api_version_t *pVersion);
ur_result_t urPlatformGetNativeHandle(::ur_platform_handle_t hPlatformOpque,
                                      ::ur_native_handle_t *phNativePlatform);
ur_result_t urPlatformCreateWithNativeHandle(
    ::ur_native_handle_t hNativePlatform, ::ur_adapter_handle_t hAdapterOpque,
    const ur_platform_native_properties_t *pProperties,
    ::ur_platform_handle_t *phPlatformOpque);
ur_result_t urPlatformGetBackendOption(::ur_platform_handle_t hPlatformOpque,
                                       const char *pFrontendOption,
                                       const char **ppPlatformOption);

// Adapter
ur_result_t urAdapterRelease(::ur_adapter_handle_t hAdapterOpque);
ur_result_t urAdapterRetain(::ur_adapter_handle_t hAdapterOpque);
ur_result_t urAdapterGetLastError(::ur_adapter_handle_t hAdapterOpque,
                                  const char **ppMessage, int32_t *pError);
ur_result_t urAdapterGetInfo(::ur_adapter_handle_t hAdapterOpque,
                             ur_adapter_info_t propName, size_t propSize,
                             void *pPropValue, size_t *pPropSizeRet);
ur_result_t urAdapterSetLoggerCallback(::ur_adapter_handle_t hAdapterOpque,
                                       ur_logger_callback_t pfnLoggerCallback,
                                       void *pUserData,
                                       ur_logger_level_t level);
ur_result_t urAdapterSetLoggerCallbackLevel(::ur_adapter_handle_t hAdapterOpque,
                                            ur_logger_level_t level);

// Sampler
ur_result_t urSamplerCreate(::ur_context_handle_t ContextOpque,
                            const ur_sampler_desc_t *Props,
                            ::ur_sampler_handle_t *SamplerOpque);
ur_result_t urSamplerRetain(::ur_sampler_handle_t SamplerOpque);
ur_result_t urSamplerRelease(::ur_sampler_handle_t SamplerOpque);
ur_result_t urSamplerGetInfo(::ur_sampler_handle_t SamplerOpque,
                             ur_sampler_info_t PropName, size_t PropValueSize,
                             void *PropValue, size_t *PropSizeRet);
ur_result_t urSamplerGetNativeHandle(::ur_sampler_handle_t SamplerOpque,
                                     ::ur_native_handle_t *NativeSampler);
ur_result_t urSamplerCreateWithNativeHandle(
    ::ur_native_handle_t NativeSampler, ::ur_context_handle_t ContextOpque,
    const ur_sampler_native_properties_t *Properties,
    ::ur_sampler_handle_t *SamplerOpque);

// Program
ur_result_t urProgramCreateWithIL(::ur_context_handle_t hContextOpque,
                                  const void *pIL, size_t length,
                                  const ur_program_properties_t *pProperties,
                                  ::ur_program_handle_t *phProgramOpque);
ur_result_t urProgramCreateWithBinary(
    ::ur_context_handle_t hContextOpque, uint32_t numDevices,
    ::ur_device_handle_t *phDevicesOpque, size_t *pLengths,
    const uint8_t **ppBinaries, const ur_program_properties_t *pProperties,
    ::ur_program_handle_t *phProgramOpque);
ur_result_t urProgramBuild(::ur_context_handle_t hContextOpque,
                           ::ur_program_handle_t hProgramOpque,
                           const char *pOptions);
ur_result_t urProgramCompile(::ur_context_handle_t hContextOpque,
                             ::ur_program_handle_t hProgramOpque,
                             const char *pOptions);
ur_result_t urProgramLink(::ur_context_handle_t hContextOpque, uint32_t count,
                          const ::ur_program_handle_t *phProgramsOpque,
                          const char *pOptions,
                          ::ur_program_handle_t *phProgramOpque);
ur_result_t urProgramRetain(::ur_program_handle_t hProgramOpque);
ur_result_t urProgramRelease(::ur_program_handle_t hProgramOpque);
ur_result_t urProgramGetFunctionPointer(::ur_device_handle_t hDeviceOpque,
                                        ::ur_program_handle_t hProgramOpque,
                                        const char *pFunctionName,
                                        void **ppFunctionPointer);
ur_result_t urProgramGetGlobalVariablePointer(
    ::ur_device_handle_t hDeviceOpque, ::ur_program_handle_t hProgramOpque,
    const char *pGlobalVariableName, size_t *pGlobalVariableSizeRet,
    void **ppGlobalVariablePointerRet);
ur_result_t urProgramGetInfo(::ur_program_handle_t hProgramOpque,
                             ur_program_info_t propName, size_t propSize,
                             void *pPropValue, size_t *pPropSizeRet);
ur_result_t urProgramGetBuildInfo(::ur_program_handle_t hProgramOpque,
                                  ::ur_device_handle_t hDeviceOpque,
                                  ur_program_build_info_t propName,
                                  size_t propSize, void *pPropValue,
                                  size_t *pPropSizeRet);
ur_result_t urProgramSetSpecializationConstants(
    ::ur_program_handle_t hProgramOpque, uint32_t count,
    const ur_specialization_constant_info_t *pSpecConstants);
ur_result_t urProgramGetNativeHandle(::ur_program_handle_t hProgramOpque,
                                     ::ur_native_handle_t *phNativeProgram);
ur_result_t urProgramCreateWithNativeHandle(
    ::ur_native_handle_t hNativeProgram, ::ur_context_handle_t hContextOpque,
    const ur_program_native_properties_t *pProperties,
    ::ur_program_handle_t *phProgramOpque);
ur_result_t
urProgramDynamicLinkExp(::ur_context_handle_t hContextOpque, uint32_t count,
                        const ::ur_program_handle_t *phProgramsOpque);
ur_result_t urProgramBuildExp(::ur_program_handle_t hProgramOpque,
                              uint32_t numDevices,
                              ::ur_device_handle_t *phDevicesOpque,
                              ur_exp_program_flags_t flags,
                              const char *pOptions);
ur_result_t urProgramCompileExp(::ur_program_handle_t hProgramOpque,
                                uint32_t numDevices,
                                ::ur_device_handle_t *phDevicesOpque,
                                ur_exp_program_flags_t flags,
                                const char *pOptions);
ur_result_t urProgramLinkExp(::ur_context_handle_t hContextOpque,
                             uint32_t numDevices,
                             ::ur_device_handle_t *phDevicesOpque,
                             ur_exp_program_flags_t flags, uint32_t count,
                             const ::ur_program_handle_t *phProgramsOpque,
                             const char *pOptions,
                             ::ur_program_handle_t *phProgramOpque);

// Virtual memory
ur_result_t urVirtualMemGranularityGetInfo(
    ::ur_context_handle_t hContextOpque, ::ur_device_handle_t hDeviceOpque,
    size_t allocationSize, ur_virtual_mem_granularity_info_t propName,
    size_t propSize, void *pPropValue, size_t *pPropSizeRet);
ur_result_t urVirtualMemReserve(::ur_context_handle_t hContextOpque,
                                const void *pStart, size_t size,
                                void **ppStart);
ur_result_t urVirtualMemFree(::ur_context_handle_t hContextOpque,
                             const void *pStart, size_t size);
ur_result_t urVirtualMemMap(::ur_context_handle_t hContextOpque,
                            const void *pStart, size_t size,
                            ::ur_physical_mem_handle_t hPhysicalMemOpque,
                            size_t offset, ur_virtual_mem_access_flags_t flags);
ur_result_t urVirtualMemUnmap(::ur_context_handle_t hContextOpque,
                              const void *pStart, size_t size);
ur_result_t urVirtualMemSetAccess(::ur_context_handle_t hContextOpque,
                                  const void *pStart, size_t size,
                                  ur_virtual_mem_access_flags_t flags);
ur_result_t urVirtualMemGetInfo(::ur_context_handle_t hContextOpque,
                                const void *pStart, size_t size,
                                ur_virtual_mem_info_t propName, size_t propSize,
                                void *pPropValue, size_t *pPropSizeRet);

// Memory export
ur_result_t urMemoryExportAllocExportableMemoryExp(
    ::ur_context_handle_t hContextOpque, ::ur_device_handle_t hDeviceOpque,
    size_t alignment, size_t size,
    ur_exp_external_mem_type_t handleTypeToExport, void **ppMem);
ur_result_t
urMemoryExportFreeExportableMemoryExp(::ur_context_handle_t hContextOpque,
                                      ::ur_device_handle_t hDeviceOpque,
                                      void *pMem);
ur_result_t urMemoryExportExportMemoryHandleExp(
    ::ur_context_handle_t hContextOpque, ::ur_device_handle_t hDeviceOpque,
    ur_exp_external_mem_type_t handleTypeToExport, void *pMem,
    void *pMemHandleRet);

// Bindless images / pitched USM
ur_result_t urUSMPitchedAllocExp(::ur_context_handle_t hContextOpque,
                                 ::ur_device_handle_t hDeviceOpque,
                                 const ur_usm_desc_t *pUSMDesc,
                                 ::ur_usm_pool_handle_t poolOpque,
                                 size_t widthInBytes, size_t height,
                                 size_t elementSizeBytes, void **ppMem,
                                 size_t *pResultPitch);
ur_result_t urBindlessImagesUnsampledImageHandleDestroyExp(
    ::ur_context_handle_t hContextOpque, ::ur_device_handle_t hDeviceOpque,
    ur_exp_image_native_handle_t hImage);
ur_result_t urBindlessImagesSampledImageHandleDestroyExp(
    ::ur_context_handle_t hContextOpque, ::ur_device_handle_t hDeviceOpque,
    ur_exp_image_native_handle_t hImage);
ur_result_t urBindlessImagesImageAllocateExp(
    ::ur_context_handle_t hContextOpque, ::ur_device_handle_t hDeviceOpque,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    ur_exp_image_mem_native_handle_t *phImageMem);
ur_result_t
urBindlessImagesImageFreeExp(::ur_context_handle_t hContextOpque,
                             ::ur_device_handle_t hDeviceOpque,
                             ur_exp_image_mem_native_handle_t hImageMem);
ur_result_t urBindlessImagesUnsampledImageCreateExp(
    ::ur_context_handle_t hContextOpque, ::ur_device_handle_t hDeviceOpque,
    ur_exp_image_mem_native_handle_t hImageMem,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    ur_exp_image_native_handle_t *phImage);
ur_result_t urBindlessImagesSampledImageCreateExp(
    ::ur_context_handle_t hContextOpque, ::ur_device_handle_t hDeviceOpque,
    ur_exp_image_mem_native_handle_t hImageMem,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    const ur_sampler_desc_t *pSamplerDesc,
    ur_exp_image_native_handle_t *phImage);
ur_result_t
urBindlessImagesImageGetInfoExp(::ur_context_handle_t hContextOpque,
                                ur_exp_image_mem_native_handle_t hImageMem,
                                ur_image_info_t propName, void *pPropValue,
                                size_t *pPropSizeRet);
ur_result_t urBindlessImagesGetImageMemoryHandleTypeSupportExp(
    ::ur_context_handle_t hContextOpque, ::ur_device_handle_t hDeviceOpque,
    const ur_image_desc_t *pImageDesc, const ur_image_format_t *pImageFormat,
    ur_exp_image_mem_type_t imageMemHandleType, ur_bool_t *pSupportedRet);
ur_result_t urBindlessImagesGetImageUnsampledHandleSupportExp(
    ::ur_context_handle_t hContextOpque, ::ur_device_handle_t hDeviceOpque,
    const ur_image_desc_t *pImageDesc, const ur_image_format_t *pImageFormat,
    ur_exp_image_mem_type_t imageMemHandleType, ur_bool_t *pSupportedRet);
ur_result_t urBindlessImagesGetImageSampledHandleSupportExp(
    ::ur_context_handle_t hContextOpque, ::ur_device_handle_t hDeviceOpque,
    const ur_image_desc_t *pImageDesc, const ur_image_format_t *pImageFormat,
    ur_exp_image_mem_type_t imageMemHandleType, ur_bool_t *pSupportedRet);
ur_result_t urBindlessImagesMipmapGetLevelExp(
    ::ur_context_handle_t hContextOpque, ::ur_device_handle_t hDeviceOpque,
    ur_exp_image_mem_native_handle_t hImageMem, uint32_t mipmapLevel,
    ur_exp_image_mem_native_handle_t *phImageMem);
ur_result_t
urBindlessImagesMipmapFreeExp(::ur_context_handle_t hContextOpque,
                              ::ur_device_handle_t hDeviceOpque,
                              ur_exp_image_mem_native_handle_t hMem);
ur_result_t urBindlessImagesImportExternalMemoryExp(
    ::ur_context_handle_t hContextOpque, ::ur_device_handle_t hDeviceOpque,
    size_t size, ur_exp_external_mem_type_t memHandleType,
    ur_exp_external_mem_desc_t *pExternalMemDesc,
    ::ur_exp_external_mem_handle_t *phExternalMemOpque);
ur_result_t urBindlessImagesMapExternalArrayExp(
    ::ur_context_handle_t hContextOpque, ::ur_device_handle_t hDeviceOpque,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    ::ur_exp_external_mem_handle_t hExternalMemOpque,
    ur_exp_image_mem_native_handle_t *phImageMem);
ur_result_t urBindlessImagesMapExternalLinearMemoryExp(
    ::ur_context_handle_t hContextOpque, ::ur_device_handle_t hDeviceOpque,
    uint64_t offset, uint64_t size,
    ::ur_exp_external_mem_handle_t hExternalMemOpque, void **ppRetMem);
ur_result_t urBindlessImagesReleaseExternalMemoryExp(
    ::ur_context_handle_t hContextOpque, ::ur_device_handle_t hDeviceOpque,
    ::ur_exp_external_mem_handle_t hExternalMemOpque);
ur_result_t
urBindlessImagesFreeMappedLinearMemoryExp(::ur_context_handle_t hContextOpque,
                                          ::ur_device_handle_t hDeviceOpque,
                                          void *pMem);
ur_result_t urBindlessImagesSupportsImportingHandleTypeExp(
    ::ur_device_handle_t hDeviceOpque, ur_exp_external_mem_type_t memHandleType,
    ur_bool_t *pSupportedRet);
ur_result_t urBindlessImagesImportExternalSemaphoreExp(
    ::ur_context_handle_t hContextOpque, ::ur_device_handle_t hDeviceOpque,
    ur_exp_external_semaphore_type_t semHandleType,
    ur_exp_external_semaphore_desc_t *pExternalSemaphoreDesc,
    ::ur_exp_external_semaphore_handle_t *phExternalSemaphoreOpque);
ur_result_t urBindlessImagesReleaseExternalSemaphoreExp(
    ::ur_context_handle_t hContextOpque, ::ur_device_handle_t hDeviceOpque,
    ::ur_exp_external_semaphore_handle_t hExternalSemaphoreOpque);

} // namespace ur::level_zero
