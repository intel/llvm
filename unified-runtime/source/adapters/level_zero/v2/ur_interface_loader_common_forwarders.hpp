//===--------- ur_interface_loader_common_forwarders.hpp - L0 Adapter ----===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Several official UR entry points are implemented once, in the common
// `ur::level_zero` namespace (see common/api.hpp). The generated interface
// loader declares every entry point inside `ur::level_zero::v2`. For the common
// entry points those `ur::level_zero::v2` symbols would otherwise be undefined.

#pragma once

#include <unified-runtime/ur_api.h>

#include "../common/api.hpp"
#include "ur_interface_loader.hpp"

namespace ur::level_zero::v2 {

inline ur_result_t urDeviceGet(::ur_platform_handle_t hPlatform,
                               ::ur_device_type_t DeviceType,
                               uint32_t NumEntries,
                               ::ur_device_handle_t *phDevices,
                               uint32_t *pNumDevices) {
  return ::ur::level_zero::urDeviceGet(hPlatform, DeviceType, NumEntries,
                                       phDevices, pNumDevices);
}
inline ur_result_t urDeviceRetain(::ur_device_handle_t hDevice) {
  return ::ur::level_zero::urDeviceRetain(hDevice);
}
inline ur_result_t urDeviceRelease(::ur_device_handle_t hDevice) {
  return ::ur::level_zero::urDeviceRelease(hDevice);
}
inline ur_result_t
urDevicePartition(::ur_device_handle_t hDevice,
                  const ::ur_device_partition_properties_t *pProperties,
                  uint32_t NumDevices, ::ur_device_handle_t *phSubDevices,
                  uint32_t *pNumDevicesRet) {
  return ::ur::level_zero::urDevicePartition(hDevice, pProperties, NumDevices,
                                             phSubDevices, pNumDevicesRet);
}
inline ur_result_t urDeviceSelectBinary(::ur_device_handle_t hDevice,
                                        const ::ur_device_binary_t *pBinaries,
                                        uint32_t NumBinaries,
                                        uint32_t *pSelectedBinary) {
  return ::ur::level_zero::urDeviceSelectBinary(hDevice, pBinaries, NumBinaries,
                                                pSelectedBinary);
}
inline ur_result_t
urDeviceGetNativeHandle(::ur_device_handle_t hDevice,
                        ::ur_native_handle_t *phNativeDevice) {
  return ::ur::level_zero::urDeviceGetNativeHandle(hDevice, phNativeDevice);
}
inline ur_result_t urDeviceCreateWithNativeHandle(
    ::ur_native_handle_t hNativeDevice, ::ur_adapter_handle_t hAdapter,
    const ::ur_device_native_properties_t *pProperties,
    ::ur_device_handle_t *phDevice) {
  return ::ur::level_zero::urDeviceCreateWithNativeHandle(
      hNativeDevice, hAdapter, pProperties, phDevice);
}
inline ur_result_t urDeviceGetGlobalTimestamps(::ur_device_handle_t hDevice,
                                               uint64_t *pDeviceTimestamp,
                                               uint64_t *pHostTimestamp) {
  return ::ur::level_zero::urDeviceGetGlobalTimestamps(
      hDevice, pDeviceTimestamp, pHostTimestamp);
}
inline ur_result_t urDeviceWaitExp(::ur_device_handle_t hDevice) {
  return ::ur::level_zero::urDeviceWaitExp(hDevice);
}
inline ur_result_t urPlatformGet(::ur_adapter_handle_t hAdapter,
                                 uint32_t NumEntries,
                                 ::ur_platform_handle_t *phPlatforms,
                                 uint32_t *pNumPlatforms) {
  return ::ur::level_zero::urPlatformGet(hAdapter, NumEntries, phPlatforms,
                                         pNumPlatforms);
}
inline ur_result_t urPlatformGetApiVersion(::ur_platform_handle_t hPlatform,
                                           ur_api_version_t *pVersion) {
  return ::ur::level_zero::urPlatformGetApiVersion(hPlatform, pVersion);
}
inline ur_result_t
urPlatformGetNativeHandle(::ur_platform_handle_t hPlatform,
                          ::ur_native_handle_t *phNativePlatform) {
  return ::ur::level_zero::urPlatformGetNativeHandle(hPlatform,
                                                     phNativePlatform);
}
inline ur_result_t urPlatformCreateWithNativeHandle(
    ::ur_native_handle_t hNativePlatform, ::ur_adapter_handle_t hAdapter,
    const ur_platform_native_properties_t *pProperties,
    ::ur_platform_handle_t *phPlatform) {
  return ::ur::level_zero::urPlatformCreateWithNativeHandle(
      hNativePlatform, hAdapter, pProperties, phPlatform);
}
inline ur_result_t urPlatformGetBackendOption(::ur_platform_handle_t hPlatform,
                                              const char *pFrontendOption,
                                              const char **ppPlatformOption) {
  return ::ur::level_zero::urPlatformGetBackendOption(
      hPlatform, pFrontendOption, ppPlatformOption);
}
inline ur_result_t urAdapterRelease(::ur_adapter_handle_t hAdapter) {
  return ::ur::level_zero::urAdapterRelease(hAdapter);
}
inline ur_result_t urAdapterRetain(::ur_adapter_handle_t hAdapter) {
  return ::ur::level_zero::urAdapterRetain(hAdapter);
}
inline ur_result_t urAdapterGetLastError(::ur_adapter_handle_t hAdapter,
                                         const char **ppMessage,
                                         int32_t *pError) {
  return ::ur::level_zero::urAdapterGetLastError(hAdapter, ppMessage, pError);
}
inline ur_result_t urAdapterGetInfo(::ur_adapter_handle_t hAdapter,
                                    ur_adapter_info_t propName, size_t propSize,
                                    void *pPropValue, size_t *pPropSizeRet) {
  return ::ur::level_zero::urAdapterGetInfo(hAdapter, propName, propSize,
                                            pPropValue, pPropSizeRet);
}
inline ur_result_t
urAdapterSetLoggerCallback(::ur_adapter_handle_t hAdapter,
                           ur_logger_callback_t pfnLoggerCallback,
                           void *pUserData, ur_logger_level_t level) {
  return ::ur::level_zero::urAdapterSetLoggerCallback(
      hAdapter, pfnLoggerCallback, pUserData, level);
}
inline ur_result_t
urAdapterSetLoggerCallbackLevel(::ur_adapter_handle_t hAdapter,
                                ur_logger_level_t level) {
  return ::ur::level_zero::urAdapterSetLoggerCallbackLevel(hAdapter, level);
}
inline ur_result_t urSamplerCreate(::ur_context_handle_t Context,
                                   const ur_sampler_desc_t *Props,
                                   ::ur_sampler_handle_t *Sampler) {
  return ::ur::level_zero::urSamplerCreate(Context, Props, Sampler);
}
inline ur_result_t urSamplerRetain(::ur_sampler_handle_t Sampler) {
  return ::ur::level_zero::urSamplerRetain(Sampler);
}
inline ur_result_t urSamplerRelease(::ur_sampler_handle_t Sampler) {
  return ::ur::level_zero::urSamplerRelease(Sampler);
}
inline ur_result_t urSamplerGetInfo(::ur_sampler_handle_t Sampler,
                                    ur_sampler_info_t PropName,
                                    size_t PropValueSize, void *PropValue,
                                    size_t *PropSizeRet) {
  return ::ur::level_zero::urSamplerGetInfo(Sampler, PropName, PropValueSize,
                                            PropValue, PropSizeRet);
}
inline ur_result_t
urSamplerGetNativeHandle(::ur_sampler_handle_t Sampler,
                         ::ur_native_handle_t *NativeSampler) {
  return ::ur::level_zero::urSamplerGetNativeHandle(Sampler, NativeSampler);
}
inline ur_result_t urSamplerCreateWithNativeHandle(
    ::ur_native_handle_t NativeSampler, ::ur_context_handle_t Context,
    const ur_sampler_native_properties_t *Properties,
    ::ur_sampler_handle_t *Sampler) {
  return ::ur::level_zero::urSamplerCreateWithNativeHandle(
      NativeSampler, Context, Properties, Sampler);
}
inline ur_result_t
urProgramCreateWithIL(::ur_context_handle_t hContext, const void *pIL,
                      size_t length, const ur_program_properties_t *pProperties,
                      ::ur_program_handle_t *phProgram) {
  return ::ur::level_zero::urProgramCreateWithIL(hContext, pIL, length,
                                                 pProperties, phProgram);
}
inline ur_result_t
urProgramCreateWithBinary(::ur_context_handle_t hContext, uint32_t numDevices,
                          ::ur_device_handle_t *phDevices, size_t *pLengths,
                          const uint8_t **ppBinaries,
                          const ur_program_properties_t *pProperties,
                          ::ur_program_handle_t *phProgram) {
  return ::ur::level_zero::urProgramCreateWithBinary(
      hContext, numDevices, phDevices, pLengths, ppBinaries, pProperties,
      phProgram);
}
inline ur_result_t urProgramBuild(::ur_context_handle_t hContext,
                                  ::ur_program_handle_t hProgram,
                                  const char *pOptions) {
  return ::ur::level_zero::urProgramBuild(hContext, hProgram, pOptions);
}
inline ur_result_t urProgramCompile(::ur_context_handle_t hContext,
                                    ::ur_program_handle_t hProgram,
                                    const char *pOptions) {
  return ::ur::level_zero::urProgramCompile(hContext, hProgram, pOptions);
}
inline ur_result_t urProgramLink(::ur_context_handle_t hContext, uint32_t count,
                                 const ::ur_program_handle_t *phPrograms,
                                 const char *pOptions,
                                 ::ur_program_handle_t *phProgram) {
  return ::ur::level_zero::urProgramLink(hContext, count, phPrograms, pOptions,
                                         phProgram);
}
inline ur_result_t urProgramRetain(::ur_program_handle_t hProgram) {
  return ::ur::level_zero::urProgramRetain(hProgram);
}
inline ur_result_t urProgramRelease(::ur_program_handle_t hProgram) {
  return ::ur::level_zero::urProgramRelease(hProgram);
}
inline ur_result_t urProgramGetFunctionPointer(::ur_device_handle_t hDevice,
                                               ::ur_program_handle_t hProgram,
                                               const char *pFunctionName,
                                               void **ppFunctionPointer) {
  return ::ur::level_zero::urProgramGetFunctionPointer(
      hDevice, hProgram, pFunctionName, ppFunctionPointer);
}
inline ur_result_t urProgramGetGlobalVariablePointer(
    ::ur_device_handle_t hDevice, ::ur_program_handle_t hProgram,
    const char *pGlobalVariableName, size_t *pGlobalVariableSizeRet,
    void **ppGlobalVariablePointerRet) {
  return ::ur::level_zero::urProgramGetGlobalVariablePointer(
      hDevice, hProgram, pGlobalVariableName, pGlobalVariableSizeRet,
      ppGlobalVariablePointerRet);
}
inline ur_result_t urProgramGetInfo(::ur_program_handle_t hProgram,
                                    ur_program_info_t propName, size_t propSize,
                                    void *pPropValue, size_t *pPropSizeRet) {
  return ::ur::level_zero::urProgramGetInfo(hProgram, propName, propSize,
                                            pPropValue, pPropSizeRet);
}
inline ur_result_t urProgramGetBuildInfo(::ur_program_handle_t hProgram,
                                         ::ur_device_handle_t hDevice,
                                         ur_program_build_info_t propName,
                                         size_t propSize, void *pPropValue,
                                         size_t *pPropSizeRet) {
  return ::ur::level_zero::urProgramGetBuildInfo(
      hProgram, hDevice, propName, propSize, pPropValue, pPropSizeRet);
}
inline ur_result_t urProgramSetSpecializationConstants(
    ::ur_program_handle_t hProgram, uint32_t count,
    const ur_specialization_constant_info_t *pSpecConstants) {
  return ::ur::level_zero::urProgramSetSpecializationConstants(hProgram, count,
                                                               pSpecConstants);
}
inline ur_result_t
urProgramGetNativeHandle(::ur_program_handle_t hProgram,
                         ::ur_native_handle_t *phNativeProgram) {
  return ::ur::level_zero::urProgramGetNativeHandle(hProgram, phNativeProgram);
}
inline ur_result_t urProgramCreateWithNativeHandle(
    ::ur_native_handle_t hNativeProgram, ::ur_context_handle_t hContext,
    const ur_program_native_properties_t *pProperties,
    ::ur_program_handle_t *phProgram) {
  return ::ur::level_zero::urProgramCreateWithNativeHandle(
      hNativeProgram, hContext, pProperties, phProgram);
}
inline ur_result_t
urProgramDynamicLinkExp(::ur_context_handle_t hContext, uint32_t count,
                        const ::ur_program_handle_t *phPrograms) {
  return ::ur::level_zero::urProgramDynamicLinkExp(hContext, count, phPrograms);
}
inline ur_result_t urProgramBuildExp(::ur_program_handle_t hProgram,
                                     uint32_t numDevices,
                                     ::ur_device_handle_t *phDevices,
                                     ur_exp_program_flags_t flags,
                                     const char *pOptions) {
  return ::ur::level_zero::urProgramBuildExp(hProgram, numDevices, phDevices,
                                             flags, pOptions);
}
inline ur_result_t urProgramCompileExp(::ur_program_handle_t hProgram,
                                       uint32_t numDevices,
                                       ::ur_device_handle_t *phDevices,
                                       ur_exp_program_flags_t flags,
                                       const char *pOptions) {
  return ::ur::level_zero::urProgramCompileExp(hProgram, numDevices, phDevices,
                                               flags, pOptions);
}
inline ur_result_t
urProgramLinkExp(::ur_context_handle_t hContext, uint32_t numDevices,
                 ::ur_device_handle_t *phDevices, ur_exp_program_flags_t flags,
                 uint32_t count, const ::ur_program_handle_t *phPrograms,
                 const char *pOptions, ::ur_program_handle_t *phProgram) {
  return ::ur::level_zero::urProgramLinkExp(hContext, numDevices, phDevices,
                                            flags, count, phPrograms, pOptions,
                                            phProgram);
}
inline ur_result_t urVirtualMemGranularityGetInfo(
    ::ur_context_handle_t hContext, ::ur_device_handle_t hDevice,
    size_t allocationSize, ur_virtual_mem_granularity_info_t propName,
    size_t propSize, void *pPropValue, size_t *pPropSizeRet) {
  return ::ur::level_zero::urVirtualMemGranularityGetInfo(
      hContext, hDevice, allocationSize, propName, propSize, pPropValue,
      pPropSizeRet);
}
inline ur_result_t urVirtualMemReserve(::ur_context_handle_t hContext,
                                       const void *pStart, size_t size,
                                       void **ppStart) {
  return ::ur::level_zero::urVirtualMemReserve(hContext, pStart, size, ppStart);
}
inline ur_result_t urVirtualMemFree(::ur_context_handle_t hContext,
                                    const void *pStart, size_t size) {
  return ::ur::level_zero::urVirtualMemFree(hContext, pStart, size);
}
inline ur_result_t urVirtualMemMap(::ur_context_handle_t hContext,
                                   const void *pStart, size_t size,
                                   ::ur_physical_mem_handle_t hPhysicalMem,
                                   size_t offset,
                                   ur_virtual_mem_access_flags_t flags) {
  return ::ur::level_zero::urVirtualMemMap(hContext, pStart, size, hPhysicalMem,
                                           offset, flags);
}
inline ur_result_t urVirtualMemUnmap(::ur_context_handle_t hContext,
                                     const void *pStart, size_t size) {
  return ::ur::level_zero::urVirtualMemUnmap(hContext, pStart, size);
}
inline ur_result_t urVirtualMemSetAccess(::ur_context_handle_t hContext,
                                         const void *pStart, size_t size,
                                         ur_virtual_mem_access_flags_t flags) {
  return ::ur::level_zero::urVirtualMemSetAccess(hContext, pStart, size, flags);
}
inline ur_result_t urVirtualMemGetInfo(::ur_context_handle_t hContext,
                                       const void *pStart, size_t size,
                                       ur_virtual_mem_info_t propName,
                                       size_t propSize, void *pPropValue,
                                       size_t *pPropSizeRet) {
  return ::ur::level_zero::urVirtualMemGetInfo(
      hContext, pStart, size, propName, propSize, pPropValue, pPropSizeRet);
}
inline ur_result_t urMemoryExportAllocExportableMemoryExp(
    ::ur_context_handle_t hContext, ::ur_device_handle_t hDevice,
    size_t alignment, size_t size,
    ur_exp_external_mem_type_t handleTypeToExport, void **ppMem) {
  return ::ur::level_zero::urMemoryExportAllocExportableMemoryExp(
      hContext, hDevice, alignment, size, handleTypeToExport, ppMem);
}
inline ur_result_t urMemoryExportFreeExportableMemoryExp(
    ::ur_context_handle_t hContext, ::ur_device_handle_t hDevice, void *pMem) {
  return ::ur::level_zero::urMemoryExportFreeExportableMemoryExp(hContext,
                                                                 hDevice, pMem);
}
inline ur_result_t urMemoryExportExportMemoryHandleExp(
    ::ur_context_handle_t hContext, ::ur_device_handle_t hDevice,
    ur_exp_external_mem_type_t handleTypeToExport, void *pMem,
    void *pMemHandleRet) {
  return ::ur::level_zero::urMemoryExportExportMemoryHandleExp(
      hContext, hDevice, handleTypeToExport, pMem, pMemHandleRet);
}
inline ur_result_t urUSMPitchedAllocExp(::ur_context_handle_t hContext,
                                        ::ur_device_handle_t hDevice,
                                        const ur_usm_desc_t *pUSMDesc,
                                        ::ur_usm_pool_handle_t pool,
                                        size_t widthInBytes, size_t height,
                                        size_t elementSizeBytes, void **ppMem,
                                        size_t *pResultPitch) {
  return ::ur::level_zero::urUSMPitchedAllocExp(
      hContext, hDevice, pUSMDesc, pool, widthInBytes, height, elementSizeBytes,
      ppMem, pResultPitch);
}
inline ur_result_t urBindlessImagesUnsampledImageHandleDestroyExp(
    ::ur_context_handle_t hContext, ::ur_device_handle_t hDevice,
    ur_exp_image_native_handle_t hImage) {
  return ::ur::level_zero::urBindlessImagesUnsampledImageHandleDestroyExp(
      hContext, hDevice, hImage);
}
inline ur_result_t urBindlessImagesSampledImageHandleDestroyExp(
    ::ur_context_handle_t hContext, ::ur_device_handle_t hDevice,
    ur_exp_image_native_handle_t hImage) {
  return ::ur::level_zero::urBindlessImagesSampledImageHandleDestroyExp(
      hContext, hDevice, hImage);
}
inline ur_result_t urBindlessImagesImageAllocateExp(
    ::ur_context_handle_t hContext, ::ur_device_handle_t hDevice,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    ur_exp_image_mem_native_handle_t *phImageMem) {
  return ::ur::level_zero::urBindlessImagesImageAllocateExp(
      hContext, hDevice, pImageFormat, pImageDesc, phImageMem);
}
inline ur_result_t
urBindlessImagesImageFreeExp(::ur_context_handle_t hContext,
                             ::ur_device_handle_t hDevice,
                             ur_exp_image_mem_native_handle_t hImageMem) {
  return ::ur::level_zero::urBindlessImagesImageFreeExp(hContext, hDevice,
                                                        hImageMem);
}
inline ur_result_t urBindlessImagesUnsampledImageCreateExp(
    ::ur_context_handle_t hContext, ::ur_device_handle_t hDevice,
    ur_exp_image_mem_native_handle_t hImageMem,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    ur_exp_image_native_handle_t *phImage) {
  return ::ur::level_zero::urBindlessImagesUnsampledImageCreateExp(
      hContext, hDevice, hImageMem, pImageFormat, pImageDesc, phImage);
}
inline ur_result_t urBindlessImagesSampledImageCreateExp(
    ::ur_context_handle_t hContext, ::ur_device_handle_t hDevice,
    ur_exp_image_mem_native_handle_t hImageMem,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    const ur_sampler_desc_t *pSamplerDesc,
    ur_exp_image_native_handle_t *phImage) {
  return ::ur::level_zero::urBindlessImagesSampledImageCreateExp(
      hContext, hDevice, hImageMem, pImageFormat, pImageDesc, pSamplerDesc,
      phImage);
}
inline ur_result_t urBindlessImagesImageGetInfoExp(
    ::ur_context_handle_t hContext, ur_exp_image_mem_native_handle_t hImageMem,
    ur_image_info_t propName, void *pPropValue, size_t *pPropSizeRet) {
  return ::ur::level_zero::urBindlessImagesImageGetInfoExp(
      hContext, hImageMem, propName, pPropValue, pPropSizeRet);
}
inline ur_result_t urBindlessImagesGetImageMemoryHandleTypeSupportExp(
    ::ur_context_handle_t hContext, ::ur_device_handle_t hDevice,
    const ur_image_desc_t *pImageDesc, const ur_image_format_t *pImageFormat,
    ur_exp_image_mem_type_t imageMemHandleType, ur_bool_t *pSupportedRet) {
  return ::ur::level_zero::urBindlessImagesGetImageMemoryHandleTypeSupportExp(
      hContext, hDevice, pImageDesc, pImageFormat, imageMemHandleType,
      pSupportedRet);
}
inline ur_result_t urBindlessImagesGetImageUnsampledHandleSupportExp(
    ::ur_context_handle_t hContext, ::ur_device_handle_t hDevice,
    const ur_image_desc_t *pImageDesc, const ur_image_format_t *pImageFormat,
    ur_exp_image_mem_type_t imageMemHandleType, ur_bool_t *pSupportedRet) {
  return ::ur::level_zero::urBindlessImagesGetImageUnsampledHandleSupportExp(
      hContext, hDevice, pImageDesc, pImageFormat, imageMemHandleType,
      pSupportedRet);
}
inline ur_result_t urBindlessImagesGetImageSampledHandleSupportExp(
    ::ur_context_handle_t hContext, ::ur_device_handle_t hDevice,
    const ur_image_desc_t *pImageDesc, const ur_image_format_t *pImageFormat,
    ur_exp_image_mem_type_t imageMemHandleType, ur_bool_t *pSupportedRet) {
  return ::ur::level_zero::urBindlessImagesGetImageSampledHandleSupportExp(
      hContext, hDevice, pImageDesc, pImageFormat, imageMemHandleType,
      pSupportedRet);
}
inline ur_result_t urBindlessImagesMipmapGetLevelExp(
    ::ur_context_handle_t hContext, ::ur_device_handle_t hDevice,
    ur_exp_image_mem_native_handle_t hImageMem, uint32_t mipmapLevel,
    ur_exp_image_mem_native_handle_t *phImageMem) {
  return ::ur::level_zero::urBindlessImagesMipmapGetLevelExp(
      hContext, hDevice, hImageMem, mipmapLevel, phImageMem);
}
inline ur_result_t urBindlessImagesImportExternalMemoryExp(
    ::ur_context_handle_t hContext, ::ur_device_handle_t hDevice, size_t size,
    ur_exp_external_mem_type_t memHandleType,
    ur_exp_external_mem_desc_t *pExternalMemDesc,
    ::ur_exp_external_mem_handle_t *phExternalMem) {
  return ::ur::level_zero::urBindlessImagesImportExternalMemoryExp(
      hContext, hDevice, size, memHandleType, pExternalMemDesc, phExternalMem);
}
inline ur_result_t urBindlessImagesMapExternalArrayExp(
    ::ur_context_handle_t hContext, ::ur_device_handle_t hDevice,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    ::ur_exp_external_mem_handle_t hExternalMem,
    ur_exp_image_mem_native_handle_t *phImageMem) {
  return ::ur::level_zero::urBindlessImagesMapExternalArrayExp(
      hContext, hDevice, pImageFormat, pImageDesc, hExternalMem, phImageMem);
}
inline ur_result_t urBindlessImagesMapExternalLinearMemoryExp(
    ::ur_context_handle_t hContext, ::ur_device_handle_t hDevice,
    uint64_t offset, uint64_t size, ::ur_exp_external_mem_handle_t hExternalMem,
    void **ppRetMem) {
  return ::ur::level_zero::urBindlessImagesMapExternalLinearMemoryExp(
      hContext, hDevice, offset, size, hExternalMem, ppRetMem);
}
inline ur_result_t urBindlessImagesReleaseExternalMemoryExp(
    ::ur_context_handle_t hContext, ::ur_device_handle_t hDevice,
    ::ur_exp_external_mem_handle_t hExternalMem) {
  return ::ur::level_zero::urBindlessImagesReleaseExternalMemoryExp(
      hContext, hDevice, hExternalMem);
}
inline ur_result_t urBindlessImagesFreeMappedLinearMemoryExp(
    ::ur_context_handle_t hContext, ::ur_device_handle_t hDevice, void *pMem) {
  return ::ur::level_zero::urBindlessImagesFreeMappedLinearMemoryExp(
      hContext, hDevice, pMem);
}
inline ur_result_t urBindlessImagesSupportsImportingHandleTypeExp(
    ::ur_device_handle_t hDevice, ur_exp_external_mem_type_t memHandleType,
    ur_bool_t *pSupportedRet) {
  return ::ur::level_zero::urBindlessImagesSupportsImportingHandleTypeExp(
      hDevice, memHandleType, pSupportedRet);
}
inline ur_result_t urBindlessImagesImportExternalSemaphoreExp(
    ::ur_context_handle_t hContext, ::ur_device_handle_t hDevice,
    ur_exp_external_semaphore_type_t semHandleType,
    ur_exp_external_semaphore_desc_t *pExternalSemaphoreDesc,
    ::ur_exp_external_semaphore_handle_t *phExternalSemaphore) {
  return ::ur::level_zero::urBindlessImagesImportExternalSemaphoreExp(
      hContext, hDevice, semHandleType, pExternalSemaphoreDesc,
      phExternalSemaphore);
}
inline ur_result_t urBindlessImagesReleaseExternalSemaphoreExp(
    ::ur_context_handle_t hContext, ::ur_device_handle_t hDevice,
    ::ur_exp_external_semaphore_handle_t hExternalSemaphore) {
  return ::ur::level_zero::urBindlessImagesReleaseExternalSemaphoreExp(
      hContext, hDevice, hExternalSemaphore);
}
inline ur_result_t
urBindlessImagesMipmapFreeExp(::ur_context_handle_t hContext,
                              ::ur_device_handle_t hDevice,
                              ur_exp_image_mem_native_handle_t hMem) {
  return ::ur::level_zero::urBindlessImagesMipmapFreeExp(hContext, hDevice,
                                                         hMem);
}

} // namespace ur::level_zero::v2
