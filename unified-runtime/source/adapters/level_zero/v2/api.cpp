//===--------- api.cpp - Level Zero Adapter ------------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <mutex>
#include <ur_api.h>
#include <ze_api.h>

#include "../common.hpp"
#include "logger/ur_logger.hpp"

std::mutex ZeCall::GlobalLock;

namespace ur::level_zero {

ur_result_t
urContextSetExtendedDeleter(ur_context_handle_t hContext,
                            ur_context_extended_deleter_t pfnDeleter,
                            void *pUserData) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urKernelSetSpecializationConstants(
    ur_kernel_handle_t hKernel, uint32_t count,
    const ur_specialization_constant_info_t *pSpecConstants) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urEventSetCallback(ur_event_handle_t hEvent,
                               ur_execution_info_t execStatus,
                               ur_event_callback_t pfnNotify, void *pUserData) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urUSMPitchedAllocExp(ur_context_handle_t hContext,
                                 ur_device_handle_t hDevice,
                                 const ur_usm_desc_t *pUSMDesc,
                                 ur_usm_pool_handle_t pool, size_t widthInBytes,
                                 size_t height, size_t elementSizeBytes,
                                 void **ppMem, size_t *pResultPitch) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urBindlessImagesUnsampledImageHandleDestroyExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_native_handle_t hImage) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urBindlessImagesSampledImageHandleDestroyExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_native_handle_t hImage) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urBindlessImagesImageAllocateExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    ur_exp_image_mem_native_handle_t *phImageMem) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t
urBindlessImagesImageFreeExp(ur_context_handle_t hContext,
                             ur_device_handle_t hDevice,
                             ur_exp_image_mem_native_handle_t hImageMem) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urBindlessImagesUnsampledImageCreateExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_mem_native_handle_t hImageMem,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    ur_exp_image_native_handle_t *phImage) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urBindlessImagesSampledImageCreateExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_mem_native_handle_t hImageMem,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    ur_sampler_handle_t hSampler, ur_exp_image_native_handle_t *phImage) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urBindlessImagesImageGetInfoExp(
    ur_context_handle_t hContext, ur_exp_image_mem_native_handle_t hImageMem,
    ur_image_info_t propName, void *pPropValue, size_t *pPropSizeRet) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urBindlessImagesMipmapGetLevelExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_mem_native_handle_t hImageMem, uint32_t mipmapLevel,
    ur_exp_image_mem_native_handle_t *phImageMem) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t
urBindlessImagesMipmapFreeExp(ur_context_handle_t hContext,
                              ur_device_handle_t hDevice,
                              ur_exp_image_mem_native_handle_t hMem) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urBindlessImagesImportExternalMemoryExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, size_t size,
    ur_exp_external_mem_type_t memHandleType,
    ur_exp_external_mem_desc_t *pExternalMemDesc,
    ur_exp_external_mem_handle_t *phExternalMem) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urBindlessImagesMapExternalArrayExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    ur_exp_external_mem_handle_t hExternalMem,
    ur_exp_image_mem_native_handle_t *phImageMem) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urBindlessImagesMapExternalLinearMemoryExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, uint64_t offset,
    uint64_t size, ur_exp_external_mem_handle_t hExternalMem, void **ppRetMem) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urBindlessImagesReleaseExternalMemoryExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_external_mem_handle_t hExternalMem) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urBindlessImagesImportExternalSemaphoreExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_external_semaphore_type_t semHandleType,
    ur_exp_external_semaphore_desc_t *pExternalSemaphoreDesc,
    ur_exp_external_semaphore_handle_t *phExternalSemaphore) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urBindlessImagesReleaseExternalSemaphoreExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_external_semaphore_handle_t hExternalSemaphore) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urCommandBufferUpdateKernelLaunchExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, uint32_t numKernelUpdates,
    const ur_exp_command_buffer_update_kernel_launch_desc_t
        *pUpdateKernelLaunch) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urCommandBufferUpdateSignalEventExp(
    ur_exp_command_buffer_command_handle_t hCommand,
    ur_event_handle_t *phEvent) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urCommandBufferUpdateWaitEventsExp(
    ur_exp_command_buffer_command_handle_t hCommand,
    uint32_t NumEventsInWaitList, const ur_event_handle_t *phEventWaitList) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t UR_APICALL urUSMPoolCreateExp(ur_context_handle_t hContext,
                                          ur_device_handle_t hDevice,
                                          ur_usm_pool_desc_t *PoolDesc,
                                          ur_usm_pool_handle_t *pPool) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t UR_APICALL urUSMPoolDestroyExp(ur_context_handle_t hContext,
                                           ur_device_handle_t hDevice,
                                           ur_usm_pool_handle_t hPool) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t UR_APICALL urUSMPoolSetThresholdExp(ur_context_handle_t hContext,
                                                ur_device_handle_t hDevice,
                                                ur_usm_pool_handle_t hPool,
                                                size_t newThreshold) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t UR_APICALL urUSMPoolGetDefaultDevicePoolExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_usm_pool_handle_t *pPool) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t UR_APICALL urUSMPoolGetInfoExp(ur_usm_pool_handle_t hPool,
                                           ur_usm_pool_info_t propName,
                                           void *pPropValue,
                                           size_t *pPropSizeRet) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t UR_APICALL urUSMPoolGetDevicePoolExp(ur_context_handle_t hContext,
                                                 ur_device_handle_t hDevice,
                                                 ur_usm_pool_handle_t *pPool) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t UR_APICALL urUSMPoolSetDevicePoolExp(ur_context_handle_t hContext,
                                                 ur_device_handle_t hDevice,
                                                 ur_usm_pool_handle_t hPool) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t UR_APICALL urUSMPoolTrimToExp(ur_context_handle_t hContext,
                                          ur_device_handle_t hDevice,
                                          ur_usm_pool_handle_t hPool,
                                          size_t minBytesToKeep) {
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

} // namespace ur::level_zero
