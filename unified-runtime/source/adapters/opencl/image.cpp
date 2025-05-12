//===---------- image.cpp - OpenCL Adapter ---------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urUSMPitchedAllocExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] const ur_usm_desc_t *pUSMDesc,
    [[maybe_unused]] ur_usm_pool_handle_t pool,
    [[maybe_unused]] size_t widthInBytes, [[maybe_unused]] size_t height,
    [[maybe_unused]] size_t elementSizeBytes, [[maybe_unused]] void **ppMem,
    [[maybe_unused]] size_t *pResultPitch) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesUnsampledImageHandleDestroyExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_exp_image_native_handle_t hImage) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesSampledImageHandleDestroyExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_exp_image_native_handle_t hImage) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImageAllocateExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] const ur_image_format_t *pImageFormat,
    [[maybe_unused]] const ur_image_desc_t *pImageDesc,
    [[maybe_unused]] ur_exp_image_mem_native_handle_t *phImageMem) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImageFreeExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_exp_image_mem_native_handle_t hImageMem) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesUnsampledImageCreateExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_exp_image_mem_native_handle_t hImageMem,
    [[maybe_unused]] const ur_image_format_t *pImageFormat,
    [[maybe_unused]] const ur_image_desc_t *pImageDesc,
    [[maybe_unused]] ur_exp_image_native_handle_t *phImage) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesSampledImageCreateExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_exp_image_mem_native_handle_t hImageMem,
    [[maybe_unused]] const ur_image_format_t *pImageFormat,
    [[maybe_unused]] const ur_image_desc_t *pImageDesc,
    [[maybe_unused]] ur_sampler_handle_t hSampler,
    [[maybe_unused]] ur_exp_image_native_handle_t *phImage) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImageCopyExp(
    [[maybe_unused]] ur_queue_handle_t hQueue,
    [[maybe_unused]] const void *pSrc, [[maybe_unused]] void *pDst,
    [[maybe_unused]] const ur_image_desc_t *pSrcImageDesc,
    [[maybe_unused]] const ur_image_desc_t *pDstImageDesc,
    [[maybe_unused]] const ur_image_format_t *pSrcImageFormat,
    [[maybe_unused]] const ur_image_format_t *pDstImageFormat,
    [[maybe_unused]] ur_exp_image_copy_region_t *pCopyRegion,
    [[maybe_unused]] ur_exp_image_copy_flags_t imageCopyFlags,
    [[maybe_unused]] uint32_t numEventsInWaitList,
    [[maybe_unused]] const ur_event_handle_t *phEventWaitList,
    [[maybe_unused]] ur_event_handle_t *phEvent) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImageGetInfoExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_exp_image_mem_native_handle_t hImageMem,
    [[maybe_unused]] ur_image_info_t propName,
    [[maybe_unused]] void *pPropValue, [[maybe_unused]] size_t *pPropSizeRet) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesGetImageMemoryHandleTypeSupportExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] const ur_image_desc_t *pImageDesc,
    [[maybe_unused]] const ur_image_format_t *pImageFormat,
    [[maybe_unused]] ur_exp_image_mem_type_t imageMemHandleType,
    [[maybe_unused]] ur_bool_t *pSupportedRet) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesGetImageUnsampledHandleSupportExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] const ur_image_desc_t *pImageDesc,
    [[maybe_unused]] const ur_image_format_t *pImageFormat,
    [[maybe_unused]] ur_exp_image_mem_type_t imageMemHandleType,
    [[maybe_unused]] ur_bool_t *pSupportedRet) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesGetImageSampledHandleSupportExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] const ur_image_desc_t *pImageDesc,
    [[maybe_unused]] const ur_image_format_t *pImageFormat,
    [[maybe_unused]] ur_exp_image_mem_type_t imageMemHandleType,
    [[maybe_unused]] ur_bool_t *pSupportedRet) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesMipmapGetLevelExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_exp_image_mem_native_handle_t hImageMem,
    [[maybe_unused]] uint32_t mipmapLevel,
    [[maybe_unused]] ur_exp_image_mem_native_handle_t *phImageMem) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesMipmapFreeExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_exp_image_mem_native_handle_t hMem) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImportExternalMemoryExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice, [[maybe_unused]] size_t size,
    [[maybe_unused]] ur_exp_external_mem_type_t memHandleType,
    [[maybe_unused]] ur_exp_external_mem_desc_t *pExternalMemDesc,
    [[maybe_unused]] ur_exp_external_mem_handle_t *phExternalMem) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesMapExternalArrayExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] const ur_image_format_t *pImageFormat,
    [[maybe_unused]] const ur_image_desc_t *pImageDesc,
    [[maybe_unused]] ur_exp_external_mem_handle_t hExternalMem,
    [[maybe_unused]] ur_exp_image_mem_native_handle_t *phImageMem) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesMapExternalLinearMemoryExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] uint64_t offset, [[maybe_unused]] uint64_t size,
    [[maybe_unused]] ur_exp_external_mem_handle_t hExternalMem,
    [[maybe_unused]] void **phRetMem) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesReleaseExternalMemoryExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_exp_external_mem_handle_t hExternalMem) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesFreeMappedLinearMemoryExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice, [[maybe_unused]] void *pMem) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImportExternalSemaphoreExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_exp_external_semaphore_type_t semHandleType,
    [[maybe_unused]] ur_exp_external_semaphore_desc_t *pExternalSemaphoreDesc,
    [[maybe_unused]] ur_exp_external_semaphore_handle_t
        *phExternalSemaphoreHandle) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesReleaseExternalSemaphoreExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_exp_external_semaphore_handle_t hExternalSemaphore) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesWaitExternalSemaphoreExp(
    [[maybe_unused]] ur_queue_handle_t hQueue,
    [[maybe_unused]] ur_exp_external_semaphore_handle_t hSemaphore,
    [[maybe_unused]] bool hasValue, [[maybe_unused]] uint64_t waitValue,
    [[maybe_unused]] uint32_t numEventsInWaitList,
    [[maybe_unused]] const ur_event_handle_t *phEventWaitList,
    [[maybe_unused]] ur_event_handle_t *phEvent) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesSignalExternalSemaphoreExp(
    [[maybe_unused]] ur_queue_handle_t hQueue,
    [[maybe_unused]] ur_exp_external_semaphore_handle_t hSemaphore,
    [[maybe_unused]] bool hasValue, [[maybe_unused]] uint64_t signalValue,
    [[maybe_unused]] uint32_t numEventsInWaitList,
    [[maybe_unused]] const ur_event_handle_t *phEventWaitList,
    [[maybe_unused]] ur_event_handle_t *phEvent) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
