//===--------- image.cpp - NativeCPU Adapter ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "ur/ur.hpp"

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
    [[maybe_unused]] ur_exp_image_handle_t hImage) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesSampledImageHandleDestroyExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_exp_image_handle_t hImage) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImageAllocateExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] const ur_image_format_t *pImageFormat,
    [[maybe_unused]] const ur_image_desc_t *pImageDesc,
    [[maybe_unused]] ur_exp_image_mem_handle_t *phImageMem) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImageFreeExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_exp_image_mem_handle_t hImageMem) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesUnsampledImageCreateExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_exp_image_mem_handle_t hImageMem,
    [[maybe_unused]] const ur_image_format_t *pImageFormat,
    [[maybe_unused]] const ur_image_desc_t *pImageDesc,
    [[maybe_unused]] ur_mem_handle_t *phMem,
    [[maybe_unused]] ur_exp_image_handle_t *phImage) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesSampledImageCreateExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_exp_image_mem_handle_t hImageMem,
    [[maybe_unused]] const ur_image_format_t *pImageFormat,
    [[maybe_unused]] const ur_image_desc_t *pImageDesc,
    [[maybe_unused]] ur_sampler_handle_t hSampler,
    [[maybe_unused]] ur_mem_handle_t *phMem,
    [[maybe_unused]] ur_exp_image_handle_t *phImage) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImageCopyExp(
    [[maybe_unused]] ur_queue_handle_t hQueue, [[maybe_unused]] void *pDst,
    [[maybe_unused]] void *pSrc,
    [[maybe_unused]] const ur_image_format_t *pImageFormat,
    [[maybe_unused]] const ur_image_desc_t *pImageDesc,
    [[maybe_unused]] ur_exp_image_copy_flags_t imageCopyFlags,
    [[maybe_unused]] ur_rect_offset_t srcOffset,
    [[maybe_unused]] ur_rect_offset_t dstOffset,
    [[maybe_unused]] ur_rect_region_t copyExtent,
    [[maybe_unused]] ur_rect_region_t hostExtent,
    [[maybe_unused]] uint32_t numEventsInWaitList,
    [[maybe_unused]] const ur_event_handle_t *phEventWaitList,
    [[maybe_unused]] ur_event_handle_t *phEvent) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImageGetInfoExp(
    [[maybe_unused]] ur_exp_image_mem_handle_t hImageMem,
    [[maybe_unused]] ur_image_info_t propName,
    [[maybe_unused]] void *pPropValue, [[maybe_unused]] size_t *pPropSizeRet) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesMipmapGetLevelExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_exp_image_mem_handle_t hImageMem,
    [[maybe_unused]] uint32_t mipmapLevel,
    [[maybe_unused]] ur_exp_image_mem_handle_t *phImageMem) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesMipmapFreeExp([[maybe_unused]] ur_context_handle_t hContext,
                              [[maybe_unused]] ur_device_handle_t hDevice,
                              [[maybe_unused]] ur_exp_image_mem_handle_t hMem) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImportOpaqueFDExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice, [[maybe_unused]] size_t size,
    [[maybe_unused]] ur_exp_interop_mem_desc_t *pInteropMemDesc,
    [[maybe_unused]] ur_exp_interop_mem_handle_t *phInteropMem) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesMapExternalArrayExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] const ur_image_format_t *pImageFormat,
    [[maybe_unused]] const ur_image_desc_t *pImageDesc,
    [[maybe_unused]] ur_exp_interop_mem_handle_t hInteropMem,
    [[maybe_unused]] ur_exp_image_mem_handle_t *phImageMem) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesReleaseInteropExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_exp_interop_mem_handle_t hInteropMem) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesImportExternalSemaphoreOpaqueFDExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_exp_interop_semaphore_desc_t *pInteropSemaphoreDesc,
    [[maybe_unused]] ur_exp_interop_semaphore_handle_t *phInteropSemaphore) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesDestroyExternalSemaphoreExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_exp_interop_semaphore_handle_t hInteropSemaphore) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesWaitExternalSemaphoreExp(
    [[maybe_unused]] ur_queue_handle_t hQueue,
    [[maybe_unused]] ur_exp_interop_semaphore_handle_t hSemaphore,
    [[maybe_unused]] uint32_t numEventsInWaitList,
    [[maybe_unused]] const ur_event_handle_t *phEventWaitList,
    [[maybe_unused]] ur_event_handle_t *phEvent) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesSignalExternalSemaphoreExp(
    [[maybe_unused]] ur_queue_handle_t hQueue,
    [[maybe_unused]] ur_exp_interop_semaphore_handle_t hSemaphore,
    [[maybe_unused]] uint32_t numEventsInWaitList,
    [[maybe_unused]] const ur_event_handle_t *phEventWaitList,
    [[maybe_unused]] ur_event_handle_t *phEvent) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
