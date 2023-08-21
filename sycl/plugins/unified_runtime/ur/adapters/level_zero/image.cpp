//===--------- image.cpp - Level Zero Adapter ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "image.hpp"
#include "common.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urUSMPitchedAllocExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_usm_desc_t *pUSMDesc, ur_usm_pool_handle_t pool,
    size_t widthInBytes, size_t height, size_t elementSizeBytes, void **ppMem,
    size_t *pResultPitch) {
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = pUSMDesc;
  std::ignore = pool;
  std::ignore = widthInBytes;
  std::ignore = height;
  std::ignore = elementSizeBytes;
  std::ignore = ppMem;
  std::ignore = pResultPitch;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesUnsampledImageHandleDestroyExp(ur_context_handle_t hContext,
                                               ur_device_handle_t hDevice,
                                               ur_exp_image_handle_t hImage) {
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = hImage;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesSampledImageHandleDestroyExp(ur_context_handle_t hContext,
                                             ur_device_handle_t hDevice,
                                             ur_exp_image_handle_t hImage) {
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = hImage;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImageAllocateExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    ur_exp_image_mem_handle_t *phImageMem) {
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = pImageFormat;
  std::ignore = pImageDesc;
  std::ignore = phImageMem;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImageFreeExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_mem_handle_t hImageMem) {
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = hImageMem;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesUnsampledImageCreateExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_mem_handle_t hImageMem, const ur_image_format_t *pImageFormat,
    const ur_image_desc_t *pImageDesc, ur_mem_handle_t *phMem,
    ur_exp_image_handle_t *phImage) {
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = hImageMem;
  std::ignore = pImageFormat;
  std::ignore = pImageDesc;
  std::ignore = phMem;
  std::ignore = phImage;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesSampledImageCreateExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_mem_handle_t hImageMem, const ur_image_format_t *pImageFormat,
    const ur_image_desc_t *pImageDesc, ur_sampler_handle_t hSampler,
    ur_mem_handle_t *phMem, ur_exp_image_handle_t *phImage) {
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = hImageMem;
  std::ignore = pImageFormat;
  std::ignore = pImageDesc;
  std::ignore = hSampler;
  std::ignore = phMem;
  std::ignore = phImage;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImageCopyExp(
    ur_queue_handle_t hQueue, void *pDst, void *pSrc,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    ur_exp_image_copy_flags_t imageCopyFlags, ur_rect_offset_t srcOffset,
    ur_rect_offset_t dstOffset, ur_rect_region_t copyExtent,
    ur_rect_region_t hostExtent, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = pDst;
  std::ignore = pSrc;
  std::ignore = pImageFormat;
  std::ignore = pImageDesc;
  std::ignore = imageCopyFlags;
  std::ignore = srcOffset;
  std::ignore = dstOffset;
  std::ignore = copyExtent;
  std::ignore = hostExtent;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImageGetInfoExp(
    ur_exp_image_mem_handle_t hImageMem, ur_image_info_t propName,
    void *pPropValue, size_t *pPropSizeRet) {
  std::ignore = hImageMem;
  std::ignore = propName;
  std::ignore = pPropValue;
  std::ignore = pPropSizeRet;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesMipmapGetLevelExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_mem_handle_t hImageMem, uint32_t mipmapLevel,
    ur_exp_image_mem_handle_t *phImageMem) {
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = hImageMem;
  std::ignore = mipmapLevel;
  std::ignore = phImageMem;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesMipmapFreeExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_mem_handle_t hMem) {
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = hMem;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImportOpaqueFDExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, size_t size,
    ur_exp_interop_mem_desc_t *pInteropMemDesc,
    ur_exp_interop_mem_handle_t *phInteropMem) {
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = size;
  std::ignore = pInteropMemDesc;
  std::ignore = phInteropMem;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesMapExternalArrayExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    ur_exp_interop_mem_handle_t hInteropMem,
    ur_exp_image_mem_handle_t *phImageMem) {
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = pImageFormat;
  std::ignore = pImageDesc;
  std::ignore = hInteropMem;
  std::ignore = phImageMem;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesReleaseInteropExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_interop_mem_handle_t hInteropMem) {
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = hInteropMem;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesImportExternalSemaphoreOpaqueFDExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_interop_semaphore_desc_t *pInteropSemaphoreDesc,
    ur_exp_interop_semaphore_handle_t *phInteropSemaphoreHandle) {
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = pInteropSemaphoreDesc;
  std::ignore = phInteropSemaphoreHandle;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesDestroyExternalSemaphoreExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_interop_semaphore_handle_t hInteropSemaphore) {
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = hInteropSemaphore;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesWaitExternalSemaphoreExp(
    ur_queue_handle_t hQueue, ur_exp_interop_semaphore_handle_t hSemaphore,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = hSemaphore;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesSignalExternalSemaphoreExp(
    ur_queue_handle_t hQueue, ur_exp_interop_semaphore_handle_t hSemaphore,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = hSemaphore;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
