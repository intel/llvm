//===--------- image.cpp - Level Zero Adapter -----------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"

#include "../helpers/image_helpers.hpp"
#include "logger/ur_logger.hpp"
#include "queue_api.hpp"
#include "queue_handle.hpp"
#include "ur_interface_loader.hpp"
#include "v2/context.hpp"
#include "v2/memory.hpp"

namespace ur::level_zero {

ur_result_t
urBindlessImagesImageFreeExp([[maybe_unused]] ur_context_handle_t hContext,
                             [[maybe_unused]] ur_device_handle_t hDevice,
                             ur_exp_image_mem_native_handle_t hImageMem) {
  ur_mem_image_t *urImg = reinterpret_cast<ur_mem_image_t *>(hImageMem);
  ZE2UR_CALL(zeImageDestroy, (urImg->getZeImage()));
  return UR_RESULT_SUCCESS;
}

ur_result_t urBindlessImagesImageCopyExp(
    ur_queue_handle_t hQueue, const void *pSrc, void *pDst,
    const ur_image_desc_t *pSrcImageDesc, const ur_image_desc_t *pDstImageDesc,
    const ur_image_format_t *pSrcImageFormat,
    const ur_image_format_t *pDstImageFormat,
    ur_exp_image_copy_region_t *pCopyRegion,
    ur_exp_image_copy_flags_t imageCopyFlags, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) try {

  ur_result_t res = hQueue->get().bindlessImagesImageCopyExp(
      pSrc, pDst, pSrcImageDesc, pDstImageDesc, pSrcImageFormat,
      pDstImageFormat, pCopyRegion, imageCopyFlags, numEventsInWaitList,
      phEventWaitList, phEvent);
  return res;
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urBindlessImagesWaitExternalSemaphoreExp(
    ur_queue_handle_t hQueue, ur_exp_external_semaphore_handle_t hSemaphore,
    bool hasWaitValue, uint64_t waitValue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) try {
  return hQueue->get().bindlessImagesWaitExternalSemaphoreExp(
      hSemaphore, hasWaitValue, waitValue, numEventsInWaitList, phEventWaitList,
      phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}
ur_result_t urBindlessImagesSignalExternalSemaphoreExp(
    ur_queue_handle_t hQueue, ur_exp_external_semaphore_handle_t hSemaphore,
    bool hasSignalValue, uint64_t signalValue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) try {
  return hQueue->get().bindlessImagesSignalExternalSemaphoreExp(
      hSemaphore, hasSignalValue, signalValue, numEventsInWaitList,
      phEventWaitList, phEvent);
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urBindlessImagesImageGetInfoExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_exp_image_mem_native_handle_t hImageMem,
    [[maybe_unused]] ur_image_info_t propName,
    [[maybe_unused]] void *pPropValue, [[maybe_unused]] size_t *pPropSizeRet) {
  logger::error(
      logger::LegacyMessage("[UR][L0_v2] {} function not implemented!"),
      "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urBindlessImagesMipmapFreeExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_exp_image_mem_native_handle_t hMem) {

  logger::error(
      logger::LegacyMessage("[UR][L0_v2] {} function not implemented!"),
      "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urBindlessImagesImportExternalMemoryExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice, [[maybe_unused]] size_t size,
    [[maybe_unused]] ur_exp_external_mem_type_t memHandleType,
    [[maybe_unused]] ur_exp_external_mem_desc_t *pExternalMemDesc,
    [[maybe_unused]] ur_exp_external_mem_handle_t *phExternalMem) {

  logger::error(
      logger::LegacyMessage("[UR][L0_v2] {} function not implemented!"),
      "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urBindlessImagesMapExternalArrayExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] const ur_image_format_t *pImageFormat,
    [[maybe_unused]] const ur_image_desc_t *pImageDesc,
    [[maybe_unused]] ur_exp_external_mem_handle_t hExternalMem,
    [[maybe_unused]] ur_exp_image_mem_native_handle_t *phImageMem) {

  logger::error(
      logger::LegacyMessage("[UR][L0_v2] {} function not implemented!"),
      "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urBindlessImagesMapExternalLinearMemoryExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, uint64_t offset,
    uint64_t size, ur_exp_external_mem_handle_t hExternalMem, void **phRetMem) {
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = size;
  std::ignore = offset;
  std::ignore = hExternalMem;
  std::ignore = phRetMem;

  logger::error(
      logger::LegacyMessage("[UR][L0_v2] {} function not implemented!"),
      "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urBindlessImagesReleaseExternalMemoryExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_exp_external_mem_handle_t hExternalMem) {

  logger::error(
      logger::LegacyMessage("[UR][L0_v2] {} function not implemented!"),
      "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urBindlessImagesImportExternalSemaphoreExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_exp_external_semaphore_type_t semHandleType,
    [[maybe_unused]] ur_exp_external_semaphore_desc_t *pExternalSemaphoreDesc,
    [[maybe_unused]] ur_exp_external_semaphore_handle_t
        *phExternalSemaphoreHandle) {
  logger::error(
      logger::LegacyMessage("[UR][L0_v2] {} function not implemented!"),
      "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urBindlessImagesReleaseExternalSemaphoreExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_exp_external_semaphore_handle_t hExternalSemaphore) {
  logger::error(
      logger::LegacyMessage("[UR][L0_v2] {} function not implemented!"),
      "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

} // namespace ur::level_zero
//===--------- image.cpp - Level Zero Adapter ----------------------------===//