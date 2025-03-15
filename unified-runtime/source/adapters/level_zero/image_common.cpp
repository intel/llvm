//===--------- image.cpp - Level Zero Adapter -----------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <loader/ze_loader.h>

#include "common.hpp"
#include "context.hpp"
#include "event.hpp"
#include "helpers/image_helpers.hpp"
#include "logger/ur_logger.hpp"
#ifdef UR_ADAPTER_LEVEL_ZERO_V2
#include "v2/memory.hpp"
#else
#include "memory.hpp"
#endif
#include "sampler.hpp"
#include "ur_interface_loader.hpp"

zeMemGetPitchFor2dImage_pfn zeMemGetPitchFor2dImageFunctionPtr = nullptr;

namespace ur::level_zero {

ur_result_t urUSMPitchedAllocExp(ur_context_handle_t hContext,
                                 ur_device_handle_t hDevice,
                                 const ur_usm_desc_t *pUSMDesc,
                                 ur_usm_pool_handle_t pool, size_t widthInBytes,
                                 size_t height, size_t elementSizeBytes,
                                 void **ppMem, size_t *pResultPitch) {
  std::shared_lock<ur_shared_mutex> Lock(hContext->Mutex);

  UR_ASSERT(hContext && hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(widthInBytes != 0, UR_RESULT_ERROR_INVALID_USM_SIZE);
  UR_ASSERT(ppMem && pResultPitch, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  static std::once_flag InitFlag;
  std::call_once(InitFlag, [&]() {
    ze_driver_handle_t DriverHandle = hContext->getPlatform()->ZeDriver;
    auto Result = zeDriverGetExtensionFunctionAddress(
        DriverHandle, "zeMemGetPitchFor2dImage",
        (void **)&zeMemGetPitchFor2dImageFunctionPtr);
    if (Result != ZE_RESULT_SUCCESS)
      logger::error(
          "zeDriverGetExtensionFunctionAddress zeMemGetPitchFor2dImage "
          "failed, err = {}",
          Result);
  });
  if (!zeMemGetPitchFor2dImageFunctionPtr)
    return UR_RESULT_ERROR_INVALID_OPERATION;

  size_t Width = widthInBytes / elementSizeBytes;
  size_t RowPitch;
  ze_device_handle_t ZeDeviceTranslated;
  ZE2UR_CALL(zelLoaderTranslateHandle, (ZEL_HANDLE_DEVICE, hDevice->ZeDevice,
                                        (void **)&ZeDeviceTranslated));
  ZE2UR_CALL(zeMemGetPitchFor2dImageFunctionPtr,
             (hContext->ZeContext, ZeDeviceTranslated, Width, height,
              elementSizeBytes, &RowPitch));
  *pResultPitch = RowPitch;

  size_t Size = height * RowPitch;
  UR_CALL(ur::level_zero::urUSMDeviceAlloc(hContext, hDevice, pUSMDesc, pool,
                                           Size, ppMem));

  return UR_RESULT_SUCCESS;
}

ur_result_t urBindlessImagesImageAllocateExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    ur_exp_image_mem_native_handle_t *phImageMem) {
  std::shared_lock<ur_shared_mutex> Lock(hContext->Mutex);

  UR_ASSERT(hContext && hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pImageFormat && pImageDesc && phImageMem,
            UR_RESULT_ERROR_INVALID_NULL_POINTER);

  ZeStruct<ze_image_desc_t> ZeImageDesc;
  UR_CALL(ur2zeImageDesc(pImageFormat, pImageDesc, ZeImageDesc));

  ze_image_bindless_exp_desc_t ZeImageBindlessDesc;
  ZeImageBindlessDesc.stype = ZE_STRUCTURE_TYPE_BINDLESS_IMAGE_EXP_DESC;
  ZeImageBindlessDesc.pNext = nullptr;
  ZeImageBindlessDesc.flags = ZE_IMAGE_BINDLESS_EXP_FLAG_BINDLESS;
  ZeImageDesc.pNext = &ZeImageBindlessDesc;

  ze_image_handle_t ZeImage;
  ZE2UR_CALL(zeImageCreate,
             (hContext->ZeContext, hDevice->ZeDevice, &ZeImageDesc, &ZeImage));
  ZE2UR_CALL(zeContextMakeImageResident,
             (hContext->ZeContext, hDevice->ZeDevice, ZeImage));
  UR_CALL(createUrImgFromZeImage(ZeImage, ZeImageDesc, phImageMem));
  return UR_RESULT_SUCCESS;
}

ur_result_t urBindlessImagesUnsampledImageCreateExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_mem_native_handle_t hImageMem,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    ur_exp_image_native_handle_t *phImage) {
  UR_CALL(bindlessImagesCreateImpl(hContext, hDevice, hImageMem, pImageFormat,
                                   pImageDesc, nullptr, phImage));
  return UR_RESULT_SUCCESS;
}

ur_result_t urBindlessImagesSampledImageCreateExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_mem_native_handle_t hImageMem,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    ur_sampler_handle_t hSampler, ur_exp_image_native_handle_t *phImage) {
  UR_CALL(bindlessImagesCreateImpl(hContext, hDevice, hImageMem, pImageFormat,
                                   pImageDesc, hSampler, phImage));
  return UR_RESULT_SUCCESS;
}

ur_result_t urBindlessImagesUnsampledImageHandleDestroyExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_native_handle_t hImage) {

  UR_ASSERT(hContext && hDevice && hImage, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  auto item = hDevice->ZeOffsetToImageHandleMap.find(hImage);

  if (item != hDevice->ZeOffsetToImageHandleMap.end()) {
    ZE2UR_CALL(zeImageDestroy, (item->second));
    hDevice->ZeOffsetToImageHandleMap.erase(item);
  } else {
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urBindlessImagesSampledImageHandleDestroyExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_native_handle_t hImage) {
  // Sampled image is a combination of unsampled image and sampler.
  // Sampler is released in urSamplerRelease.
  return ur::level_zero::urBindlessImagesUnsampledImageHandleDestroyExp(
      hContext, hDevice, hImage);
}

ur_result_t
urBindlessImagesMipmapFreeExp(ur_context_handle_t hContext,
                              ur_device_handle_t hDevice,
                              ur_exp_image_mem_native_handle_t hMem) {
  return ur::level_zero::urBindlessImagesImageFreeExp(hContext, hDevice, hMem);
}

ur_result_t urBindlessImagesImageGetInfoExp(
    ur_context_handle_t, ur_exp_image_mem_native_handle_t hImageMem,
    ur_image_info_t propName, void *pPropValue, size_t *pPropSizeRet) {
  UR_ASSERT(hImageMem, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(UR_IMAGE_INFO_DEPTH >= propName,
            UR_RESULT_ERROR_INVALID_ENUMERATION);
  UR_ASSERT(pPropValue || pPropSizeRet, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  ur_bindless_mem_handle_t *urImg =
      reinterpret_cast<ur_bindless_mem_handle_t *>(hImageMem);
  ze_image_desc_t &Desc = urImg->getZeImageDesc();

  switch (propName) {
  case UR_IMAGE_INFO_WIDTH:
    if (pPropValue) {
      *(uint64_t *)pPropValue = Desc.width;
    }
    if (pPropSizeRet) {
      *pPropSizeRet = sizeof(uint64_t);
    }
    return UR_RESULT_SUCCESS;
  case UR_IMAGE_INFO_HEIGHT:
    if (pPropValue) {
      *(uint32_t *)pPropValue = Desc.height;
    }
    if (pPropSizeRet) {
      *pPropSizeRet = sizeof(uint32_t);
    }
    return UR_RESULT_SUCCESS;
  case UR_IMAGE_INFO_DEPTH:
    if (pPropValue) {
      *(uint32_t *)pPropValue = Desc.depth;
    }
    if (pPropSizeRet) {
      *pPropSizeRet = sizeof(uint32_t);
    }
    return UR_RESULT_SUCCESS;
  case UR_IMAGE_INFO_FORMAT:
    if (pPropValue) {
      ur_image_format_t UrImageFormat;
      UR_CALL(ze2urImageFormat(&Desc, &UrImageFormat));
      *(ur_image_format_t *)pPropValue = UrImageFormat;
    }
    if (pPropSizeRet) {
      *pPropSizeRet = sizeof(ur_image_format_t);
    }
    return UR_RESULT_SUCCESS;
  default:
    return UR_RESULT_ERROR_INVALID_VALUE;
  }
}

ur_result_t urBindlessImagesMipmapGetLevelExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_mem_native_handle_t hImageMem, uint32_t mipmapLevel,
    ur_exp_image_mem_native_handle_t *phImageMem) {
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = hImageMem;
  std::ignore = mipmapLevel;
  std::ignore = phImageMem;
  logger::error(logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

} // namespace ur::level_zero
