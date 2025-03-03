//===--------- image.cpp - Level Zero Adapter -----------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "helpers/image_helpers.hpp"
#include "logger/ur_logger.hpp"
#include "ur_level_zero.hpp"
#include "ur_interface_loader.hpp"


typedef ze_result_t(ZE_APICALL *zeImageGetDeviceOffsetExp_pfn)(
    ze_image_handle_t hImage, uint64_t *pDeviceOffset);

typedef ze_result_t(ZE_APICALL *zeMemGetPitchFor2dImage_pfn)(
    ze_context_handle_t hContext, ze_device_handle_t hDevice, size_t imageWidth,
    size_t imageHeight, unsigned int elementSizeInBytes, size_t *rowPitch);

namespace {

[[maybe_unused]] zeMemGetPitchFor2dImage_pfn zeMemGetPitchFor2dImageFunctionPtr = nullptr;

zeImageGetDeviceOffsetExp_pfn zeImageGetDeviceOffsetExpFunctionPtr = nullptr;

// TODO: unify with image_helpers.cpp/ur2zeImageDesc
static ur_result_t
ur2zeImageDescBindless(const ur_image_format_t *ImageFormat,
                       const ur_image_desc_t *ImageDesc,
                       ZeStruct<ze_image_desc_t> &ZeImageDesc) {
  std::cerr << "[L0_v2]" << __FUNCTION__ << std::endl;
  
  auto [ZeImageFormatType, ZeImageFormatTypeSize] =
      getImageFormatTypeAndSize(ImageFormat);
  if (ZeImageFormatTypeSize == 0) {
    return UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT;
  }
  // TODO: populate the layout mapping
  ze_image_format_layout_t ZeImageFormatLayout;
  switch (ImageFormat->channelOrder) {
  case UR_IMAGE_CHANNEL_ORDER_A:
  case UR_IMAGE_CHANNEL_ORDER_R: {
    switch (ZeImageFormatTypeSize) {
    case 8:
      ZeImageFormatLayout = ZE_IMAGE_FORMAT_LAYOUT_8;
      break;
    case 16:
      ZeImageFormatLayout = ZE_IMAGE_FORMAT_LAYOUT_16;
      break;
    case 32:
      ZeImageFormatLayout = ZE_IMAGE_FORMAT_LAYOUT_32;
      break;
    default:
      logger::error("ur2zeImageDescBindless: unexpected data type size");
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
    break;
  }
  case UR_IMAGE_CHANNEL_ORDER_RG:
  case UR_IMAGE_CHANNEL_ORDER_RA:
  case UR_IMAGE_CHANNEL_ORDER_RX: {
    switch (ZeImageFormatTypeSize) {
    case 8:
      ZeImageFormatLayout = ZE_IMAGE_FORMAT_LAYOUT_8_8;
      break;
    case 16:
      ZeImageFormatLayout = ZE_IMAGE_FORMAT_LAYOUT_16_16;
      break;
    case 32:
      ZeImageFormatLayout = ZE_IMAGE_FORMAT_LAYOUT_32_32;
      break;
    default:
      logger::error("ur2zeImageDescBindless: unexpected data type size");
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
    break;
  }
  case UR_IMAGE_CHANNEL_ORDER_RGBX:
  case UR_IMAGE_CHANNEL_ORDER_RGBA:
  case UR_IMAGE_CHANNEL_ORDER_ARGB:
  case UR_IMAGE_CHANNEL_ORDER_BGRA: {
    switch (ZeImageFormatTypeSize) {
    case 8:
      ZeImageFormatLayout = ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8;
      break;
    case 16:
      ZeImageFormatLayout = ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16;
      break;
    case 32:
      ZeImageFormatLayout = ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32;
      break;
    default:
      logger::error("ur2zeImageDescBindless: unexpected data type size");
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
    break;
  }
  case UR_IMAGE_CHANNEL_ORDER_RGB:
  case UR_IMAGE_CHANNEL_ORDER_RGX: {
    switch (ZeImageFormatTypeSize) {
    case 8:
      ZeImageFormatLayout = ZE_IMAGE_FORMAT_LAYOUT_8_8_8;
      break;
    case 16:
      ZeImageFormatLayout = ZE_IMAGE_FORMAT_LAYOUT_16_16_16;
      break;
    case 32:
      ZeImageFormatLayout = ZE_IMAGE_FORMAT_LAYOUT_32_32_32;
      break;
    default:
      logger::error("ur2zeImageDescBindless: unexpected data type size");
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
    break;
  }
  default:
    logger::error("format channel order = {}", ImageFormat->channelOrder);
    die("ur2zeImageDescBindless: unsupported image channel order\n");
    break;
  }

  ze_image_format_t ZeFormatDesc = {
      ZeImageFormatLayout, ZeImageFormatType,
      // TODO: are swizzles deducted from image_format->image_channel_order?
      ZE_IMAGE_FORMAT_SWIZZLE_R, ZE_IMAGE_FORMAT_SWIZZLE_G,
      ZE_IMAGE_FORMAT_SWIZZLE_B, ZE_IMAGE_FORMAT_SWIZZLE_A};

  ze_image_type_t ZeImageType;
  switch (ImageDesc->type) {
  case UR_MEM_TYPE_IMAGE1D:
    ZeImageType = ZE_IMAGE_TYPE_1D;
    break;
  case UR_MEM_TYPE_IMAGE2D:
    ZeImageType = ZE_IMAGE_TYPE_2D;
    break;
  case UR_MEM_TYPE_IMAGE3D:
    ZeImageType = ZE_IMAGE_TYPE_3D;
    break;
  case UR_MEM_TYPE_IMAGE1D_ARRAY:
    ZeImageType = ZE_IMAGE_TYPE_1DARRAY;
    break;
  case UR_MEM_TYPE_IMAGE2D_ARRAY:
    ZeImageType = ZE_IMAGE_TYPE_2DARRAY;
    break;
  default:
    logger::error("ur2zeImageDescBindless: unsupported image type");
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  ZeImageDesc.stype = ZE_STRUCTURE_TYPE_IMAGE_DESC;
  ZeImageDesc.pNext = ImageDesc->pNext;
  ZeImageDesc.type = ZeImageType;
  ZeImageDesc.format = ZeFormatDesc;
  ZeImageDesc.width = ur_cast<uint64_t>(ImageDesc->width);
  ZeImageDesc.height =
      std::max(ur_cast<uint64_t>(ImageDesc->height), (uint64_t)1);
  ZeImageDesc.depth =
      std::max(ur_cast<uint64_t>(ImageDesc->depth), (uint64_t)1);
  ZeImageDesc.arraylevels = ur_cast<uint32_t>(ImageDesc->arraySize);
  ZeImageDesc.miplevels = ImageDesc->numMipLevel;

  std::cerr << "[L0_v2]" << __FUNCTION__ << " success" <<std::endl;
  return UR_RESULT_SUCCESS;
}


ur_result_t bindlessImagesCreateImpl(ur_context_handle_t hContext,
                                     ur_device_handle_t hDevice,
                                     ur_exp_image_mem_native_handle_t hImageMem,
                                     const ur_image_format_t *pImageFormat,
                                     const ur_image_desc_t *pImageDesc,
                                     ur_sampler_handle_t hSampler,
                                     ur_exp_image_native_handle_t *phImage) {
  std::shared_lock<ur_shared_mutex> Lock(hContext->Mutex);
 
  std::cerr << "[L0_v2]" << __FUNCTION__ << std::endl;
 
  ZeStruct<ze_image_desc_t> ZeImageDesc;
  UR_CALL(ur2zeImageDescBindless(pImageFormat, pImageDesc, ZeImageDesc));

  ZeStruct<ze_image_bindless_exp_desc_t> BindlessDesc;

  BindlessDesc.flags = ZE_IMAGE_BINDLESS_EXP_FLAG_BINDLESS;
  ZeImageDesc.pNext = &BindlessDesc;

  ZeStruct<ze_sampler_desc_t> ZeSamplerDesc;
  if (hSampler) {
    ZeSamplerDesc = hSampler->ZeSamplerDesc;
    BindlessDesc.pNext = &ZeSamplerDesc;
    BindlessDesc.flags |= ZE_IMAGE_BINDLESS_EXP_FLAG_SAMPLED_IMAGE;
  }

  ze_image_handle_t ZeImage;

  ze_memory_allocation_properties_t MemAllocProperties{
      ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES, nullptr,
      ZE_MEMORY_TYPE_UNKNOWN, 0, 0};
  ZE2UR_CALL(zeMemGetAllocProperties,
             (hContext->ZeContext, reinterpret_cast<const void *>(hImageMem),
              &MemAllocProperties, nullptr));

  if (MemAllocProperties.type == ZE_MEMORY_TYPE_UNKNOWN) {
    // _ur_image *UrImage = reinterpret_cast<_ur_image *>(hImageMem);
    ze_image_handle_t zeImg = reinterpret_cast<ze_image_handle_t>(hImageMem);

    // std::cerr << "[L0_v2]" << __FUNCTION__ << " hImageMem=" << UrImage << std::endl;
    ZE2UR_CALL(zeImageViewCreateExt,
               (hContext->ZeContext, hDevice->ZeDevice, &ZeImageDesc,
                /* UrImage->ZeImage */zeImg, &ZeImage));
    ZE2UR_CALL(zeContextMakeImageResident,
               (hContext->ZeContext, hDevice->ZeDevice, ZeImage));
  } else if (MemAllocProperties.type == ZE_MEMORY_TYPE_DEVICE ||
             MemAllocProperties.type == ZE_MEMORY_TYPE_HOST ||
             MemAllocProperties.type == ZE_MEMORY_TYPE_SHARED) {
    ZeStruct<ze_image_pitched_exp_desc_t> PitchedDesc;
    PitchedDesc.ptr = reinterpret_cast<void *>(hImageMem);
    if (hSampler) {
      ZeSamplerDesc.pNext = &PitchedDesc;
    } else {
      BindlessDesc.pNext = &PitchedDesc;
    }

    ZE2UR_CALL(zeImageCreate, (hContext->ZeContext, hDevice->ZeDevice,
                               &ZeImageDesc, &ZeImage));
    ZE2UR_CALL(zeContextMakeImageResident,
               (hContext->ZeContext, hDevice->ZeDevice, ZeImage));
  } else {
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  static std::once_flag InitFlag;
  std::call_once(InitFlag, [&]() {
    ze_driver_handle_t DriverHandle = hContext->getPlatform()->ZeDriver;
    auto Result = zeDriverGetExtensionFunctionAddress(
        DriverHandle, "zeImageGetDeviceOffsetExp",
        (void **)&zeImageGetDeviceOffsetExpFunctionPtr);
    if (Result != ZE_RESULT_SUCCESS)
      logger::error("zeDriverGetExtensionFunctionAddress "
                    "zeImageGetDeviceOffsetExpv failed, err = {}",
                    Result);
  });
  if (!zeImageGetDeviceOffsetExpFunctionPtr)
    return UR_RESULT_ERROR_INVALID_OPERATION;

  uint64_t DeviceOffset{};
  ze_image_handle_t ZeImageTranslated;
  ZE2UR_CALL(zelLoaderTranslateHandle,
             (ZEL_HANDLE_IMAGE, ZeImage, (void **)&ZeImageTranslated));
  ZE2UR_CALL(zeImageGetDeviceOffsetExpFunctionPtr,
             (ZeImageTranslated, &DeviceOffset));
  *phImage = DeviceOffset;

  hDevice->ZeOffsetToImageHandleMap[*phImage] = ZeImage;

  std::cerr << "[L0_v2]" << __FUNCTION__ << " success" <<std::endl;
  return UR_RESULT_SUCCESS;
}

} // namespace

namespace ur::level_zero {

ur_result_t urUSMPitchedAllocExp([[maybe_unused]] ur_context_handle_t hContext,
                                 [[maybe_unused]] ur_device_handle_t hDevice,
                                 [[maybe_unused]] const ur_usm_desc_t *pUSMDesc,
                                 [[maybe_unused]] ur_usm_pool_handle_t pool, 
                                 [[maybe_unused]] size_t widthInBytes,
                                 [[maybe_unused]] size_t height, 
                                 [[maybe_unused]] size_t elementSizeBytes,
                                 [[maybe_unused]] void **ppMem, 
                                 [[maybe_unused]] size_t *pResultPitch) {
    std::cerr << "[UR API][L0_v2]" <<__FUNCTION__ << std::endl;
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

ur_result_t urBindlessImagesUnsampledImageHandleDestroyExp(
    [[maybe_unused]] ur_context_handle_t hContext, 
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_exp_image_native_handle_t hImage) {

  std::cerr << "[UR API][L0_v2]" <<__FUNCTION__ << std::endl;

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
    [[maybe_unused]] ur_context_handle_t hContext, 
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_exp_image_native_handle_t hImage) {
  std::cerr << "[UR API][L0_v2]" <<__FUNCTION__ << std::endl;
  logger::error(logger::LegacyMessage("[UR][L0_v2] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return static_cast<ur_result_t>(2);

}

ur_result_t urBindlessImagesImageAllocateExp(
    [[maybe_unused]] ur_context_handle_t hContext, 
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] const ur_image_format_t *pImageFormat, 
    [[maybe_unused]] const ur_image_desc_t *pImageDesc,
    [[maybe_unused]] ur_exp_image_mem_native_handle_t *phImageMem) {

  std::cerr << "[UR API][L0_v2]" <<__FUNCTION__ << std::endl;
  std::shared_lock<ur_shared_mutex> Lock(hContext->Mutex);

  UR_ASSERT(hContext && hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pImageFormat && pImageDesc && phImageMem,
            UR_RESULT_ERROR_INVALID_NULL_POINTER);

  ZeStruct<ze_image_desc_t> ZeImageDesc;
  UR_CALL(ur2zeImageDescBindless(pImageFormat, pImageDesc, ZeImageDesc));

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
  *phImageMem = reinterpret_cast<ur_exp_image_mem_native_handle_t>(ZeImage);
  return UR_RESULT_SUCCESS;
  
}

ur_result_t
urBindlessImagesImageFreeExp([[maybe_unused]] ur_context_handle_t hContext,
                             [[maybe_unused]] ur_device_handle_t hDevice,
                             [[maybe_unused]] ur_exp_image_mem_native_handle_t hImageMem) {

  std::cerr << "[UR API][L0_v2]" <<__FUNCTION__ << " hImageMem=0x" << std::hex << hImageMem << std::endl;
  UR_CALL(ur::level_zero::urMemRelease(
      reinterpret_cast<ur_mem_handle_t>(hImageMem)));
  return UR_RESULT_SUCCESS;
}

ur_result_t urBindlessImagesUnsampledImageCreateExp(
    [[maybe_unused]] ur_context_handle_t hContext, 
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_exp_image_mem_native_handle_t hImageMem,
    [[maybe_unused]] const ur_image_format_t *pImageFormat, 
    [[maybe_unused]] const ur_image_desc_t *pImageDesc,
    [[maybe_unused]] ur_exp_image_native_handle_t *phImage) {

  std::cerr << "[UR API][L0_v2]" <<__FUNCTION__ << std::endl;
  UR_CALL(bindlessImagesCreateImpl(hContext, hDevice, hImageMem, pImageFormat,
                                    pImageDesc, nullptr, phImage));

  std::cerr << "[UR API][L0v2 adapter] urBindlessImagesUnsampledImageCreateExp done" << std::endl;
  return UR_RESULT_SUCCESS;
}

ur_result_t urBindlessImagesSampledImageCreateExp(
    [[maybe_unused]] ur_context_handle_t hContext, 
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_exp_image_mem_native_handle_t hImageMem,
    [[maybe_unused]] const ur_image_format_t *pImageFormat, 
    [[maybe_unused]] const ur_image_desc_t *pImageDesc,
    [[maybe_unused]] ur_sampler_handle_t hSampler, 
    [[maybe_unused]] ur_exp_image_native_handle_t *phImage) {

  std::cerr << "[UR API][L0_v2]" <<__FUNCTION__ << std::endl;

  UR_CALL(bindlessImagesCreateImpl(hContext, hDevice, hImageMem, pImageFormat,
                                   pImageDesc, nullptr, phImage));
  return UR_RESULT_SUCCESS;
}



ur_result_t urBindlessImagesImageGetInfoExp(
    [[maybe_unused]] ur_context_handle_t, 
    [[maybe_unused]] ur_exp_image_mem_native_handle_t hImageMem,
    [[maybe_unused]] ur_image_info_t propName, 
    [[maybe_unused]] void *pPropValue, 
    [[maybe_unused]] size_t *pPropSizeRet) {
  std::cerr << "[UR API][L0_v2]" <<__FUNCTION__ << std::endl;

  logger::error(logger::LegacyMessage("[UR][L0_v2] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return static_cast<ur_result_t>(7);

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

  std::cerr << "[UR API][L0_v2]" <<__FUNCTION__ << std::endl;

  logger::error(logger::LegacyMessage("[UR][L0_v2] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return static_cast<ur_result_t>(8);
}

ur_result_t
urBindlessImagesMipmapFreeExp([[maybe_unused]] ur_context_handle_t hContext,
                              [[maybe_unused]] ur_device_handle_t hDevice,
                              [[maybe_unused]] ur_exp_image_mem_native_handle_t hMem) {

  std::cerr << "[UR API][L0_v2]" <<__FUNCTION__ << std::endl;

  logger::error(logger::LegacyMessage("[UR][L0_v2] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return static_cast<ur_result_t>(9);
}

ur_result_t urBindlessImagesImportExternalMemoryExp(
    [[maybe_unused]] ur_context_handle_t hContext, 
    [[maybe_unused]] ur_device_handle_t hDevice, 
    [[maybe_unused]] size_t size,
    [[maybe_unused]] ur_exp_external_mem_type_t memHandleType,
    [[maybe_unused]] ur_exp_external_mem_desc_t *pExternalMemDesc,
    [[maybe_unused]] ur_exp_external_mem_handle_t *phExternalMem) {

  std::cerr << "[UR API][L0_v2]" <<__FUNCTION__ << std::endl;

  logger::error(logger::LegacyMessage("[UR][L0_v2] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return static_cast<ur_result_t>(10);

}

ur_result_t urBindlessImagesMapExternalArrayExp(
    [[maybe_unused]] ur_context_handle_t hContext, 
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] const ur_image_format_t *pImageFormat, 
    [[maybe_unused]] const ur_image_desc_t *pImageDesc,
    [[maybe_unused]] ur_exp_external_mem_handle_t hExternalMem,
    [[maybe_unused]] ur_exp_image_mem_native_handle_t *phImageMem) {

  std::cerr << "[UR API][L0_v2]" <<__FUNCTION__ << std::endl;

  logger::error(logger::LegacyMessage("[UR][L0_v2] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return static_cast<ur_result_t>(11);
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

std::cerr << "[UR API][L0_v2]" <<__FUNCTION__ << std::endl;
  logger::error(logger::LegacyMessage("[UR][L0_v2] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return static_cast<ur_result_t>(12);
}

ur_result_t urBindlessImagesReleaseExternalMemoryExp(
    [[maybe_unused]] ur_context_handle_t hContext, 
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_exp_external_mem_handle_t hExternalMem) {

std::cerr << "[UR API][L0_v2]" <<__FUNCTION__ << std::endl;

  logger::error(logger::LegacyMessage("[UR][L0_v2] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return static_cast<ur_result_t>(13);
}

ur_result_t urBindlessImagesImportExternalSemaphoreExp(
    [[maybe_unused]] ur_context_handle_t hContext, 
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_exp_external_semaphore_type_t semHandleType,
    [[maybe_unused]] ur_exp_external_semaphore_desc_t *pExternalSemaphoreDesc,
    [[maybe_unused]] ur_exp_external_semaphore_handle_t *phExternalSemaphoreHandle) {
std::cerr << "[UR API][L0_v2]" <<__FUNCTION__ << std::endl;
  logger::error(logger::LegacyMessage("[UR][L0_v2] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return static_cast<ur_result_t>(14);

}

ur_result_t urBindlessImagesReleaseExternalSemaphoreExp(
    [[maybe_unused]] ur_context_handle_t hContext, 
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_exp_external_semaphore_handle_t hExternalSemaphore) {
std::cerr << "[UR API][L0_v2]" <<__FUNCTION__ << std::endl;
  logger::error(logger::LegacyMessage("[UR][L0_v2] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return static_cast<ur_result_t>(15);

}

// ur_result_t urBindlessImagesWaitExternalSemaphoreExp() { return UR_RESULT_SUCCESS; }
// ur_result_t urBindlessImagesSignalExternalSemaphoreExp() { return UR_RESULT_SUCCESS; }

} // namespace ur::level_zero
//===--------- memory.cpp - Level Zero Adapter ----------------------------===//