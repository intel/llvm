//===--------- image.cpp - Level Zero Adapter -----------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "image.hpp"
#include "common.hpp"
#include "context.hpp"
#include "event.hpp"
#include "logger/ur_logger.hpp"
#include "sampler.hpp"
#include "ur_interface_loader.hpp"
#include "ur_level_zero.hpp"
#include "ze_api.h"

typedef ze_result_t(ZE_APICALL *zeImageGetDeviceOffsetExp_pfn)(
    ze_image_handle_t hImage, uint64_t *pDeviceOffset);

typedef ze_result_t(ZE_APICALL *zeMemGetPitchFor2dImage_pfn)(
    ze_context_handle_t hContext, ze_device_handle_t hDevice, size_t imageWidth,
    size_t imageHeight, unsigned int elementSizeInBytes, size_t *rowPitch);

namespace {

zeMemGetPitchFor2dImage_pfn zeMemGetPitchFor2dImageFunctionPtr = nullptr;

zeImageGetDeviceOffsetExp_pfn zeImageGetDeviceOffsetExpFunctionPtr = nullptr;

/// Construct UR image format from ZE image desc.
ur_result_t ze2urImageFormat(const ze_image_desc_t *ZeImageDesc,
                             ur_image_format_t *UrImageFormat) {
  const ze_image_format_t &ZeImageFormat = ZeImageDesc->format;
  size_t ZeImageFormatTypeSize;
  switch (ZeImageFormat.layout) {
  case ZE_IMAGE_FORMAT_LAYOUT_8:
  case ZE_IMAGE_FORMAT_LAYOUT_8_8:
  case ZE_IMAGE_FORMAT_LAYOUT_8_8_8:
  case ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8:
    ZeImageFormatTypeSize = 8;
    break;
  case ZE_IMAGE_FORMAT_LAYOUT_16:
  case ZE_IMAGE_FORMAT_LAYOUT_16_16:
  case ZE_IMAGE_FORMAT_LAYOUT_16_16_16:
  case ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16:
    ZeImageFormatTypeSize = 16;
    break;
  case ZE_IMAGE_FORMAT_LAYOUT_32:
  case ZE_IMAGE_FORMAT_LAYOUT_32_32:
  case ZE_IMAGE_FORMAT_LAYOUT_32_32_32:
  case ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32:
    ZeImageFormatTypeSize = 32;
    break;
  default:
    logger::error(
        "ze2urImageFormat: unsupported image format layout: layout = {}",
        ZeImageFormat.layout);
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  ur_image_channel_order_t ChannelOrder;
  switch (ZeImageFormat.layout) {
  case ZE_IMAGE_FORMAT_LAYOUT_8:
  case ZE_IMAGE_FORMAT_LAYOUT_16:
  case ZE_IMAGE_FORMAT_LAYOUT_32:
    switch (ZeImageFormat.x) {
    case ZE_IMAGE_FORMAT_SWIZZLE_R:
      ChannelOrder = UR_IMAGE_CHANNEL_ORDER_R;
      break;
    case ZE_IMAGE_FORMAT_SWIZZLE_A:
      ChannelOrder = UR_IMAGE_CHANNEL_ORDER_A;
      break;
    default:
      logger::error(
          "ze2urImageFormat: unexpected image format channel x: x = {}",
          ZeImageFormat.x);
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
    break;
  case ZE_IMAGE_FORMAT_LAYOUT_8_8:
  case ZE_IMAGE_FORMAT_LAYOUT_16_16:
  case ZE_IMAGE_FORMAT_LAYOUT_32_32:
    if (ZeImageFormat.x != ZE_IMAGE_FORMAT_SWIZZLE_R) {
      logger::error(
          "ze2urImageFormat: unexpected image format channel x: x = {}",
          ZeImageFormat.x);
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
    switch (ZeImageFormat.y) {
    case ZE_IMAGE_FORMAT_SWIZZLE_G:
      ChannelOrder = UR_IMAGE_CHANNEL_ORDER_RG;
      break;
    case ZE_IMAGE_FORMAT_SWIZZLE_A:
      ChannelOrder = UR_IMAGE_CHANNEL_ORDER_RA;
      break;
    case ZE_IMAGE_FORMAT_SWIZZLE_X:
      ChannelOrder = UR_IMAGE_CHANNEL_ORDER_RX;
      break;
    default:
      logger::error(
          "ze2urImageFormat: unexpected image format channel y: y = {}\n",
          ZeImageFormat.y);
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
    break;
  case ZE_IMAGE_FORMAT_LAYOUT_8_8_8:
  case ZE_IMAGE_FORMAT_LAYOUT_16_16_16:
  case ZE_IMAGE_FORMAT_LAYOUT_32_32_32:
    if (ZeImageFormat.x == ZE_IMAGE_FORMAT_SWIZZLE_R &&
        ZeImageFormat.y == ZE_IMAGE_FORMAT_SWIZZLE_G) {
      switch (ZeImageFormat.z) {
      case ZE_IMAGE_FORMAT_SWIZZLE_B:
        ChannelOrder = UR_IMAGE_CHANNEL_ORDER_RGB;
        break;
      case ZE_IMAGE_FORMAT_SWIZZLE_X:
        ChannelOrder = UR_IMAGE_CHANNEL_ORDER_RGX;
        break;
      default:
        logger::error(
            "ze2urImageFormat: unexpected image format channel z: z = {}\n",
            ZeImageFormat.z);
        return UR_RESULT_ERROR_INVALID_VALUE;
      }
    } else {
      logger::error("ze2urImageFormat: unexpected image format channel");
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
    break;
  case ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8:
  case ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16:
  case ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32:
    if (ZeImageFormat.x == ZE_IMAGE_FORMAT_SWIZZLE_R &&
        ZeImageFormat.y == ZE_IMAGE_FORMAT_SWIZZLE_G &&
        ZeImageFormat.z == ZE_IMAGE_FORMAT_SWIZZLE_B) {
      switch (ZeImageFormat.w) {
      case ZE_IMAGE_FORMAT_SWIZZLE_X:
        ChannelOrder = UR_IMAGE_CHANNEL_ORDER_RGBX;
        break;
      case ZE_IMAGE_FORMAT_SWIZZLE_A:
        ChannelOrder = UR_IMAGE_CHANNEL_ORDER_RGBA;
        break;
      default:
        logger::error("ze2urImageFormat: unexpected image format channel w: w "
                      "= {}",
                      ZeImageFormat.x);
        return UR_RESULT_ERROR_INVALID_VALUE;
      }
    } else if (ZeImageFormat.x == ZE_IMAGE_FORMAT_SWIZZLE_A &&
               ZeImageFormat.y == ZE_IMAGE_FORMAT_SWIZZLE_R &&
               ZeImageFormat.z == ZE_IMAGE_FORMAT_SWIZZLE_G &&
               ZeImageFormat.w == ZE_IMAGE_FORMAT_SWIZZLE_B) {
      ChannelOrder = UR_IMAGE_CHANNEL_ORDER_ARGB;
    } else if (ZeImageFormat.x == ZE_IMAGE_FORMAT_SWIZZLE_B &&
               ZeImageFormat.y == ZE_IMAGE_FORMAT_SWIZZLE_G &&
               ZeImageFormat.z == ZE_IMAGE_FORMAT_SWIZZLE_R &&
               ZeImageFormat.w == ZE_IMAGE_FORMAT_SWIZZLE_A) {
      ChannelOrder = UR_IMAGE_CHANNEL_ORDER_BGRA;
    } else {
      logger::error("ze2urImageFormat: unexpected image format channel");
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
    break;
  default:
    logger::error(
        "ze2urImageFormat: unsupported image format layout: layout = {}",
        ZeImageFormat.layout);
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  ur_image_channel_type_t ChannelType;
  switch (ZeImageFormat.type) {
  case ZE_IMAGE_FORMAT_TYPE_UINT:
    switch (ZeImageFormatTypeSize) {
    case 8:
      ChannelType = UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8;
      break;
    case 16:
      ChannelType = UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16;
      break;
    case 32:
      ChannelType = UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32;
      break;
    default:
      logger::error("ze2urImageFormat: unexpected image format type size: size "
                    "= {}",
                    ZeImageFormatTypeSize);
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
    break;
  case ZE_IMAGE_FORMAT_TYPE_SINT:
    switch (ZeImageFormatTypeSize) {
    case 8:
      ChannelType = UR_IMAGE_CHANNEL_TYPE_SIGNED_INT8;
      break;
    case 16:
      ChannelType = UR_IMAGE_CHANNEL_TYPE_SIGNED_INT16;
      break;
    case 32:
      ChannelType = UR_IMAGE_CHANNEL_TYPE_SIGNED_INT32;
      break;
    default:
      logger::error("ze2urImageFormat: unexpected image format type size: size "
                    "= {}",
                    ZeImageFormatTypeSize);
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
    break;
  case ZE_IMAGE_FORMAT_TYPE_UNORM:
    switch (ZeImageFormatTypeSize) {
    case 8:
      ChannelType = UR_IMAGE_CHANNEL_TYPE_UNORM_INT8;
      break;
    case 16:
      ChannelType = UR_IMAGE_CHANNEL_TYPE_UNORM_INT16;
      break;
    default:
      logger::error("ze2urImageFormat: unexpected image format type size: size "
                    "= {}",
                    ZeImageFormatTypeSize);
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
    break;
  case ZE_IMAGE_FORMAT_TYPE_SNORM:
    switch (ZeImageFormatTypeSize) {
    case 8:
      ChannelType = UR_IMAGE_CHANNEL_TYPE_SNORM_INT8;
      break;
    case 16:
      ChannelType = UR_IMAGE_CHANNEL_TYPE_SNORM_INT16;
      break;
    default:
      logger::error("ze2urImageFormat: unexpected image format type size: size "
                    "= {}",
                    ZeImageFormatTypeSize);
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
    break;
  case ZE_IMAGE_FORMAT_TYPE_FLOAT:
    switch (ZeImageFormatTypeSize) {
    case 16:
      ChannelType = UR_IMAGE_CHANNEL_TYPE_HALF_FLOAT;
      break;
    case 32:
      ChannelType = UR_IMAGE_CHANNEL_TYPE_FLOAT;
      break;
    default:
      logger::error("ze2urImageFormat: unexpected image format type size: size "
                    "= {}",
                    ZeImageFormatTypeSize);
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
    break;
  default:
    logger::error("ze2urImageFormat: unsupported image format type: type = {}",
                  ZeImageFormat.type);
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  UrImageFormat->channelOrder = ChannelOrder;
  UrImageFormat->channelType = ChannelType;
  return UR_RESULT_SUCCESS;
}

/// Construct ZE image desc from UR image format and desc.
ur_result_t ur2zeImageDesc(const ur_image_format_t *ImageFormat,
                           const ur_image_desc_t *ImageDesc,
                           ZeStruct<ze_image_desc_t> &ZeImageDesc) {
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
      logger::error("ur2zeImageDesc: unexpected data type size");
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
      logger::error("ur2zeImageDesc: unexpected data type size");
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
      logger::error("ur2zeImageDesc: unexpected data type size");
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
      logger::error("ur2zeImageDesc: unexpected data type size");
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
    break;
  }
  default:
    logger::error("format channel order = {}", ImageFormat->channelOrder);
    die("ur2zeImageDesc: unsupported image channel order\n");
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
    logger::error("ur2zeImageDesc: unsupported image type");
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

  return UR_RESULT_SUCCESS;
}

/// Return element size in bytes of a pixel.
uint32_t getPixelSizeBytes(const ur_image_format_t *Format) {
  uint32_t NumChannels = 0;
  switch (Format->channelOrder) {
  case UR_IMAGE_CHANNEL_ORDER_A:
  case UR_IMAGE_CHANNEL_ORDER_R:
  case UR_IMAGE_CHANNEL_ORDER_INTENSITY:
  case UR_IMAGE_CHANNEL_ORDER_LUMINANCE:
  case UR_IMAGE_CHANNEL_ORDER_FORCE_UINT32:
    NumChannels = 1;
    break;
  case UR_IMAGE_CHANNEL_ORDER_RG:
  case UR_IMAGE_CHANNEL_ORDER_RA:
  case UR_IMAGE_CHANNEL_ORDER_RX:
    NumChannels = 2;
    break;
  case UR_IMAGE_CHANNEL_ORDER_RGB:
  case UR_IMAGE_CHANNEL_ORDER_RGX:
    NumChannels = 3;
    break;
  case UR_IMAGE_CHANNEL_ORDER_RGBA:
  case UR_IMAGE_CHANNEL_ORDER_BGRA:
  case UR_IMAGE_CHANNEL_ORDER_ARGB:
  case UR_IMAGE_CHANNEL_ORDER_ABGR:
  case UR_IMAGE_CHANNEL_ORDER_RGBX:
  case UR_IMAGE_CHANNEL_ORDER_SRGBA:
    NumChannels = 4;
    break;
  default:
    ur::unreachable();
  }
  uint32_t ChannelTypeSizeInBytes = 0;
  switch (Format->channelType) {
  case UR_IMAGE_CHANNEL_TYPE_SNORM_INT8:
  case UR_IMAGE_CHANNEL_TYPE_UNORM_INT8:
  case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT8:
  case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8:
    ChannelTypeSizeInBytes = 1;
    break;
  case UR_IMAGE_CHANNEL_TYPE_SNORM_INT16:
  case UR_IMAGE_CHANNEL_TYPE_UNORM_INT16:
  case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT16:
  case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16:
  case UR_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565:
  case UR_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555:
  case UR_IMAGE_CHANNEL_TYPE_HALF_FLOAT:
    ChannelTypeSizeInBytes = 2;
    break;
  case UR_IMAGE_CHANNEL_TYPE_INT_101010:
  case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT32:
  case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32:
  case UR_IMAGE_CHANNEL_TYPE_FLOAT:
  case UR_IMAGE_CHANNEL_TYPE_FORCE_UINT32:
    ChannelTypeSizeInBytes = 4;
    break;
  default:
    ur::unreachable();
  }
  return NumChannels * ChannelTypeSizeInBytes;
}

ur_result_t bindlessImagesCreateImpl(ur_context_handle_t hContext,
                                     ur_device_handle_t hDevice,
                                     ur_exp_image_mem_native_handle_t hImageMem,
                                     const ur_image_format_t *pImageFormat,
                                     const ur_image_desc_t *pImageDesc,
                                     ur_sampler_handle_t hSampler,
                                     ur_exp_image_native_handle_t *phImage) {
  std::shared_lock<ur_shared_mutex> Lock(hContext->Mutex);

  UR_ASSERT(hContext && hDevice && hImageMem,
            UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pImageFormat && pImageDesc && phImage,
            UR_RESULT_ERROR_INVALID_NULL_POINTER);

  ZeStruct<ze_image_desc_t> ZeImageDesc;
  UR_CALL(ur2zeImageDesc(pImageFormat, pImageDesc, ZeImageDesc));

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
    _ur_image *UrImage = reinterpret_cast<_ur_image *>(hImageMem);

    ZE2UR_CALL(zeImageViewCreateExt,
               (hContext->ZeContext, hDevice->ZeDevice, &ZeImageDesc,
                UrImage->ZeImage, &ZeImage));
    ZE2UR_CALL(zeContextMakeImageResident,
               (hContext->ZeContext, hDevice->ZeDevice, ZeImage));
  } else if (MemAllocProperties.type == ZE_MEMORY_TYPE_DEVICE ||
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

  return UR_RESULT_SUCCESS;
}

} // namespace

ur_result_t getImageRegionHelper(ze_image_desc_t ZeImageDesc,
                                 ur_rect_offset_t *Origin,
                                 ur_rect_region_t *Region,
                                 ze_image_region_t &ZeRegion) {
  UR_ASSERT(Origin, UR_RESULT_ERROR_INVALID_VALUE);
  UR_ASSERT(Region, UR_RESULT_ERROR_INVALID_VALUE);

  if (ZeImageDesc.type == ZE_IMAGE_TYPE_1D ||
      ZeImageDesc.type == ZE_IMAGE_TYPE_1DARRAY) {
    Region->height = 1;
    Region->depth = 1;
  } else if (ZeImageDesc.type == ZE_IMAGE_TYPE_2D ||
             ZeImageDesc.type == ZE_IMAGE_TYPE_2DARRAY) {
    Region->depth = 1;
  }

#ifndef NDEBUG
  UR_ASSERT((ZeImageDesc.type == ZE_IMAGE_TYPE_1D && Origin->y == 0 &&
             Origin->z == 0) ||
                (ZeImageDesc.type == ZE_IMAGE_TYPE_1DARRAY && Origin->z == 0) ||
                (ZeImageDesc.type == ZE_IMAGE_TYPE_2D && Origin->z == 0) ||
                (ZeImageDesc.type == ZE_IMAGE_TYPE_2DARRAY) ||
                (ZeImageDesc.type == ZE_IMAGE_TYPE_3D),
            UR_RESULT_ERROR_INVALID_VALUE);

  UR_ASSERT(Region->width && Region->height && Region->depth,
            UR_RESULT_ERROR_INVALID_VALUE);
  UR_ASSERT(
      (ZeImageDesc.type == ZE_IMAGE_TYPE_1D && Region->height == 1 &&
       Region->depth == 1) ||
          (ZeImageDesc.type == ZE_IMAGE_TYPE_1DARRAY && Region->depth == 1) ||
          (ZeImageDesc.type == ZE_IMAGE_TYPE_2D && Region->depth == 1) ||
          (ZeImageDesc.type == ZE_IMAGE_TYPE_2DARRAY) ||
          (ZeImageDesc.type == ZE_IMAGE_TYPE_3D),
      UR_RESULT_ERROR_INVALID_VALUE);
#endif // !NDEBUG

  uint32_t OriginX = ur_cast<uint32_t>(Origin->x);
  uint32_t OriginY = ur_cast<uint32_t>(Origin->y);
  uint32_t OriginZ = ur_cast<uint32_t>(Origin->z);

  uint32_t Width = ur_cast<uint32_t>(Region->width);
  uint32_t Height = (ZeImageDesc.type == ZE_IMAGE_TYPE_1DARRAY)
                        ? ZeImageDesc.arraylevels
                        : ur_cast<uint32_t>(Region->height);
  uint32_t Depth = (ZeImageDesc.type == ZE_IMAGE_TYPE_2DARRAY)
                       ? ZeImageDesc.arraylevels
                       : ur_cast<uint32_t>(Region->depth);

  ZeRegion = {OriginX, OriginY, OriginZ, Width, Height, Depth};

  return UR_RESULT_SUCCESS;
}

std::pair<ze_image_format_type_t, size_t>
getImageFormatTypeAndSize(const ur_image_format_t *ImageFormat) {
  ze_image_format_type_t ZeImageFormatType;
  size_t ZeImageFormatTypeSize;
  switch (ImageFormat->channelType) {
  case UR_IMAGE_CHANNEL_TYPE_FLOAT: {
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_FLOAT;
    ZeImageFormatTypeSize = 32;
    break;
  }
  case UR_IMAGE_CHANNEL_TYPE_HALF_FLOAT: {
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_FLOAT;
    ZeImageFormatTypeSize = 16;
    break;
  }
  case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32: {
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_UINT;
    ZeImageFormatTypeSize = 32;
    break;
  }
  case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16: {
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_UINT;
    ZeImageFormatTypeSize = 16;
    break;
  }
  case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8: {
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_UINT;
    ZeImageFormatTypeSize = 8;
    break;
  }
  case UR_IMAGE_CHANNEL_TYPE_UNORM_INT16: {
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_UNORM;
    ZeImageFormatTypeSize = 16;
    break;
  }
  case UR_IMAGE_CHANNEL_TYPE_UNORM_INT8: {
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_UNORM;
    ZeImageFormatTypeSize = 8;
    break;
  }
  case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT32: {
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_SINT;
    ZeImageFormatTypeSize = 32;
    break;
  }
  case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT16: {
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_SINT;
    ZeImageFormatTypeSize = 16;
    break;
  }
  case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT8: {
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_SINT;
    ZeImageFormatTypeSize = 8;
    break;
  }
  case UR_IMAGE_CHANNEL_TYPE_SNORM_INT16: {
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_SNORM;
    ZeImageFormatTypeSize = 16;
    break;
  }
  case UR_IMAGE_CHANNEL_TYPE_SNORM_INT8: {
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_SNORM;
    ZeImageFormatTypeSize = 8;
    break;
  }
  default:
    logger::error(
        "urMemImageCreate: unsupported image data type: data type = {}",
        ImageFormat->channelType);
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_FORCE_UINT32;
    ZeImageFormatTypeSize = 0;
  }
  return {ZeImageFormatType, ZeImageFormatTypeSize};
}

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
  UR_CALL(createUrMemFromZeImage(hContext, ZeImage, /*OwnZeMemHandle*/ true,
                                 ZeImageDesc, phImageMem));
  return UR_RESULT_SUCCESS;
}

ur_result_t
urBindlessImagesImageFreeExp(ur_context_handle_t hContext,
                             ur_device_handle_t hDevice,
                             ur_exp_image_mem_native_handle_t hImageMem) {
  std::ignore = hContext;
  std::ignore = hDevice;
  UR_CALL(ur::level_zero::urMemRelease(
      reinterpret_cast<ur_mem_handle_t>(hImageMem)));
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

ur_result_t urBindlessImagesImageCopyExp(
    ur_queue_handle_t hQueue, [[maybe_unused]] const void *pSrc,
    [[maybe_unused]] void *pDst,
    [[maybe_unused]] const ur_image_desc_t *pSrcImageDesc,
    [[maybe_unused]] const ur_image_desc_t *pDstImageDesc,
    [[maybe_unused]] const ur_image_format_t *pSrcImageFormat,
    [[maybe_unused]] const ur_image_format_t *pDstImageFormat,
    [[maybe_unused]] ur_exp_image_copy_region_t *pCopyRegion,
    [[maybe_unused]] ur_exp_image_copy_flags_t imageCopyFlags,
    [[maybe_unused]] uint32_t numEventsInWaitList,
    [[maybe_unused]] const ur_event_handle_t *phEventWaitList,
    [[maybe_unused]] ur_event_handle_t *phEvent) {
  std::scoped_lock<ur_shared_mutex> Lock(hQueue->Mutex);

  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pDst && pSrc && pSrcImageFormat && pSrcImageDesc && pDstImageDesc &&
                pCopyRegion,
            UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(pSrcImageDesc->type == pDstImageDesc->type,
            UR_RESULT_ERROR_INVALID_VALUE);
  UR_ASSERT(!(UR_EXP_IMAGE_COPY_FLAGS_MASK & imageCopyFlags),
            UR_RESULT_ERROR_INVALID_ENUMERATION);
  UR_ASSERT(!(pSrcImageDesc && UR_MEM_TYPE_IMAGE1D_ARRAY < pSrcImageDesc->type),
            UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR);

  ZeStruct<ze_image_desc_t> ZeImageDesc;
  UR_CALL(ur2zeImageDesc(pSrcImageFormat, pSrcImageDesc, ZeImageDesc));

  bool UseCopyEngine = hQueue->useCopyEngine(/*PreferCopyEngine*/ true);

  _ur_ze_event_list_t TmpWaitList;
  UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
      numEventsInWaitList, phEventWaitList, hQueue, UseCopyEngine));

  bool Blocking = false;
  // We want to batch these commands to avoid extra submissions (costly)
  bool OkToBatch = true;

  // Get a new command list to be used on this call
  ur_command_list_ptr_t CommandList{};
  UR_CALL(hQueue->Context->getAvailableCommandList(
      hQueue, CommandList, UseCopyEngine, numEventsInWaitList, phEventWaitList,
      OkToBatch, nullptr /*ForcedCmdQueue*/));

  ze_event_handle_t ZeEvent = nullptr;
  ur_event_handle_t InternalEvent;
  bool IsInternal = phEvent == nullptr;
  ur_event_handle_t *Event = phEvent ? phEvent : &InternalEvent;
  UR_CALL(createEventAndAssociateQueue(hQueue, Event, UR_COMMAND_MEM_IMAGE_COPY,
                                       CommandList, IsInternal,
                                       /*IsMultiDevice*/ false));
  UR_CALL(setSignalEvent(hQueue, UseCopyEngine, &ZeEvent, Event,
                         numEventsInWaitList, phEventWaitList,
                         CommandList->second.ZeQueue));
  (*Event)->WaitList = TmpWaitList;

  const auto &ZeCommandList = CommandList->first;
  const auto &WaitList = (*Event)->WaitList;

  if (imageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_HOST_TO_DEVICE) {
    uint32_t SrcRowPitch =
        pSrcImageDesc->width * getPixelSizeBytes(pSrcImageFormat);
    uint32_t SrcSlicePitch = SrcRowPitch * pSrcImageDesc->height;
    if (pDstImageDesc->rowPitch == 0) {
      // Copy to Non-USM memory
      ze_image_region_t DstRegion;
      UR_CALL(getImageRegionHelper(ZeImageDesc, &pCopyRegion->dstOffset,
                                   &pCopyRegion->copyExtent, DstRegion));
      auto *UrImage = static_cast<_ur_image *>(pDst);
      const char *SrcPtr =
          static_cast<const char *>(pSrc) +
          pCopyRegion->srcOffset.z * SrcSlicePitch +
          pCopyRegion->srcOffset.y * SrcRowPitch +
          pCopyRegion->srcOffset.x * getPixelSizeBytes(pSrcImageFormat);
      ZE2UR_CALL(zeCommandListAppendImageCopyFromMemoryExt,
                 (ZeCommandList, UrImage->ZeImage, SrcPtr, &DstRegion,
                  SrcRowPitch, SrcSlicePitch, ZeEvent, WaitList.Length,
                  WaitList.ZeEventList));
    } else {
      // Copy to pitched USM memory
      uint32_t DstRowPitch = pDstImageDesc->rowPitch;
      ze_copy_region_t ZeDstRegion = {(uint32_t)pCopyRegion->dstOffset.x,
                                      (uint32_t)pCopyRegion->dstOffset.y,
                                      (uint32_t)pCopyRegion->dstOffset.z,
                                      DstRowPitch,
                                      (uint32_t)pCopyRegion->copyExtent.height,
                                      (uint32_t)pCopyRegion->copyExtent.depth};
      uint32_t DstSlicePitch = 0;
      ze_copy_region_t ZeSrcRegion = {(uint32_t)pCopyRegion->srcOffset.x,
                                      (uint32_t)pCopyRegion->srcOffset.y,
                                      (uint32_t)pCopyRegion->srcOffset.z,
                                      SrcRowPitch,
                                      (uint32_t)pCopyRegion->copyExtent.height,
                                      (uint32_t)pCopyRegion->copyExtent.depth};
      ZE2UR_CALL(zeCommandListAppendMemoryCopyRegion,
                 (ZeCommandList, pDst, &ZeDstRegion, DstRowPitch, DstSlicePitch,
                  pSrc, &ZeSrcRegion, SrcRowPitch, SrcSlicePitch, ZeEvent,
                  WaitList.Length, WaitList.ZeEventList));
    }
  } else if (imageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_HOST) {
    uint32_t DstRowPitch =
        pDstImageDesc->width * getPixelSizeBytes(pDstImageFormat);
    uint32_t DstSlicePitch = DstRowPitch * pDstImageDesc->height;
    if (pSrcImageDesc->rowPitch == 0) {
      // Copy from Non-USM memory to host
      ze_image_region_t SrcRegion;
      UR_CALL(getImageRegionHelper(ZeImageDesc, &pCopyRegion->srcOffset,
                                   &pCopyRegion->copyExtent, SrcRegion));
      auto *UrImage = static_cast<const _ur_image *>(pSrc);
      char *DstPtr =
          static_cast<char *>(pDst) + pCopyRegion->dstOffset.z * DstSlicePitch +
          pCopyRegion->dstOffset.y * DstRowPitch +
          pCopyRegion->dstOffset.x * getPixelSizeBytes(pDstImageFormat);
      ZE2UR_CALL(zeCommandListAppendImageCopyToMemoryExt,
                 (ZeCommandList, DstPtr, UrImage->ZeImage, &SrcRegion,
                  DstRowPitch, DstSlicePitch, ZeEvent, WaitList.Length,
                  WaitList.ZeEventList));
    } else {
      // Copy from pitched USM memory to host
      ze_copy_region_t ZeDstRegion = {(uint32_t)pCopyRegion->dstOffset.x,
                                      (uint32_t)pCopyRegion->dstOffset.y,
                                      (uint32_t)pCopyRegion->dstOffset.z,
                                      DstRowPitch,
                                      (uint32_t)pCopyRegion->copyExtent.height,
                                      (uint32_t)pCopyRegion->copyExtent.depth};
      uint32_t SrcRowPitch = pSrcImageDesc->rowPitch;
      ze_copy_region_t ZeSrcRegion = {(uint32_t)pCopyRegion->srcOffset.x,
                                      (uint32_t)pCopyRegion->srcOffset.y,
                                      (uint32_t)pCopyRegion->srcOffset.z,
                                      SrcRowPitch,
                                      (uint32_t)pCopyRegion->copyExtent.height,
                                      (uint32_t)pCopyRegion->copyExtent.depth};
      uint32_t SrcSlicePitch = 0;
      ZE2UR_CALL(zeCommandListAppendMemoryCopyRegion,
                 (ZeCommandList, pDst, &ZeDstRegion, DstRowPitch, DstSlicePitch,
                  pSrc, &ZeSrcRegion, SrcRowPitch, SrcSlicePitch, ZeEvent,
                  WaitList.Length, WaitList.ZeEventList));
    }
  } else if (imageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_DEVICE) {
    ze_image_region_t DstRegion;
    UR_CALL(getImageRegionHelper(ZeImageDesc, &pCopyRegion->dstOffset,
                                 &pCopyRegion->copyExtent, DstRegion));
    ze_image_region_t SrcRegion;
    UR_CALL(getImageRegionHelper(ZeImageDesc, &pCopyRegion->srcOffset,
                                 &pCopyRegion->copyExtent, SrcRegion));
    auto *UrImageDst = static_cast<_ur_image *>(pDst);
    auto *UrImageSrc = static_cast<const _ur_image *>(pSrc);
    ZE2UR_CALL(zeCommandListAppendImageCopyRegion,
               (ZeCommandList, UrImageDst->ZeImage, UrImageSrc->ZeImage,
                &DstRegion, &SrcRegion, ZeEvent, WaitList.Length,
                WaitList.ZeEventList));
  } else {
    logger::error("urBindlessImagesImageCopyExp: unexpected imageCopyFlags");
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  UR_CALL(hQueue->executeCommandList(CommandList, Blocking, OkToBatch));

  return UR_RESULT_SUCCESS;
}

ur_result_t urBindlessImagesImageGetInfoExp(
    ur_context_handle_t, ur_exp_image_mem_native_handle_t hImageMem,
    ur_image_info_t propName, void *pPropValue, size_t *pPropSizeRet) {
  UR_ASSERT(hImageMem, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(UR_IMAGE_INFO_DEPTH >= propName,
            UR_RESULT_ERROR_INVALID_ENUMERATION);
  UR_ASSERT(pPropValue || pPropSizeRet, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  auto *UrImage = reinterpret_cast<_ur_image *>(hImageMem);
  ze_image_desc_t &Desc = UrImage->ZeImageDesc;
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

ur_result_t
urBindlessImagesMipmapFreeExp(ur_context_handle_t hContext,
                              ur_device_handle_t hDevice,
                              ur_exp_image_mem_native_handle_t hMem) {
  return ur::level_zero::urBindlessImagesImageFreeExp(hContext, hDevice, hMem);
}

ur_result_t urBindlessImagesImportExternalMemoryExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, size_t size,
    ur_exp_external_mem_type_t memHandleType,
    ur_exp_external_mem_desc_t *pExternalMemDesc,
    ur_exp_external_mem_handle_t *phExternalMem) {

  UR_ASSERT(hContext && hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pExternalMemDesc && phExternalMem,
            UR_RESULT_ERROR_INVALID_NULL_POINTER);

  struct ur_ze_external_memory_data *externalMemoryData =
      new struct ur_ze_external_memory_data;

  void *pNext = const_cast<void *>(pExternalMemDesc->pNext);
  while (pNext != nullptr) {
    const ur_base_desc_t *BaseDesc = static_cast<const ur_base_desc_t *>(pNext);
    if (BaseDesc->stype == UR_STRUCTURE_TYPE_EXP_FILE_DESCRIPTOR) {
      ze_external_memory_import_fd_t *importFd =
          new ze_external_memory_import_fd_t;
      importFd->stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD;
      importFd->pNext = nullptr;
      auto FileDescriptor =
          static_cast<const ur_exp_file_descriptor_t *>(pNext);
      importFd->fd = FileDescriptor->fd;
      importFd->flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD;
      externalMemoryData->importExtensionDesc = importFd;
      externalMemoryData->type = UR_ZE_EXTERNAL_OPAQUE_FD;
    } else if (BaseDesc->stype == UR_STRUCTURE_TYPE_EXP_WIN32_HANDLE) {
      ze_external_memory_import_win32_handle_t *importWin32 =
          new ze_external_memory_import_win32_handle_t;
      importWin32->stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_WIN32;
      importWin32->pNext = nullptr;
      auto Win32Handle = static_cast<const ur_exp_win32_handle_t *>(pNext);

      switch (memHandleType) {
      case UR_EXP_EXTERNAL_MEM_TYPE_WIN32_NT:
        importWin32->flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32;
        break;
      case UR_EXP_EXTERNAL_MEM_TYPE_WIN32_NT_DX12_RESOURCE:
        importWin32->flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D12_RESOURCE;
        break;
      case UR_EXP_EXTERNAL_MEM_TYPE_OPAQUE_FD:
      default:
        delete importWin32;
        delete externalMemoryData;
        return UR_RESULT_ERROR_INVALID_VALUE;
      }
      importWin32->handle = Win32Handle->handle;
      externalMemoryData->importExtensionDesc = importWin32;
      externalMemoryData->type = UR_ZE_EXTERNAL_WIN32;
    }
    pNext = const_cast<void *>(BaseDesc->pNext);
  }
  externalMemoryData->size = size;

  *phExternalMem =
      reinterpret_cast<ur_exp_external_mem_handle_t>(externalMemoryData);
  return UR_RESULT_SUCCESS;
}

ur_result_t urBindlessImagesMapExternalArrayExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    ur_exp_external_mem_handle_t hExternalMem,
    ur_exp_image_mem_native_handle_t *phImageMem) {

  UR_ASSERT(hContext && hDevice && hExternalMem,
            UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pImageFormat && pImageDesc, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  struct ur_ze_external_memory_data *externalMemoryData =
      reinterpret_cast<ur_ze_external_memory_data *>(hExternalMem);

  ze_image_bindless_exp_desc_t ZeImageBindlessDesc = {};
  ZeImageBindlessDesc.stype = ZE_STRUCTURE_TYPE_BINDLESS_IMAGE_EXP_DESC;

  ZeStruct<ze_image_desc_t> ZeImageDesc;
  UR_CALL(ur2zeImageDesc(pImageFormat, pImageDesc, ZeImageDesc));

  ZeImageBindlessDesc.pNext = externalMemoryData->importExtensionDesc;
  ZeImageBindlessDesc.flags = ZE_IMAGE_BINDLESS_EXP_FLAG_BINDLESS;
  ZeImageDesc.pNext = &ZeImageBindlessDesc;

  ze_image_handle_t ZeImage;
  ZE2UR_CALL(zeImageCreate,
             (hContext->ZeContext, hDevice->ZeDevice, &ZeImageDesc, &ZeImage));
  ZE2UR_CALL(zeContextMakeImageResident,
             (hContext->ZeContext, hDevice->ZeDevice, ZeImage));
  UR_CALL(createUrMemFromZeImage(hContext, ZeImage, /*OwnZeMemHandle*/ true,
                                 ZeImageDesc, phImageMem));
  externalMemoryData->urMemoryHandle =
      reinterpret_cast<ur_mem_handle_t>(*phImageMem);
  return UR_RESULT_SUCCESS;
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
  logger::error("[UR][L0] {} function not implemented!",
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urBindlessImagesReleaseExternalMemoryExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_external_mem_handle_t hExternalMem) {

  UR_ASSERT(hContext && hDevice && hExternalMem,
            UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  struct ur_ze_external_memory_data *externalMemoryData =
      reinterpret_cast<ur_ze_external_memory_data *>(hExternalMem);

  UR_CALL(ur::level_zero::urMemRelease(externalMemoryData->urMemoryHandle));

  switch (externalMemoryData->type) {
  case UR_ZE_EXTERNAL_OPAQUE_FD:
    delete (reinterpret_cast<ze_external_memory_import_fd_t *>(
        externalMemoryData->importExtensionDesc));
    break;
  case UR_ZE_EXTERNAL_WIN32:
    delete (reinterpret_cast<ze_external_memory_import_win32_handle_t *>(
        externalMemoryData->importExtensionDesc));
    break;
  default:
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  delete (externalMemoryData);

  return UR_RESULT_SUCCESS;
}

ur_result_t urBindlessImagesImportExternalSemaphoreExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_external_semaphore_type_t semHandleType,
    ur_exp_external_semaphore_desc_t *pExternalSemaphoreDesc,
    ur_exp_external_semaphore_handle_t *phExternalSemaphoreHandle) {

  auto UrPlatform = hContext->getPlatform();
  if (UrPlatform->ZeExternalSemaphoreExt.Supported == false) {
    logger::error(logger::LegacyMessage("[UR][L0] "),
                  " {} function not supported!", __FUNCTION__);
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }
  ze_intel_external_semaphore_exp_desc_t SemDesc = {
      ZE_INTEL_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_EXP_DESC, nullptr,
      ZE_EXTERNAL_SEMAPHORE_EXP_FLAGS_OPAQUE_FD};
  ze_intel_external_semaphore_exp_handle_t ExtSemaphoreHandle;
  ze_intel_external_semaphore_desc_fd_exp_desc_t FDExpDesc = {
      ZE_INTEL_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_FD_EXP_DESC, nullptr, 0};
  _ze_intel_external_semaphore_win32_exp_desc_t Win32ExpDesc = {
      ZE_INTEL_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_WIN32_EXP_DESC, nullptr,
      nullptr, nullptr};
  void *pNext = const_cast<void *>(pExternalSemaphoreDesc->pNext);
  while (pNext != nullptr) {
    const ur_base_desc_t *BaseDesc = static_cast<const ur_base_desc_t *>(pNext);
    if (BaseDesc->stype == UR_STRUCTURE_TYPE_EXP_FILE_DESCRIPTOR) {
      auto FileDescriptor =
          static_cast<const ur_exp_file_descriptor_t *>(pNext);
      FDExpDesc.fd = FileDescriptor->fd;
      SemDesc.pNext = &FDExpDesc;
      SemDesc.flags = ZE_EXTERNAL_SEMAPHORE_EXP_FLAGS_OPAQUE_FD;
    } else if (BaseDesc->stype == UR_STRUCTURE_TYPE_EXP_WIN32_HANDLE) {
      SemDesc.pNext = &Win32ExpDesc;
      auto Win32Handle = static_cast<const ur_exp_win32_handle_t *>(pNext);
      switch (semHandleType) {
      case UR_EXP_EXTERNAL_SEMAPHORE_TYPE_WIN32_NT:
        SemDesc.flags = ZE_EXTERNAL_SEMAPHORE_EXP_FLAGS_OPAQUE_WIN32;
        break;
      case UR_EXP_EXTERNAL_SEMAPHORE_TYPE_WIN32_NT_DX12_FENCE:
        SemDesc.flags = ZE_EXTERNAL_SEMAPHORE_EXP_FLAGS_D3D12_FENCE;
        break;
      case UR_EXP_EXTERNAL_SEMAPHORE_TYPE_OPAQUE_FD:
        SemDesc.flags = ZE_EXTERNAL_SEMAPHORE_EXP_FLAGS_OPAQUE_FD;
        break;
      default:
        return UR_RESULT_ERROR_INVALID_VALUE;
      }
      Win32ExpDesc.handle = Win32Handle->handle;
    }
    pNext = const_cast<void *>(BaseDesc->pNext);
  }

  ZE2UR_CALL(UrPlatform->ZeExternalSemaphoreExt.zexImportExternalSemaphoreExp,
             (hDevice->ZeDevice, &SemDesc, &ExtSemaphoreHandle));
  *phExternalSemaphoreHandle =
      (ur_exp_external_semaphore_handle_t)ExtSemaphoreHandle;

  return UR_RESULT_SUCCESS;
}

ur_result_t urBindlessImagesReleaseExternalSemaphoreExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_external_semaphore_handle_t hExternalSemaphore) {
  std::ignore = hDevice;
  auto UrPlatform = hContext->getPlatform();
  if (UrPlatform->ZeExternalSemaphoreExt.Supported == false) {
    logger::error(logger::LegacyMessage("[UR][L0] "),
                  " {} function not supported!", __FUNCTION__);
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }
  ZE2UR_CALL(
      UrPlatform->ZeExternalSemaphoreExt.zexDeviceReleaseExternalSemaphoreExp,
      ((ze_intel_external_semaphore_exp_handle_t)hExternalSemaphore));

  return UR_RESULT_SUCCESS;
}

ur_result_t urBindlessImagesWaitExternalSemaphoreExp(
    ur_queue_handle_t hQueue, ur_exp_external_semaphore_handle_t hSemaphore,
    bool hasValue, uint64_t waitValue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  auto UrPlatform = hQueue->Context->getPlatform();
  if (UrPlatform->ZeExternalSemaphoreExt.Supported == false) {
    logger::error(logger::LegacyMessage("[UR][L0] "),
                  " {} function not supported!", __FUNCTION__);
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  bool UseCopyEngine = false;

  // We want to batch these commands to avoid extra submissions (costly)
  bool OkToBatch = true;

  _ur_ze_event_list_t TmpWaitList;
  UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
      numEventsInWaitList, phEventWaitList, hQueue, UseCopyEngine));

  // Get a new command list to be used on this call
  ur_command_list_ptr_t CommandList{};
  UR_CALL(hQueue->Context->getAvailableCommandList(
      hQueue, CommandList, UseCopyEngine, numEventsInWaitList, phEventWaitList,
      OkToBatch, nullptr /*ForcedCmdQueue*/));

  ze_event_handle_t ZeEvent = nullptr;
  ur_event_handle_t InternalEvent;
  bool IsInternal = phEvent == nullptr;
  ur_event_handle_t *Event = phEvent ? phEvent : &InternalEvent;
  UR_CALL(createEventAndAssociateQueue(hQueue, Event,
                                       UR_COMMAND_EXTERNAL_SEMAPHORE_WAIT_EXP,
                                       CommandList, IsInternal,
                                       /*IsMultiDevice*/ false));
  UR_CALL(setSignalEvent(hQueue, UseCopyEngine, &ZeEvent, Event,
                         numEventsInWaitList, phEventWaitList,
                         CommandList->second.ZeQueue));
  (*Event)->WaitList = TmpWaitList;

  const auto &ZeCommandList = CommandList->first;
  const auto &WaitList = (*Event)->WaitList;

  ze_intel_external_semaphore_wait_exp_params_t WaitParams = {
      ZE_INTEL_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_WAIT_PARAMS_EXP, nullptr, 0};
  WaitParams.value = hasValue ? waitValue : 0;
  const ze_intel_external_semaphore_exp_handle_t hExtSemaphore =
      reinterpret_cast<ze_intel_external_semaphore_exp_handle_t>(hSemaphore);
  ZE2UR_CALL(UrPlatform->ZeExternalSemaphoreExt
                 .zexCommandListAppendWaitExternalSemaphoresExp,
             (ZeCommandList, 1, &hExtSemaphore, &WaitParams, ZeEvent,
              WaitList.Length, WaitList.ZeEventList));

  return UR_RESULT_SUCCESS;
}

ur_result_t urBindlessImagesSignalExternalSemaphoreExp(
    ur_queue_handle_t hQueue, ur_exp_external_semaphore_handle_t hSemaphore,
    bool hasValue, uint64_t signalValue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = hSemaphore;
  std::ignore = hasValue;
  std::ignore = signalValue;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;
  auto UrPlatform = hQueue->Context->getPlatform();
  if (UrPlatform->ZeExternalSemaphoreExt.Supported == false) {
    logger::error(logger::LegacyMessage("[UR][L0] "),
                  " {} function not supported!", __FUNCTION__);
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  bool UseCopyEngine = false;

  // We want to batch these commands to avoid extra submissions (costly)
  bool OkToBatch = true;

  _ur_ze_event_list_t TmpWaitList;
  UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
      numEventsInWaitList, phEventWaitList, hQueue, UseCopyEngine));

  // Get a new command list to be used on this call
  ur_command_list_ptr_t CommandList{};
  UR_CALL(hQueue->Context->getAvailableCommandList(
      hQueue, CommandList, UseCopyEngine, numEventsInWaitList, phEventWaitList,
      OkToBatch, nullptr /*ForcedCmdQueue*/));

  ze_event_handle_t ZeEvent = nullptr;
  ur_event_handle_t InternalEvent;
  bool IsInternal = phEvent == nullptr;
  ur_event_handle_t *Event = phEvent ? phEvent : &InternalEvent;
  UR_CALL(createEventAndAssociateQueue(hQueue, Event,
                                       UR_COMMAND_EXTERNAL_SEMAPHORE_SIGNAL_EXP,
                                       CommandList, IsInternal,
                                       /*IsMultiDevice*/ false));
  UR_CALL(setSignalEvent(hQueue, UseCopyEngine, &ZeEvent, Event,
                         numEventsInWaitList, phEventWaitList,
                         CommandList->second.ZeQueue));
  (*Event)->WaitList = TmpWaitList;

  const auto &ZeCommandList = CommandList->first;
  const auto &WaitList = (*Event)->WaitList;

  ze_intel_external_semaphore_signal_exp_params_t SignalParams = {
      ZE_INTEL_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_EXP, nullptr, 0};
  SignalParams.value = hasValue ? signalValue : 0;
  const ze_intel_external_semaphore_exp_handle_t hExtSemaphore =
      reinterpret_cast<ze_intel_external_semaphore_exp_handle_t>(hSemaphore);

  ZE2UR_CALL(UrPlatform->ZeExternalSemaphoreExt
                 .zexCommandListAppendSignalExternalSemaphoresExp,
             (ZeCommandList, 1, &hExtSemaphore, &SignalParams, ZeEvent,
              WaitList.Length, WaitList.ZeEventList));

  return UR_RESULT_SUCCESS;
}

} // namespace ur::level_zero
