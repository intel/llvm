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
#include "ur_level_zero.hpp"

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
  case ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8:
    ZeImageFormatTypeSize = 8;
    break;
  case ZE_IMAGE_FORMAT_LAYOUT_16:
  case ZE_IMAGE_FORMAT_LAYOUT_16_16:
  case ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16:
    ZeImageFormatTypeSize = 16;
    break;
  case ZE_IMAGE_FORMAT_LAYOUT_32:
  case ZE_IMAGE_FORMAT_LAYOUT_32_32:
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
          ZeImageFormat.x);
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
                                     ur_exp_image_mem_handle_t hImageMem,
                                     const ur_image_format_t *pImageFormat,
                                     const ur_image_desc_t *pImageDesc,
                                     ur_sampler_handle_t hSampler,
                                     ur_exp_image_handle_t *phImage) {
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
      ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES};
  ZE2UR_CALL(zeMemGetAllocProperties,
             (hContext->ZeContext, hImageMem, &MemAllocProperties, nullptr));
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
    PitchedDesc.ptr = hImageMem;
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
  *phImage = reinterpret_cast<ur_exp_image_handle_t>(DeviceOffset);

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
    ur::unreachable();
  }
  return {ZeImageFormatType, ZeImageFormatTypeSize};
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPitchedAllocExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_usm_desc_t *pUSMDesc, ur_usm_pool_handle_t pool,
    size_t widthInBytes, size_t height, size_t elementSizeBytes, void **ppMem,
    size_t *pResultPitch) {
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
  UR_CALL(urUSMDeviceAlloc(hContext, hDevice, pUSMDesc, pool, Size, ppMem));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesUnsampledImageHandleDestroyExp(ur_context_handle_t hContext,
                                               ur_device_handle_t hDevice,
                                               ur_exp_image_handle_t hImage) {
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = hImage;

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesSampledImageHandleDestroyExp(ur_context_handle_t hContext,
                                             ur_device_handle_t hDevice,
                                             ur_exp_image_handle_t hImage) {
  // Sampled image is a combination of unsampled image and sampler.
  // Sampler is released in urSamplerRelease.
  return urBindlessImagesUnsampledImageHandleDestroyExp(hContext, hDevice,
                                                        hImage);
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImageAllocateExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    ur_exp_image_mem_handle_t *phImageMem) {
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

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImageFreeExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_mem_handle_t hImageMem) {
  std::ignore = hContext;
  std::ignore = hDevice;
  UR_CALL(urMemRelease(reinterpret_cast<ur_mem_handle_t>(hImageMem)));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesUnsampledImageCreateExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_mem_handle_t hImageMem, const ur_image_format_t *pImageFormat,
    const ur_image_desc_t *pImageDesc, ur_exp_image_handle_t *phImage) {
  UR_CALL(bindlessImagesCreateImpl(hContext, hDevice, hImageMem, pImageFormat,
                                   pImageDesc, nullptr, phImage));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesSampledImageCreateExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_mem_handle_t hImageMem, const ur_image_format_t *pImageFormat,
    const ur_image_desc_t *pImageDesc, ur_sampler_handle_t hSampler,
    ur_exp_image_handle_t *phImage) {
  UR_CALL(bindlessImagesCreateImpl(hContext, hDevice, hImageMem, pImageFormat,
                                   pImageDesc, hSampler, phImage));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImageCopyExp(
    ur_queue_handle_t hUrQueue, void *pDst, void *pSrc,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    ur_exp_image_copy_flags_t imageCopyFlags, ur_rect_offset_t srcOffset,
    ur_rect_offset_t dstOffset, ur_rect_region_t copyExtent,
    ur_rect_region_t hostExtent, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  auto hQueue = Legacy(hUrQueue);
  std::scoped_lock<ur_shared_mutex> Lock(hQueue->Mutex);

  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pDst && pSrc && pImageFormat && pImageDesc,
            UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(!(UR_EXP_IMAGE_COPY_FLAGS_MASK & imageCopyFlags),
            UR_RESULT_ERROR_INVALID_ENUMERATION);
  UR_ASSERT(!(pImageDesc && UR_MEM_TYPE_IMAGE1D_ARRAY < pImageDesc->type),
            UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR);

  ZeStruct<ze_image_desc_t> ZeImageDesc;
  UR_CALL(ur2zeImageDesc(pImageFormat, pImageDesc, ZeImageDesc));

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
      OkToBatch));

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

  uint32_t PixelSizeInBytes = getPixelSizeBytes(pImageFormat);

  if (imageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_HOST_TO_DEVICE) {
    uint32_t SrcRowPitch = hostExtent.width * PixelSizeInBytes;
    uint32_t SrcSlicePitch = SrcRowPitch * hostExtent.height;
    if (pImageDesc->rowPitch == 0) {
      // Copy to Non-USM memory
      ze_image_region_t DstRegion;
      UR_CALL(getImageRegionHelper(ZeImageDesc, &dstOffset, &copyExtent,
                                   DstRegion));
      auto *UrImage = static_cast<_ur_image *>(pDst);
      char *SrcPtr = static_cast<char *>(pSrc) + srcOffset.z * SrcSlicePitch +
                     srcOffset.y * SrcRowPitch + srcOffset.x * PixelSizeInBytes;
      ZE2UR_CALL(zeCommandListAppendImageCopyFromMemoryExt,
                 (ZeCommandList, UrImage->ZeImage, SrcPtr, &DstRegion,
                  SrcRowPitch, SrcSlicePitch, ZeEvent, WaitList.Length,
                  WaitList.ZeEventList));
    } else {
      // Copy to pitched USM memory
      uint32_t DstRowPitch = pImageDesc->rowPitch;
      ze_copy_region_t ZeDstRegion = {
          (uint32_t)dstOffset.x,       (uint32_t)dstOffset.y,
          (uint32_t)dstOffset.z,       DstRowPitch,
          (uint32_t)copyExtent.height, (uint32_t)copyExtent.depth};
      uint32_t DstSlicePitch = 0;
      ze_copy_region_t ZeSrcRegion = {
          (uint32_t)srcOffset.x,       (uint32_t)srcOffset.y,
          (uint32_t)srcOffset.z,       SrcRowPitch,
          (uint32_t)copyExtent.height, (uint32_t)copyExtent.depth};
      ZE2UR_CALL(zeCommandListAppendMemoryCopyRegion,
                 (ZeCommandList, pDst, &ZeDstRegion, DstRowPitch, DstSlicePitch,
                  pSrc, &ZeSrcRegion, SrcRowPitch, SrcSlicePitch, ZeEvent,
                  WaitList.Length, WaitList.ZeEventList));
    }
  } else if (imageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_HOST) {
    uint32_t DstRowPitch = hostExtent.width * PixelSizeInBytes;
    uint32_t DstSlicePitch = DstRowPitch * hostExtent.height;
    if (pImageDesc->rowPitch == 0) {
      // Copy from Non-USM memory to host
      ze_image_region_t SrcRegion;
      UR_CALL(getImageRegionHelper(ZeImageDesc, &srcOffset, &copyExtent,
                                   SrcRegion));
      auto *UrImage = static_cast<_ur_image *>(pSrc);
      char *DstPtr = static_cast<char *>(pDst) + dstOffset.z * DstSlicePitch +
                     dstOffset.y * DstRowPitch + dstOffset.x * PixelSizeInBytes;
      ZE2UR_CALL(zeCommandListAppendImageCopyToMemoryExt,
                 (ZeCommandList, DstPtr, UrImage->ZeImage, &SrcRegion,
                  DstRowPitch, DstSlicePitch, ZeEvent, WaitList.Length,
                  WaitList.ZeEventList));
    } else {
      // Copy from pitched USM memory to host
      ze_copy_region_t ZeDstRegion = {
          (uint32_t)dstOffset.x,       (uint32_t)dstOffset.y,
          (uint32_t)dstOffset.z,       DstRowPitch,
          (uint32_t)copyExtent.height, (uint32_t)copyExtent.depth};
      uint32_t SrcRowPitch = pImageDesc->rowPitch;
      ze_copy_region_t ZeSrcRegion = {
          (uint32_t)srcOffset.x,       (uint32_t)srcOffset.y,
          (uint32_t)srcOffset.z,       SrcRowPitch,
          (uint32_t)copyExtent.height, (uint32_t)copyExtent.depth};
      uint32_t SrcSlicePitch = 0;
      ZE2UR_CALL(zeCommandListAppendMemoryCopyRegion,
                 (ZeCommandList, pDst, &ZeDstRegion, DstRowPitch, DstSlicePitch,
                  pSrc, &ZeSrcRegion, SrcRowPitch, SrcSlicePitch, ZeEvent,
                  WaitList.Length, WaitList.ZeEventList));
    }
  } else if (imageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_DEVICE) {
    ze_image_region_t DstRegion;
    UR_CALL(
        getImageRegionHelper(ZeImageDesc, &dstOffset, &copyExtent, DstRegion));
    ze_image_region_t SrcRegion;
    UR_CALL(
        getImageRegionHelper(ZeImageDesc, &srcOffset, &copyExtent, SrcRegion));
    auto *UrImageDst = static_cast<_ur_image *>(pDst);
    auto *UrImageSrc = static_cast<_ur_image *>(pSrc);
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

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImageGetInfoExp(
    ur_exp_image_mem_handle_t hImageMem, ur_image_info_t propName,
    void *pPropValue, size_t *pPropSizeRet) {
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

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesMipmapGetLevelExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_mem_handle_t hImageMem, uint32_t mipmapLevel,
    ur_exp_image_mem_handle_t *phImageMem) {
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = hImageMem;
  std::ignore = mipmapLevel;
  std::ignore = phImageMem;
  logger::error(logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesMipmapFreeExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_mem_handle_t hMem) {
  return urBindlessImagesImageFreeExp(hContext, hDevice, hMem);
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImportExternalMemoryExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, size_t size,
    ur_exp_external_mem_type_t memHandleType,
    ur_exp_interop_mem_desc_t *pInteropMemDesc,
    ur_exp_interop_mem_handle_t *phInteropMem) {
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = size;
  std::ignore = memHandleType;
  std::ignore = pInteropMemDesc;
  std::ignore = phInteropMem;
  logger::error(logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
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
  logger::error(logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesReleaseInteropExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_interop_mem_handle_t hInteropMem) {
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = hInteropMem;
  logger::error(logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImportExternalSemaphoreExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_external_semaphore_type_t semHandleType,
    ur_exp_interop_semaphore_desc_t *pInteropSemaphoreDesc,
    ur_exp_interop_semaphore_handle_t *phInteropSemaphoreHandle) {
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = semHandleType;
  std::ignore = pInteropSemaphoreDesc;
  std::ignore = phInteropSemaphoreHandle;
  logger::error(logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesDestroyExternalSemaphoreExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_interop_semaphore_handle_t hInteropSemaphore) {
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = hInteropSemaphore;
  logger::error(logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesWaitExternalSemaphoreExp(
    ur_queue_handle_t hQueue, ur_exp_interop_semaphore_handle_t hSemaphore,
    bool hasValue, uint64_t waitValue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = hSemaphore;
  std::ignore = hasValue;
  std::ignore = waitValue;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;
  logger::error(logger::LegacyMessage("[UR][L0] "),
                " {} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesSignalExternalSemaphoreExp(
    ur_queue_handle_t hQueue, ur_exp_interop_semaphore_handle_t hSemaphore,
    bool hasValue, uint64_t signalValue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = hSemaphore;
  std::ignore = hasValue;
  std::ignore = signalValue;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;
  logger::error(logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
