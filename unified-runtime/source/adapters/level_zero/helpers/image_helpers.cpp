//===--------- image_helpers.cpp - Level Zero Adapter --------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "image_helpers.hpp"

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

bool Is3ChannelOrder(ur_image_channel_order_t ChannelOrder) {
  switch (ChannelOrder) {
  case UR_IMAGE_CHANNEL_ORDER_RGB:
  case UR_IMAGE_CHANNEL_ORDER_RGX:
    return true;
  default:
    return false;
  }
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
  case UR_IMAGE_CHANNEL_ORDER_RGBA: {
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
      logger::error("ur2zeImageDesc: unexpected data type Size\n");
      return UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT;
    }
    break;
  }
  default:
    logger::error("format layout = {}", ImageFormat->channelOrder);
    return UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT;
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
    return UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR;
  }

  ZeImageDesc.arraylevels = ZeImageDesc.flags = 0;
  ZeImageDesc.type = ZeImageType;
  ZeImageDesc.format = ZeFormatDesc;
  ZeImageDesc.width = ur_cast<uint64_t>(ImageDesc->width);
  ZeImageDesc.height = ur_cast<uint64_t>(ImageDesc->height);
  ZeImageDesc.depth = ur_cast<uint64_t>(ImageDesc->depth);
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
    logger::error("ur2zeImageDesc: unsupported image data type: data type = {}",
                  ImageFormat->channelType);
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_FORCE_UINT32;
    ZeImageFormatTypeSize = 0;
  }
  return {ZeImageFormatType, ZeImageFormatTypeSize};
}
