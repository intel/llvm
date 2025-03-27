//===--------- image_helpers.cpp - Level Zero Adapter --------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "adapters/level_zero/common.hpp"

#ifdef UR_ADAPTER_LEVEL_ZERO_V2
#include "../v2/context.hpp"
#else
#include "../context.hpp"
#endif
#include "../sampler.hpp"
#include "image_helpers.hpp"

#include <loader/ze_loader.h>
#include <ze_api.h>

namespace {

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

} // namespace

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
  case UR_IMAGE_CHANNEL_ORDER_R:
  case UR_IMAGE_CHANNEL_ORDER_A: {
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
      logger::error("ur2zeImageDesc: unexpected data type Size\n");
      return UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT;
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
      logger::error("ur2zeImageDesc: unexpected data type Size\n");
      return UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT;
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
      return UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT;
    }
    break;
  }
  case UR_IMAGE_CHANNEL_ORDER_RGBA:
  case UR_IMAGE_CHANNEL_ORDER_RGBX:
  case UR_IMAGE_CHANNEL_ORDER_BGRA:
  case UR_IMAGE_CHANNEL_ORDER_ARGB:
  case UR_IMAGE_CHANNEL_ORDER_ABGR: {
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

  ze_image_format_t ZeFormatDesc;
  switch (ImageFormat->channelOrder) {
  case UR_IMAGE_CHANNEL_ORDER_R:
    ZeFormatDesc = {ZeImageFormatLayout,       ZeImageFormatType,
                    ZE_IMAGE_FORMAT_SWIZZLE_R, ZE_IMAGE_FORMAT_SWIZZLE_0,
                    ZE_IMAGE_FORMAT_SWIZZLE_0, ZE_IMAGE_FORMAT_SWIZZLE_1};
    break;
  case UR_IMAGE_CHANNEL_ORDER_A:
    ZeFormatDesc = {ZeImageFormatLayout,       ZeImageFormatType,
                    ZE_IMAGE_FORMAT_SWIZZLE_A, ZE_IMAGE_FORMAT_SWIZZLE_0,
                    ZE_IMAGE_FORMAT_SWIZZLE_0, ZE_IMAGE_FORMAT_SWIZZLE_1};
    break;
  case UR_IMAGE_CHANNEL_ORDER_RG:
    ZeFormatDesc = {ZeImageFormatLayout,       ZeImageFormatType,
                    ZE_IMAGE_FORMAT_SWIZZLE_R, ZE_IMAGE_FORMAT_SWIZZLE_G,
                    ZE_IMAGE_FORMAT_SWIZZLE_0, ZE_IMAGE_FORMAT_SWIZZLE_1};
    break;
  case UR_IMAGE_CHANNEL_ORDER_RA:
    ZeFormatDesc = {ZeImageFormatLayout,       ZeImageFormatType,
                    ZE_IMAGE_FORMAT_SWIZZLE_R, ZE_IMAGE_FORMAT_SWIZZLE_A,
                    ZE_IMAGE_FORMAT_SWIZZLE_0, ZE_IMAGE_FORMAT_SWIZZLE_1};
    break;
  case UR_IMAGE_CHANNEL_ORDER_RX:
    ZeFormatDesc = {ZeImageFormatLayout,       ZeImageFormatType,
                    ZE_IMAGE_FORMAT_SWIZZLE_R, ZE_IMAGE_FORMAT_SWIZZLE_X,
                    ZE_IMAGE_FORMAT_SWIZZLE_0, ZE_IMAGE_FORMAT_SWIZZLE_1};
    break;
  case UR_IMAGE_CHANNEL_ORDER_RGB:
    ZeFormatDesc = {ZeImageFormatLayout,       ZeImageFormatType,
                    ZE_IMAGE_FORMAT_SWIZZLE_R, ZE_IMAGE_FORMAT_SWIZZLE_G,
                    ZE_IMAGE_FORMAT_SWIZZLE_B, ZE_IMAGE_FORMAT_SWIZZLE_1};
    break;
  case UR_IMAGE_CHANNEL_ORDER_RGX:
    ZeFormatDesc = {ZeImageFormatLayout,       ZeImageFormatType,
                    ZE_IMAGE_FORMAT_SWIZZLE_R, ZE_IMAGE_FORMAT_SWIZZLE_G,
                    ZE_IMAGE_FORMAT_SWIZZLE_X, ZE_IMAGE_FORMAT_SWIZZLE_1};
    break;
  case UR_IMAGE_CHANNEL_ORDER_RGBA:
    ZeFormatDesc = {ZeImageFormatLayout,       ZeImageFormatType,
                    ZE_IMAGE_FORMAT_SWIZZLE_R, ZE_IMAGE_FORMAT_SWIZZLE_G,
                    ZE_IMAGE_FORMAT_SWIZZLE_B, ZE_IMAGE_FORMAT_SWIZZLE_A};
    break;
  case UR_IMAGE_CHANNEL_ORDER_RGBX:
    ZeFormatDesc = {ZeImageFormatLayout,       ZeImageFormatType,
                    ZE_IMAGE_FORMAT_SWIZZLE_R, ZE_IMAGE_FORMAT_SWIZZLE_G,
                    ZE_IMAGE_FORMAT_SWIZZLE_B, ZE_IMAGE_FORMAT_SWIZZLE_X};
    break;
  case UR_IMAGE_CHANNEL_ORDER_BGRA:
    ZeFormatDesc = {ZeImageFormatLayout,       ZeImageFormatType,
                    ZE_IMAGE_FORMAT_SWIZZLE_B, ZE_IMAGE_FORMAT_SWIZZLE_G,
                    ZE_IMAGE_FORMAT_SWIZZLE_R, ZE_IMAGE_FORMAT_SWIZZLE_A};
    break;
  case UR_IMAGE_CHANNEL_ORDER_ARGB:
    ZeFormatDesc = {ZeImageFormatLayout,       ZeImageFormatType,
                    ZE_IMAGE_FORMAT_SWIZZLE_A, ZE_IMAGE_FORMAT_SWIZZLE_R,
                    ZE_IMAGE_FORMAT_SWIZZLE_G, ZE_IMAGE_FORMAT_SWIZZLE_B};
    break;
  case UR_IMAGE_CHANNEL_ORDER_ABGR:
    ZeFormatDesc = {ZeImageFormatLayout,       ZeImageFormatType,
                    ZE_IMAGE_FORMAT_SWIZZLE_A, ZE_IMAGE_FORMAT_SWIZZLE_B,
                    ZE_IMAGE_FORMAT_SWIZZLE_G, ZE_IMAGE_FORMAT_SWIZZLE_R};
    break;
  default:
    logger::error("ur2zeImageDesc: unsupported image channel order");
    return UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT;
  }
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
  ZeImageDesc.stype = ZE_STRUCTURE_TYPE_IMAGE_DESC;
  ZeImageDesc.pNext = ImageDesc->pNext;
  ZeImageDesc.flags = 0;
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

ur_result_t bindlessImagesHandleCopyFlags(
    const void *pSrc, void *pDst, const ur_image_desc_t *pSrcImageDesc,
    const ur_image_desc_t *pDstImageDesc,
    const ur_image_format_t *pSrcImageFormat,
    const ur_image_format_t *pDstImageFormat,
    ur_exp_image_copy_region_t *pCopyRegion,
    ur_exp_image_copy_flags_t imageCopyFlags,
    ze_command_list_handle_t ZeCommandList, ze_event_handle_t zeSignalEvent,
    uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents) {

  ZeStruct<ze_image_desc_t> zeSrcImageDesc;
  ur2zeImageDesc(pSrcImageFormat, pSrcImageDesc, zeSrcImageDesc);

  switch (imageCopyFlags) {
  case UR_EXP_IMAGE_COPY_FLAG_HOST_TO_DEVICE: {
    uint32_t SrcRowPitch =
        pSrcImageDesc->width * getPixelSizeBytes(pSrcImageFormat);
    uint32_t SrcSlicePitch = SrcRowPitch * pSrcImageDesc->height;
    if (pDstImageDesc->rowPitch == 0) {
      // Copy to Non-USM memory

      ze_image_region_t DstRegion;
      UR_CALL(getImageRegionHelper(zeSrcImageDesc, &pCopyRegion->dstOffset,
                                   &pCopyRegion->copyExtent, DstRegion));
      auto *urDstImg = static_cast<ur_bindless_mem_handle_t *>(pDst);

      const char *SrcPtr =
          static_cast<const char *>(pSrc) +
          pCopyRegion->srcOffset.z * SrcSlicePitch +
          pCopyRegion->srcOffset.y * SrcRowPitch +
          pCopyRegion->srcOffset.x * getPixelSizeBytes(pSrcImageFormat);

      ZE2UR_CALL(zeCommandListAppendImageCopyFromMemoryExt,
                 (ZeCommandList, urDstImg->getZeImage(), SrcPtr, &DstRegion,
                  SrcRowPitch, SrcSlicePitch, zeSignalEvent, numWaitEvents,
                  phWaitEvents));
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
                  pSrc, &ZeSrcRegion, SrcRowPitch, SrcSlicePitch, zeSignalEvent,
                  numWaitEvents, phWaitEvents));
    }
    return UR_RESULT_SUCCESS;
  };
  case UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_HOST: {
    uint32_t DstRowPitch =
        pDstImageDesc->width * getPixelSizeBytes(pDstImageFormat);
    uint32_t DstSlicePitch = DstRowPitch * pDstImageDesc->height;
    if (pSrcImageDesc->rowPitch == 0) {
      // Copy from Non-USM memory to host
      ze_image_region_t SrcRegion;
      UR_CALL(getImageRegionHelper(zeSrcImageDesc, &pCopyRegion->srcOffset,
                                   &pCopyRegion->copyExtent, SrcRegion));

      auto *urSrcImg = reinterpret_cast<const ur_bindless_mem_handle_t *>(pSrc);

      char *DstPtr =
          static_cast<char *>(pDst) + pCopyRegion->dstOffset.z * DstSlicePitch +
          pCopyRegion->dstOffset.y * DstRowPitch +
          pCopyRegion->dstOffset.x * getPixelSizeBytes(pDstImageFormat);
      ZE2UR_CALL(zeCommandListAppendImageCopyToMemoryExt,
                 (ZeCommandList, DstPtr, urSrcImg->getZeImage(), &SrcRegion,
                  DstRowPitch, DstSlicePitch, zeSignalEvent, numWaitEvents,
                  phWaitEvents));
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
                  pSrc, &ZeSrcRegion, SrcRowPitch, SrcSlicePitch, zeSignalEvent,
                  numWaitEvents, phWaitEvents));
    }
    return UR_RESULT_SUCCESS;
  };
  case UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_DEVICE: {
    ze_image_region_t DstRegion;
    UR_CALL(getImageRegionHelper(zeSrcImageDesc, &pCopyRegion->dstOffset,
                                 &pCopyRegion->copyExtent, DstRegion));
    ze_image_region_t SrcRegion;
    UR_CALL(getImageRegionHelper(zeSrcImageDesc, &pCopyRegion->srcOffset,
                                 &pCopyRegion->copyExtent, SrcRegion));

    auto *urImgSrc = reinterpret_cast<const ur_bindless_mem_handle_t *>(pSrc);
    auto *urImgDst = reinterpret_cast<ur_bindless_mem_handle_t *>(pDst);

    ZE2UR_CALL(zeCommandListAppendImageCopyRegion,
               (ZeCommandList, urImgDst->getZeImage(), urImgSrc->getZeImage(),
                &DstRegion, &SrcRegion, zeSignalEvent, numWaitEvents,
                phWaitEvents));

    return UR_RESULT_SUCCESS;
  };
  default:
    logger::error("ur_queue_immediate_in_order_t::bindlessImagesImageCopyExp: "
                  "unexpected imageCopyFlags");
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }
}
