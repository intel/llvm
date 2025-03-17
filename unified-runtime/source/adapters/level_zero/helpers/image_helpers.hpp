//===--------- image_helpers.hpp - Level Zero Adapter --------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <ur/ur.hpp>
#include <ze_api.h>
#include <zes_api.h>

#include "../v2/common.hpp"

typedef ze_result_t(ZE_APICALL *zeImageGetDeviceOffsetExp_pfn)(
    ze_image_handle_t hImage, uint64_t *pDeviceOffset);

typedef ze_result_t(ZE_APICALL *zeMemGetPitchFor2dImage_pfn)(
    ze_context_handle_t hContext, ze_device_handle_t hDevice, size_t imageWidth,
    size_t imageHeight, unsigned int elementSizeInBytes, size_t *rowPitch);

struct ur_bindless_mem_handle_t {

  ur_bindless_mem_handle_t(ze_image_handle_t zeImage,
                           const ZeStruct<ze_image_desc_t> &zeImageDesc)
      : zeImage(zeImage), zeImageDesc(zeImageDesc) {};

  ze_image_handle_t getZeImage() const { return zeImage.get(); }
  ze_image_desc_t &getZeImageDesc() { return zeImageDesc; }

private:
  v2::raii::ze_image_handle_t zeImage;
  ZeStruct<ze_image_desc_t> zeImageDesc;
};

/// Construct UR bindless image struct from ZE image handle and desc.

inline ur_result_t createUrImgFromZeImage(ze_image_handle_t ZeImage,
                                   const ZeStruct<ze_image_desc_t> &ZeImageDesc,
                                   ur_exp_image_mem_native_handle_t * pImg) {
  try {
    ur_bindless_mem_handle_t * UrImg = new ur_bindless_mem_handle_t(ZeImage, ZeImageDesc);
    *pImg = reinterpret_cast<ur_exp_image_mem_native_handle_t>(UrImg);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

/// Construct UR image format from ZE image desc.
ur_result_t ze2urImageFormat(const ze_image_desc_t *ZeImageDesc,
                             ur_image_format_t *UrImageFormat);

/// Construct ZE image desc from UR image format and desc.
ur_result_t ur2zeImageDesc(const ur_image_format_t *ImageFormat,
                           const ur_image_desc_t *ImageDesc,
                           ZeStruct<ze_image_desc_t> &ZeImageDesc);

/// Return element size in bytes of a pixel.
uint32_t getPixelSizeBytes(const ur_image_format_t *Format);

bool Is3ChannelOrder(ur_image_channel_order_t ChannelOrder);

ur_result_t getImageRegionHelper(ze_image_desc_t ZeImageDesc,
                                 ur_rect_offset_t *Origin,
                                 ur_rect_region_t *Region,
                                 ze_image_region_t &ZeRegion);

std::pair<ze_image_format_type_t, size_t>
getImageFormatTypeAndSize(const ur_image_format_t *ImageFormat);

ur_result_t bindlessImagesCreateImpl(ur_context_handle_t hContext,
                                     ur_device_handle_t hDevice,
                                     ur_exp_image_mem_native_handle_t hImageMem,
                                     const ur_image_format_t *pImageFormat,
                                     const ur_image_desc_t *pImageDesc,
                                     ur_sampler_handle_t hSampler,
                                     ur_exp_image_native_handle_t *phImage);
