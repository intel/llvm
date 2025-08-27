//===--------- image_common.hpp - Level Zero Adapter
//--------------------===//
//
// Copyright (C) 2024-2025 Intel Corporation
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

#include "v2/common.hpp"

struct ur_bindless_mem_handle_t {
  // Constructor for bindless image handle
  ur_bindless_mem_handle_t(ze_image_handle_t zeImage,
                           const ZeStruct<ze_image_desc_t> &zeImageDesc)
      : zeImage(zeImage) {

    format = zeImageDesc.format;
    width = zeImageDesc.width;
    height = zeImageDesc.height;
    depth = zeImageDesc.depth;
  };

  ze_image_handle_t getZeImage() const { return zeImage.get(); }

  ze_image_format_t getFormat() const { return format; }
  uint64_t getWidth() const { return width; }
  uint64_t getHeight() const { return height; }
  uint64_t getDepth() const { return depth; }

private:
  v2::raii::ze_image_handle_t zeImage;
  ze_image_format_t format;
  uint64_t width;
  uint64_t height;
  uint64_t depth;
};

/// Construct ZE image desc from UR image format and desc.
ur_result_t ur2zeImageDesc(const ur_image_format_t *ImageFormat,
                           const ur_image_desc_t *ImageDesc,
                           ZeStruct<ze_image_desc_t> &ZeImageDesc);

ur_result_t getImageRegionHelper(ze_image_desc_t ZeImageDesc,
                                 ur_rect_offset_t *Origin,
                                 ur_rect_region_t *Region,
                                 ze_image_region_t &ZeRegion);

ur_result_t bindlessImagesHandleCopyFlags(
    const void *pSrc, void *pDst, const ur_image_desc_t *pSrcImageDesc,
    const ur_image_desc_t *pDstImageDesc,
    const ur_image_format_t *pSrcImageFormat,
    const ur_image_format_t *pDstImageFormat,
    ur_exp_image_copy_region_t *pCopyRegion,
    ur_exp_image_copy_flags_t imageCopyFlags,
    ze_command_list_handle_t ZeCommandList, ze_event_handle_t zeSignalEvent,
    uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents);

bool is3ChannelOrder(ur_image_channel_order_t ChannelOrder);

bool verifyStandardImageSupport(const ur_device_handle_t hDevice,
                                const ur_image_desc_t *pImageDesc,
                                ur_exp_image_mem_type_t imageMemHandleType);

bool verifyMipmapImageSupport(const ur_device_handle_t hDevice,
                              const ur_image_desc_t *pImageDesc,
                              ur_exp_image_mem_type_t imageMemHandleType);

bool verifyCubemapImageSupport(const ur_device_handle_t hDevice,
                               const ur_image_desc_t *pImageDesc,
                               ur_exp_image_mem_type_t imageMemHandleType);

bool verifyLayeredImageSupport(const ur_device_handle_t hDevice,
                               const ur_image_desc_t *pImageDesc,
                               ur_exp_image_mem_type_t imageMemHandleType);

bool verifyGatherImageSupport(const ur_device_handle_t hDevice,
                              const ur_image_desc_t *pImageDesc,
                              ur_exp_image_mem_type_t imageMemHandleType);

bool verifyCommonImagePropertiesSupport(
    const ur_device_handle_t hDevice, const ur_image_desc_t *pImageDesc,
    const ur_image_format_t *pImageFormat,
    ur_exp_image_mem_type_t imageMemHandleType);
