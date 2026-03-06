/*
 * Copyright (C) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef _ZE_BINDLESS_IMAGE_EXP_H
#define _ZE_BINDLESS_IMAGE_EXP_H
#if defined(__cplusplus)
#pragma once
#endif

#include "level_zero/ze_stypes.h"
#include <level_zero/ze_api.h>

#ifndef ZE_BINDLESS_IMAGE_EXP_NAME
#define ZE_BINDLESS_IMAGE_EXP_NAME "ZE_experimental_bindless_image"

typedef enum _ze_bindless_image_exp_version_t {
  ZE_BINDLESS_IMAGE_EXP_VERSION_1_0 = 0,
  ZE_BINDLESS_IMAGE_EXP_VERSION_CURRENT = 1,
  ZE_BINDLESS_IMAGE_EXP_VERSION_FORCE_UINT32 = 0x7fffffff
} ze_bindless_image_exp_version_t;

typedef enum _ze_image_bindless_exp_flags_t {
  ZE_IMAGE_BINDLESS_EXP_FLAG_BINDLESS = ZE_BIT(0),
  ZE_IMAGE_BINDLESS_EXP_FLAG_SAMPLED_IMAGE = ZE_BIT(1),
  ZE_IMAGE_BINDLESS_EXP_FLAG_FORCE_UINT32 = 0x7fffffff
} ze_image_bindless_exp_flags_t;

typedef struct _ze_image_bindless_exp_desc_t {
  ze_structure_type_t stype; ///< [in] type of this structure
  const void
      *pNext; ///< [in][optional] must be null or a pointer to an
              ///< extension-specific structure (i.e. contains stype and pNext).
  ze_image_bindless_exp_flags_t
      flags; ///< [in] image flags.
             ///< must be 0 (default) or a valid combination of
             ///< ::ze_image_bindless_exp_flag_t; default behavior bindless
             ///< images are not used when creating handles via zeImageCreate.
} ze_image_bindless_exp_desc_t;

typedef struct _ze_image_pitched_exp_desc_t {
  ze_structure_type_t stype; ///< [in] type of this structure
  const void
      *pNext; ///< [in][optional] must be null or a pointer to an
              ///< extension-specific structure (i.e. contains stype and pNext).
  void *ptr;  ///< [in] pointer to pitched device allocation allocated using
              ///< zeMemAllocDevice.
} ze_image_pitched_exp_desc_t;

typedef struct _ze_device_pitched_alloc_exp_properties_t {
  ze_structure_type_t stype; ///< [in] type of this structure
  void
      *pNext; ///< [in,out][optional] must be null or a pointer to an
              ///< extension-specific structure (i.e. contains stype and pNext).
  size_t maxImageLinearWidth;  ///< [out] Maximum image linear width.
  size_t maxImageLinearHeight; ///< [out] Maximum image linear height.
} ze_device_pitched_alloc_exp_properties_t;

ZE_APIEXPORT ze_result_t ZE_APICALL zeMemGetPitchFor2dImage(
    ze_context_handle_t hContext, ze_device_handle_t hDevice, size_t imageWidth,
    size_t imageHeight, unsigned int elementSizeInBytes, size_t *rowPitch);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeImageGetDeviceOffsetExp(ze_image_handle_t hImage, uint64_t *pDeviceOffset);

#endif // ZE_BINDLESS_IMAGE_EXP_NAME

#define ZE_IMAGE_BINDLESS_EXP_FLAG_SAMPLED_IMAGE 2

#endif
