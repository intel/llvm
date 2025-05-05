//===--------- memory_helpers.cpp - Level Zero Adapter -------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "memory_helpers.hpp"
#include "../common.hpp"

ur_result_t
getMemoryAttrs(ze_context_handle_t hContext, void *ptr,
               ze_device_handle_t *hDevice,
               ZeStruct<ze_memory_allocation_properties_t> *properties) {
  // TODO: use UMF once
  // https://github.com/oneapi-src/unified-memory-framework/issues/687 is
  // implemented
  ZE2UR_CALL(zeMemGetAllocProperties, (hContext, ptr, properties, hDevice));
  return UR_RESULT_SUCCESS;
}

bool maybeImportUSM(ze_driver_handle_t hTranslatedDriver,
                    ze_context_handle_t hContext, void *ptr, size_t size) {
  if (!ZeUSMImport.Enabled || ptr == nullptr) {
    return false;
  }

  ZeStruct<ze_memory_allocation_properties_t> properties;
  auto ret = getMemoryAttrs(hContext, ptr, nullptr, &properties);

  if (ret == UR_RESULT_SUCCESS && properties.type == ZE_MEMORY_TYPE_UNKNOWN) {
    // Promote the host ptr to USM host memory
    ZeUSMImport.doZeUSMImport(hTranslatedDriver, ptr, size);
    return true;
  }
  return false;
}

ze_region_params ur2zeRegionParams(ur_rect_offset_t SrcOrigin,
                                   ur_rect_offset_t DstOrigin,
                                   ur_rect_region_t Region, size_t SrcRowPitch,
                                   size_t DstRowPitch, size_t SrcSlicePitch,
                                   size_t DstSlicePitch) {
  uint32_t SrcOriginX = ur_cast<uint32_t>(SrcOrigin.x);
  uint32_t SrcOriginY = ur_cast<uint32_t>(SrcOrigin.y);
  uint32_t SrcOriginZ = ur_cast<uint32_t>(SrcOrigin.z);

  uint32_t SrcPitch = SrcRowPitch;
  if (SrcPitch == 0)
    SrcPitch = ur_cast<uint32_t>(Region.width);

  if (SrcSlicePitch == 0)
    SrcSlicePitch = ur_cast<uint32_t>(Region.height) * SrcPitch;

  uint32_t DstOriginX = ur_cast<uint32_t>(DstOrigin.x);
  uint32_t DstOriginY = ur_cast<uint32_t>(DstOrigin.y);
  uint32_t DstOriginZ = ur_cast<uint32_t>(DstOrigin.z);

  uint32_t DstPitch = DstRowPitch;
  if (DstPitch == 0)
    DstPitch = ur_cast<uint32_t>(Region.width);

  if (DstSlicePitch == 0)
    DstSlicePitch = ur_cast<uint32_t>(Region.height) * DstPitch;

  uint32_t Width = ur_cast<uint32_t>(Region.width);
  uint32_t Height = ur_cast<uint32_t>(Region.height);
  uint32_t Depth = ur_cast<uint32_t>(Region.depth);

  const ze_copy_region_t ZeSrcRegion = {SrcOriginX, SrcOriginY, SrcOriginZ,
                                        Width,      Height,     Depth};
  const ze_copy_region_t ZeDstRegion = {DstOriginX, DstOriginY, DstOriginZ,
                                        Width,      Height,     Depth};

  return ze_region_params{ZeDstRegion, DstPitch, DstSlicePitch,
                          ZeSrcRegion, SrcPitch, SrcSlicePitch};
}
