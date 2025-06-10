//===--------- kernel_helpers.cpp - Level Zero Adapter -------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kernel_helpers.hpp"
#include "logger/ur_logger.hpp"

#include "../common.hpp"
#include "../device.hpp"

#ifdef UR_ADAPTER_LEVEL_ZERO_V2
#include "../v2/context.hpp"
#else
#include "../context.hpp"
#endif

ur_result_t getSuggestedLocalWorkSize(ur_device_handle_t hDevice,
                                      ze_kernel_handle_t hZeKernel,
                                      size_t GlobalWorkSize3D[3],
                                      uint32_t SuggestedLocalWorkSize3D[3]) {
  uint32_t *WG = SuggestedLocalWorkSize3D;

  // We can't call to zeKernelSuggestGroupSize if 64-bit GlobalWorkSize
  // values do not fit to 32-bit that the API only supports currently.
  bool SuggestGroupSize = true;
  for (int I : {0, 1, 2}) {
    if (GlobalWorkSize3D[I] > UINT32_MAX) {
      SuggestGroupSize = false;
    }
  }
  if (SuggestGroupSize) {
    ZE2UR_CALL(zeKernelSuggestGroupSize,
               (hZeKernel, GlobalWorkSize3D[0], GlobalWorkSize3D[1],
                GlobalWorkSize3D[2], &WG[0], &WG[1], &WG[2]));
  } else {
    for (int I : {0, 1, 2}) {
      // Try to find a I-dimension WG size that the GlobalWorkSize[I] is
      // fully divisable with. Start with the max possible size in
      // each dimension.
      uint32_t GroupSize[] = {
          hDevice->ZeDeviceComputeProperties->maxGroupSizeX,
          hDevice->ZeDeviceComputeProperties->maxGroupSizeY,
          hDevice->ZeDeviceComputeProperties->maxGroupSizeZ};
      GroupSize[I] = (std::min)(size_t(GroupSize[I]), GlobalWorkSize3D[I]);
      while (GlobalWorkSize3D[I] % GroupSize[I]) {
        --GroupSize[I];
      }
      if (GlobalWorkSize3D[I] / GroupSize[I] > UINT32_MAX) {
        UR_LOG(ERR, "getSuggestedLocalWorkSize: can't find a WG size "
                    "suitable for global work size > UINT32_MAX");
        return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
      }
      WG[I] = GroupSize[I];
    }
    UR_LOG(DEBUG,
           "getSuggestedLocalWorkSize: using computed WG size = {{{}, {}, {}}}",
           WG[0], WG[1], WG[2]);
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t setKernelGlobalOffset(ur_context_handle_t Context,
                                  ze_kernel_handle_t Kernel, uint32_t WorkDim,
                                  const size_t *GlobalWorkOffset) {
  if (!Context->getPlatform()->ZeDriverGlobalOffsetExtensionFound) {
    UR_LOG(DEBUG, "No global offset extension found on this driver");
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  auto OffsetX = GlobalWorkOffset[0];
  auto OffsetY = WorkDim > 1 ? GlobalWorkOffset[1] : 0;
  auto OffsetZ = WorkDim > 2 ? GlobalWorkOffset[2] : 0;
  ZE2UR_CALL(zeKernelSetGlobalOffsetExp, (Kernel, OffsetX, OffsetY, OffsetZ));

  return UR_RESULT_SUCCESS;
}

ur_result_t calculateKernelWorkDimensions(
    ze_kernel_handle_t Kernel, ur_device_handle_t Device,
    ze_group_count_t &ZeThreadGroupDimensions, uint32_t (&WG)[3],
    uint32_t WorkDim, const size_t *GlobalWorkSize,
    const size_t *LocalWorkSize) {

  UR_ASSERT(GlobalWorkSize, UR_RESULT_ERROR_INVALID_VALUE);
  // If LocalWorkSize is not provided then Kernel must be provided to query
  // suggested group size.
  UR_ASSERT(LocalWorkSize || Kernel, UR_RESULT_ERROR_INVALID_VALUE);

  // New variable needed because GlobalWorkSize parameter might not be of size
  // 3
  size_t GlobalWorkSize3D[3]{1, 1, 1};
  std::copy(GlobalWorkSize, GlobalWorkSize + WorkDim, GlobalWorkSize3D);

  if (LocalWorkSize) {
    WG[0] = ur_cast<uint32_t>(LocalWorkSize[0]);
    WG[1] = WorkDim >= 2 ? ur_cast<uint32_t>(LocalWorkSize[1]) : 1;
    WG[2] = WorkDim == 3 ? ur_cast<uint32_t>(LocalWorkSize[2]) : 1;
  } else {
    UR_CALL(getSuggestedLocalWorkSize(Device, Kernel, GlobalWorkSize3D, WG));
  }

  // TODO: assert if sizes do not fit into 32-bit?
  switch (WorkDim) {
  case 3:
    ZeThreadGroupDimensions.groupCountX =
        ur_cast<uint32_t>(GlobalWorkSize3D[0] / WG[0]);
    ZeThreadGroupDimensions.groupCountY =
        ur_cast<uint32_t>(GlobalWorkSize3D[1] / WG[1]);
    ZeThreadGroupDimensions.groupCountZ =
        ur_cast<uint32_t>(GlobalWorkSize3D[2] / WG[2]);
    break;
  case 2:
    ZeThreadGroupDimensions.groupCountX =
        ur_cast<uint32_t>(GlobalWorkSize3D[0] / WG[0]);
    ZeThreadGroupDimensions.groupCountY =
        ur_cast<uint32_t>(GlobalWorkSize3D[1] / WG[1]);
    WG[2] = 1;
    break;
  case 1:
    ZeThreadGroupDimensions.groupCountX =
        ur_cast<uint32_t>(GlobalWorkSize3D[0] / WG[0]);
    WG[1] = WG[2] = 1;
    break;

  default:
    UR_LOG(ERR, "calculateKernelWorkDimensions: unsupported work_dim");
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  // Error handling for non-uniform group size case
  if (GlobalWorkSize3D[0] !=
      size_t(ZeThreadGroupDimensions.groupCountX) * WG[0]) {
    UR_LOG(ERR, "calculateKernelWorkDimensions: invalid work_dim. The range "
                "is not a multiple of the group size in the 1st dimension");
    return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
  }
  if (GlobalWorkSize3D[1] !=
      size_t(ZeThreadGroupDimensions.groupCountY) * WG[1]) {
    UR_LOG(ERR, "calculateKernelWorkDimensions: invalid work_dim. The range "
                "is not a multiple of the group size in the 2nd dimension");
    return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
  }
  if (GlobalWorkSize3D[2] !=
      size_t(ZeThreadGroupDimensions.groupCountZ) * WG[2]) {
    UR_LOG(ERR, "calculateKernelWorkDimensions: invalid work_dim. The range "
                "is not a multiple of the group size in the 3rd dimension");
    return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
  }

  return UR_RESULT_SUCCESS;
}
