//===--------- tensor_map.cpp - CUDA Adapter ------------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda.h>
#include <ur_api.h>

#include "context.hpp"

struct ur_exp_tensor_map_handle_t_ {
  CUtensorMap Map;
};

#define CONVERT(URTYPE, CUTYPE)                                                \
  if ((URTYPE)&UrType)                                                         \
    return (CUTYPE);

inline CUtensorMapDataType
convertUrToCuDataType(ur_exp_tensor_map_data_type_flags_t UrType) {
  CONVERT(UR_EXP_TENSOR_MAP_DATA_TYPE_FLAG_UINT8,
          CU_TENSOR_MAP_DATA_TYPE_UINT8);
  CONVERT(UR_EXP_TENSOR_MAP_DATA_TYPE_FLAG_UINT16,
          CU_TENSOR_MAP_DATA_TYPE_UINT16);
  CONVERT(UR_EXP_TENSOR_MAP_DATA_TYPE_FLAG_UINT32,
          CU_TENSOR_MAP_DATA_TYPE_UINT32);
  CONVERT(UR_EXP_TENSOR_MAP_DATA_TYPE_FLAG_INT32,
          CU_TENSOR_MAP_DATA_TYPE_INT32);
  CONVERT(UR_EXP_TENSOR_MAP_DATA_TYPE_FLAG_UINT64,
          CU_TENSOR_MAP_DATA_TYPE_UINT64);
  CONVERT(UR_EXP_TENSOR_MAP_DATA_TYPE_FLAG_INT64,
          CU_TENSOR_MAP_DATA_TYPE_INT64);
  CONVERT(UR_EXP_TENSOR_MAP_DATA_TYPE_FLAG_FLOAT16,
          CU_TENSOR_MAP_DATA_TYPE_FLOAT16);
  CONVERT(UR_EXP_TENSOR_MAP_DATA_TYPE_FLAG_FLOAT32,
          CU_TENSOR_MAP_DATA_TYPE_FLOAT32);
  CONVERT(UR_EXP_TENSOR_MAP_DATA_TYPE_FLAG_FLOAT64,
          CU_TENSOR_MAP_DATA_TYPE_FLOAT64);
  CONVERT(UR_EXP_TENSOR_MAP_DATA_TYPE_FLAG_BFLOAT16,
          CU_TENSOR_MAP_DATA_TYPE_BFLOAT16);
  CONVERT(UR_EXP_TENSOR_MAP_DATA_TYPE_FLAG_FLOAT32_FTZ,
          CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ);
  CONVERT(UR_EXP_TENSOR_MAP_DATA_TYPE_FLAG_TFLOAT32,
          CU_TENSOR_MAP_DATA_TYPE_TFLOAT32);
  CONVERT(UR_EXP_TENSOR_MAP_DATA_TYPE_FLAG_TFLOAT32_FTZ,
          CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ);
  throw "convertUrToCuDataType failed!";
}

CUtensorMapInterleave
convertUrToCuInterleave(ur_exp_tensor_map_interleave_flags_t UrType) {
  CONVERT(UR_EXP_TENSOR_MAP_INTERLEAVE_FLAG_NONE,
          CU_TENSOR_MAP_INTERLEAVE_NONE);
  CONVERT(UR_EXP_TENSOR_MAP_INTERLEAVE_FLAG_16B, CU_TENSOR_MAP_INTERLEAVE_16B);
  CONVERT(UR_EXP_TENSOR_MAP_INTERLEAVE_FLAG_32B, CU_TENSOR_MAP_INTERLEAVE_32B);
  throw "convertUrToCuInterleave failed!";
}

CUtensorMapSwizzle
convertUrToCuSwizzle(ur_exp_tensor_map_swizzle_flags_t UrType) {
  CONVERT(UR_EXP_TENSOR_MAP_SWIZZLE_FLAG_NONE, CU_TENSOR_MAP_SWIZZLE_NONE);
  CONVERT(UR_EXP_TENSOR_MAP_SWIZZLE_FLAG_32B, CU_TENSOR_MAP_SWIZZLE_32B);
  CONVERT(UR_EXP_TENSOR_MAP_SWIZZLE_FLAG_64B, CU_TENSOR_MAP_SWIZZLE_64B);
  CONVERT(UR_EXP_TENSOR_MAP_SWIZZLE_FLAG_128B, CU_TENSOR_MAP_SWIZZLE_128B);
  throw "convertUrToCuSwizzle failed!";
}

CUtensorMapL2promotion
convertUrToCuL2Promotion(ur_exp_tensor_map_l2_promotion_flags_t UrType) {
  CONVERT(UR_EXP_TENSOR_MAP_L2_PROMOTION_FLAG_NONE,
          CU_TENSOR_MAP_L2_PROMOTION_NONE);
  CONVERT(UR_EXP_TENSOR_MAP_L2_PROMOTION_FLAG_64B,
          CU_TENSOR_MAP_L2_PROMOTION_L2_64B);
  CONVERT(UR_EXP_TENSOR_MAP_L2_PROMOTION_FLAG_128B,
          CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  CONVERT(UR_EXP_TENSOR_MAP_L2_PROMOTION_FLAG_256B,
          CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
  throw "convertUrToCul2promotion failed!";
}

CUtensorMapFloatOOBfill
convertUrToCuOobFill(ur_exp_tensor_map_oob_fill_flags_t UrType) {
  CONVERT(UR_EXP_TENSOR_MAP_OOB_FILL_FLAG_NONE,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  CONVERT(UR_EXP_TENSOR_MAP_OOB_FILL_FLAG_REQUEST_ZERO_FMA,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
  throw "convertUrToCuDataOOBfill failed!";
}

UR_APIEXPORT ur_result_t UR_APICALL urTensorMapEncodeIm2ColExp(
    ur_device_handle_t hDevice,
    ur_exp_tensor_map_data_type_flags_t TensorMapType, uint32_t TensorRank,
    void *GlobalAddress, const uint64_t *GlobalDim,
    const uint64_t *GlobalStrides, const int *PixelBoxLowerCorner,
    const int *PixelBoxUpperCorner, uint32_t ChannelsPerPixel,
    uint32_t PixelsPerColumn, const uint32_t *ElementStrides,
    ur_exp_tensor_map_interleave_flags_t Interleave,
    ur_exp_tensor_map_swizzle_flags_t Swizzle,
    ur_exp_tensor_map_l2_promotion_flags_t L2Promotion,
    ur_exp_tensor_map_oob_fill_flags_t OobFill,
    ur_exp_tensor_map_handle_t *hTensorMap) {
  ScopedContext Active(hDevice);
  try {
    UR_CHECK_ERROR(cuTensorMapEncodeIm2col(
        &(*hTensorMap)->Map, convertUrToCuDataType(TensorMapType), TensorRank,
        GlobalAddress, GlobalDim, GlobalStrides, PixelBoxLowerCorner,
        PixelBoxUpperCorner, ChannelsPerPixel, PixelsPerColumn, ElementStrides,
        convertUrToCuInterleave(Interleave), convertUrToCuSwizzle(Swizzle),
        convertUrToCuL2Promotion(L2Promotion), convertUrToCuOobFill(OobFill)));
  } catch (ur_result_t Err) {
    return Err;
  }
  return UR_RESULT_SUCCESS;
}
UR_APIEXPORT ur_result_t UR_APICALL urTensorMapEncodeTiledExp(
    ur_device_handle_t hDevice,
    ur_exp_tensor_map_data_type_flags_t TensorMapType, uint32_t TensorRank,
    void *GlobalAddress, const uint64_t *GlobalDim,
    const uint64_t *GlobalStrides, const uint32_t *BoxDim,
    const uint32_t *ElementStrides,
    ur_exp_tensor_map_interleave_flags_t Interleave,
    ur_exp_tensor_map_swizzle_flags_t Swizzle,
    ur_exp_tensor_map_l2_promotion_flags_t L2Promotion,
    ur_exp_tensor_map_oob_fill_flags_t OobFill,
    ur_exp_tensor_map_handle_t *hTensorMap) {
  ScopedContext Active(hDevice);
  try {
    UR_CHECK_ERROR(cuTensorMapEncodeTiled(
        &(*hTensorMap)->Map, convertUrToCuDataType(TensorMapType), TensorRank,
        GlobalAddress, GlobalDim, GlobalStrides, BoxDim, ElementStrides,
        convertUrToCuInterleave(Interleave), convertUrToCuSwizzle(Swizzle),
        convertUrToCuL2Promotion(L2Promotion), convertUrToCuOobFill(OobFill)));
  } catch (ur_result_t Err) {
    return Err;
  }
  return UR_RESULT_SUCCESS;
}
