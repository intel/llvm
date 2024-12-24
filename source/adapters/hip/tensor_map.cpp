//===--------- tensor_map.cpp - HIP Adapter -------------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <ur_api.h>

UR_APIEXPORT ur_result_t UR_APICALL urTensorMapEncodeIm2ColExp(
    ur_device_handle_t, ur_exp_tensor_map_data_type_flags_t, uint32_t, void *,
    const uint64_t *, const uint64_t *, const int *, const int *, uint32_t,
    uint32_t, const uint32_t *, ur_exp_tensor_map_interleave_flags_t,
    ur_exp_tensor_map_swizzle_flags_t, ur_exp_tensor_map_l2_promotion_flags_t,
    ur_exp_tensor_map_oob_fill_flags_t, ur_exp_tensor_map_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
UR_APIEXPORT ur_result_t UR_APICALL urTensorMapEncodeTiledExp(
    ur_device_handle_t, ur_exp_tensor_map_data_type_flags_t, uint32_t, void *,
    const uint64_t *, const uint64_t *, const uint32_t *, const uint32_t *,
    ur_exp_tensor_map_interleave_flags_t, ur_exp_tensor_map_swizzle_flags_t,
    ur_exp_tensor_map_l2_promotion_flags_t, ur_exp_tensor_map_oob_fill_flags_t,
    ur_exp_tensor_map_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
