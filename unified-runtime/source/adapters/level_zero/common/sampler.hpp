//===--------- sampler.hpp - Level Zero Adapter ---------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "../common.hpp"
#include "common/ur_ref_count.hpp"

struct ur_sampler_handle_t_ : ur_object {
  ur_sampler_handle_t_(ze_sampler_handle_t Sampler) : ZeSampler{Sampler} {}

  // Level Zero sampler handle.
  ze_sampler_handle_t ZeSampler;

  ZeStruct<ze_sampler_desc_t> ZeSamplerDesc;

  ur::RefCount RefCount;
};

// Construct ZE sampler desc from UR sampler desc.
ur_result_t ur2zeSamplerDesc(ze_api_version_t ZeApiVersion,
                             const ur_sampler_desc_t *SamplerDesc,
                             ZeStruct<ze_sampler_desc_t> &ZeSamplerDesc);

namespace ur::level_zero::common {

ur_result_t urSamplerCreate(ur_context_handle_t Context,
                            const ur_sampler_desc_t *Props,
                            ur_sampler_handle_t *Sampler);
ur_result_t urSamplerRetain(ur_sampler_handle_t Sampler);
ur_result_t urSamplerRelease(ur_sampler_handle_t Sampler);
ur_result_t urSamplerGetInfo(ur_sampler_handle_t Sampler,
                             ur_sampler_info_t PropName, size_t PropValueSize,
                             void *PropValue, size_t *PropSizeRet);
ur_result_t urSamplerGetNativeHandle(ur_sampler_handle_t Sampler,
                                     ur_native_handle_t *NativeSampler);
ur_result_t urSamplerCreateWithNativeHandle(
    ur_native_handle_t NativeSampler, ur_context_handle_t Context,
    const ur_sampler_native_properties_t *Properties,
    ur_sampler_handle_t *Sampler);

} // namespace ur::level_zero::common
