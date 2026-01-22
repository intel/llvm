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
#include "context_interface.hpp"

namespace ur::level_zero::common {

struct ur_sampler_handle_t_ : ur_shared_handle_base_t {
  ur_sampler_handle_t_(ze_sampler_handle_t Sampler, ur_context_handle_t Context)
      : ZeSampler{Sampler} {
    // Auto-populate ddi_table from the owning context.
    ddi_table = ddiTableOf(Context);
  }

  // Level Zero sampler handle.
  ze_sampler_handle_t ZeSampler;

  ZeStruct<ze_sampler_desc_t> ZeSamplerDesc;

  ur::RefCount RefCount;
};

// Cast from opaque UR sampler handle to the common concrete type.
inline ur_sampler_handle_t_ *cast(::ur_sampler_handle_t h) {
  return reinterpret_cast<ur_sampler_handle_t_ *>(h);
}

// Construct ZE sampler desc from UR sampler desc.
ur_result_t ur2zeSamplerDesc(ze_api_version_t ZeApiVersion,
                             const ur_sampler_desc_t *SamplerDesc,
                             ZeStruct<ze_sampler_desc_t> &ZeSamplerDesc);

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
