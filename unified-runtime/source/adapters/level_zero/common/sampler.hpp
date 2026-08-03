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

#include "api.hpp"
#include "common/ur_ref_count.hpp"
#include "helpers/shared_helpers.hpp"
#include "interfaces.hpp"

namespace ur::level_zero {

struct ur_sampler_handle_t_ : ur::level_zero::ur_object_t {
  ur_sampler_handle_t_(ze_sampler_handle_t Sampler, ur_context_handle_t Context)
      : ZeSampler{Sampler} {
    // Populate ddi_table from the owning context.
    ddi_table = ddiTableOf(Context);
  }

  // Level Zero sampler handle.
  ze_sampler_handle_t ZeSampler;

  ZeStruct<ze_sampler_desc_t> ZeSamplerDesc;

  ur::RefCount RefCount;
};

// Construct ZE sampler desc from UR sampler desc.
ur_result_t ur2zeSamplerDesc(ze_api_version_t ZeApiVersion,
                             const ur_sampler_desc_t *SamplerDesc,
                             ZeStruct<ze_sampler_desc_t> &ZeSamplerDesc);

} // namespace ur::level_zero
