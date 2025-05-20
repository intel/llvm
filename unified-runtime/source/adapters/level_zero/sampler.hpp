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

#include "common.hpp"

struct ur_sampler_handle_t_ : ur_object {
  ur_sampler_handle_t_(ze_sampler_handle_t Sampler) : ZeSampler{Sampler} {}

  // Level Zero sampler handle.
  ze_sampler_handle_t ZeSampler;

  ZeStruct<ze_sampler_desc_t> ZeSamplerDesc;
};

// Construct ZE sampler desc from UR sampler desc.
ur_result_t ur2zeSamplerDesc(ze_api_version_t ZeApiVersion,
                             const ur_sampler_desc_t *SamplerDesc,
                             ZeStruct<ze_sampler_desc_t> &ZeSamplerDesc);
