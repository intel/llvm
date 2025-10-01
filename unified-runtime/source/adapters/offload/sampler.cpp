//===--------- sampler.cpp - LLVM Offload Adapter -------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ur_api.h"

UR_APIEXPORT ur_result_t UR_APICALL urSamplerCreate(
    ur_context_handle_t /*hContext*/, const ur_sampler_desc_t * /*pDesc*/,
    ur_sampler_handle_t * /*phSampler*/) {

  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urSamplerRetain(ur_sampler_handle_t /*hSampler*/) {

  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urSamplerRelease(ur_sampler_handle_t /*hSampler*/) {

  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urSamplerGetInfo(
    ur_sampler_handle_t /*hSampler*/, ur_sampler_info_t /*propName*/,
    size_t /*propSize*/, void * /*pPropValue*/, size_t * /*pPropSizeRet*/) {

  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urSamplerGetNativeHandle(ur_sampler_handle_t /*hSampler*/,
                         ur_native_handle_t * /*phNativeSampler*/) {

  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urSamplerCreateWithNativeHandle(
    ur_native_handle_t /*hNativeSampler*/, ur_context_handle_t /*hContext*/,
    const ur_sampler_native_properties_t * /*pProperties*/,
    ur_sampler_handle_t * /*phSampler*/) {

  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
