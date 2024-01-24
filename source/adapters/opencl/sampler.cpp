//===--------- sampler.cpp - OpenCL Adapter --------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"

namespace {

cl_sampler_info ur2CLSamplerInfo(ur_sampler_info_t URInfo) {
  switch (URInfo) {
#define CASE(UR_INFO, CL_INFO)                                                 \
  case UR_INFO:                                                                \
    return CL_INFO;

    CASE(UR_SAMPLER_INFO_REFERENCE_COUNT, CL_SAMPLER_REFERENCE_COUNT)
    CASE(UR_SAMPLER_INFO_CONTEXT, CL_SAMPLER_CONTEXT)
    CASE(UR_SAMPLER_INFO_NORMALIZED_COORDS, CL_SAMPLER_NORMALIZED_COORDS)
    CASE(UR_SAMPLER_INFO_ADDRESSING_MODE, CL_SAMPLER_ADDRESSING_MODE)
    CASE(UR_SAMPLER_INFO_FILTER_MODE, CL_SAMPLER_FILTER_MODE)

#undef CASE

  default:
    cl_adapter::die("Unhandled: ur_sampler_info_t");
  }
}

cl_addressing_mode ur2CLAddressingMode(ur_sampler_addressing_mode_t Mode) {
  switch (Mode) {

#define CASE(UR_MODE, CL_MODE)                                                 \
  case UR_MODE:                                                                \
    return CL_MODE;

    CASE(UR_SAMPLER_ADDRESSING_MODE_NONE, CL_ADDRESS_NONE);
    CASE(UR_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE, CL_ADDRESS_CLAMP_TO_EDGE);
    CASE(UR_SAMPLER_ADDRESSING_MODE_CLAMP, CL_ADDRESS_CLAMP);
    CASE(UR_SAMPLER_ADDRESSING_MODE_REPEAT, CL_ADDRESS_REPEAT);
    CASE(UR_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT,
         CL_ADDRESS_MIRRORED_REPEAT);

#undef CASE

  default:
    cl_adapter::die("Unhandled: ur_sampler_addressing_mode_t");
  }
}

cl_filter_mode ur2CLFilterMode(ur_sampler_filter_mode_t Mode) {
  switch (Mode) {

#define CASE(UR_MODE, CL_MODE)                                                 \
  case UR_MODE:                                                                \
    return CL_MODE;

    CASE(UR_SAMPLER_FILTER_MODE_NEAREST, CL_FILTER_NEAREST)
    CASE(UR_SAMPLER_FILTER_MODE_LINEAR, CL_FILTER_LINEAR)

#undef CASE

  default:
    cl_adapter::die("Unhandled: ur_sampler_filter_mode_t");
  }
}

ur_sampler_addressing_mode_t cl2URAddressingMode(cl_addressing_mode Mode) {
  switch (Mode) {

#define CASE(CL_MODE, UR_MODE)                                                 \
  case CL_MODE:                                                                \
    return UR_MODE;

    CASE(CL_ADDRESS_NONE, UR_SAMPLER_ADDRESSING_MODE_NONE);
    CASE(CL_ADDRESS_CLAMP_TO_EDGE, UR_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE);
    CASE(CL_ADDRESS_CLAMP, UR_SAMPLER_ADDRESSING_MODE_CLAMP);
    CASE(CL_ADDRESS_REPEAT, UR_SAMPLER_ADDRESSING_MODE_REPEAT);
    CASE(CL_ADDRESS_MIRRORED_REPEAT,
         UR_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT);

#undef CASE

  default:
    cl_adapter::die("Unhandled: cl_addressing_mode");
  }
}

ur_sampler_filter_mode_t cl2URFilterMode(cl_filter_mode Mode) {
  switch (Mode) {
#define CASE(CL_MODE, UR_MODE)                                                 \
  case CL_MODE:                                                                \
    return UR_MODE;

    CASE(CL_FILTER_NEAREST, UR_SAMPLER_FILTER_MODE_NEAREST)
    CASE(CL_FILTER_LINEAR, UR_SAMPLER_FILTER_MODE_LINEAR);

#undef CASE

  default:
    cl_adapter::die("Unhandled: cl_filter_mode");
  }
}

void cl2URSamplerInfoValue(cl_sampler_info Info, void *InfoValue) {
  if (!InfoValue) {
    return;
  }
  switch (Info) {
  case CL_SAMPLER_ADDRESSING_MODE: {
    cl_addressing_mode CLValue =
        *reinterpret_cast<cl_addressing_mode *>(InfoValue);
    *reinterpret_cast<ur_sampler_addressing_mode_t *>(InfoValue) =
        cl2URAddressingMode(CLValue);
    break;
  }
  case CL_SAMPLER_FILTER_MODE: {
    cl_filter_mode CLMode = *reinterpret_cast<cl_filter_mode *>(InfoValue);
    *reinterpret_cast<ur_sampler_filter_mode_t *>(InfoValue) =
        cl2URFilterMode(CLMode);
    break;
  }

  default:
    break;
  }
}

} // namespace

ur_result_t urSamplerCreate(ur_context_handle_t hContext,
                            const ur_sampler_desc_t *pDesc,
                            ur_sampler_handle_t *phSampler) {

  // Initialize properties according to OpenCL 2.1 spec.
  ur_result_t ErrorCode;
  cl_addressing_mode AddressingMode =
      ur2CLAddressingMode(pDesc->addressingMode);
  cl_filter_mode FilterMode = ur2CLFilterMode(pDesc->filterMode);

  // Always call OpenCL 1.0 API
  *phSampler = cl_adapter::cast<ur_sampler_handle_t>(clCreateSampler(
      cl_adapter::cast<cl_context>(hContext),
      static_cast<cl_bool>(pDesc->normalizedCoords), AddressingMode, FilterMode,
      cl_adapter::cast<cl_int *>(&ErrorCode)));

  return mapCLErrorToUR(ErrorCode);
}

UR_APIEXPORT ur_result_t UR_APICALL
urSamplerGetInfo(ur_sampler_handle_t hSampler, ur_sampler_info_t propName,
                 size_t propSize, void *pPropValue, size_t *pPropSizeRet) {
  cl_sampler_info SamplerInfo = ur2CLSamplerInfo(propName);
  static_assert(sizeof(cl_addressing_mode) ==
                sizeof(ur_sampler_addressing_mode_t));

  size_t CheckPropSize = 0;
  ur_result_t Err = mapCLErrorToUR(
      clGetSamplerInfo(cl_adapter::cast<cl_sampler>(hSampler), SamplerInfo,
                       propSize, pPropValue, &CheckPropSize));
  if (pPropValue && CheckPropSize != propSize) {
    return UR_RESULT_ERROR_INVALID_SIZE;
  }
  CL_RETURN_ON_FAILURE(Err);
  if (pPropSizeRet) {
    *pPropSizeRet = CheckPropSize;
  }

  // Convert OpenCL returns to UR
  cl2URSamplerInfoValue(SamplerInfo, pPropValue);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urSamplerRetain(ur_sampler_handle_t hSampler) {
  return mapCLErrorToUR(
      clRetainSampler(cl_adapter::cast<cl_sampler>(hSampler)));
}

UR_APIEXPORT ur_result_t UR_APICALL
urSamplerRelease(ur_sampler_handle_t hSampler) {
  return mapCLErrorToUR(
      clReleaseSampler(cl_adapter::cast<cl_sampler>(hSampler)));
}

UR_APIEXPORT ur_result_t UR_APICALL urSamplerGetNativeHandle(
    ur_sampler_handle_t hSampler, ur_native_handle_t *phNativeSampler) {
  *phNativeSampler = reinterpret_cast<ur_native_handle_t>(
      cl_adapter::cast<cl_sampler>(hSampler));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urSamplerCreateWithNativeHandle(
    ur_native_handle_t hNativeSampler, ur_context_handle_t,
    const ur_sampler_native_properties_t *pProperties,
    ur_sampler_handle_t *phSampler) {
  *phSampler = reinterpret_cast<ur_sampler_handle_t>(
      cl_adapter::cast<cl_sampler>(hNativeSampler));
  if (!pProperties || !pProperties->isNativeHandleOwned) {
    return urSamplerRetain(*phSampler);
  }
  return UR_RESULT_SUCCESS;
}
