//===--------- sampler.cpp - CUDA Adapter ---------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sampler.hpp"
#include "common.hpp"

UR_APIEXPORT ur_result_t UR_APICALL
urSamplerCreate(ur_context_handle_t hContext, const ur_sampler_desc_t *pDesc,
                ur_sampler_handle_t *phSampler) {
  std::unique_ptr<ur_sampler_handle_t_> Sampler{
      new ur_sampler_handle_t_(hContext)};

  if (pDesc->stype == UR_STRUCTURE_TYPE_SAMPLER_DESC) {
    Sampler->Props |= static_cast<uint32_t>(pDesc->normalizedCoords);
    Sampler->Props |= pDesc->filterMode << 1;
    Sampler->Props |= pDesc->addressingMode << 2;
  } else {
    // Set default values
    Sampler->Props |= true; // Normalized Coords
    Sampler->Props |= UR_SAMPLER_ADDRESSING_MODE_CLAMP << 2;
  }

  void *pNext = const_cast<void *>(pDesc->pNext);
  while (pNext != nullptr) {
    const ur_base_desc_t *BaseDesc =
        reinterpret_cast<const ur_base_desc_t *>(pNext);
    if (BaseDesc->stype == UR_STRUCTURE_TYPE_EXP_SAMPLER_MIP_PROPERTIES) {
      const ur_exp_sampler_mip_properties_t *SamplerMipProperties =
          reinterpret_cast<const ur_exp_sampler_mip_properties_t *>(pNext);
      Sampler->MaxMipmapLevelClamp = SamplerMipProperties->maxMipmapLevelClamp;
      Sampler->MinMipmapLevelClamp = SamplerMipProperties->minMipmapLevelClamp;
      Sampler->MaxAnisotropy = SamplerMipProperties->maxAnisotropy;
      Sampler->Props |= SamplerMipProperties->mipFilterMode << 5;
    }
    pNext = const_cast<void *>(BaseDesc->pNext);
  }

  *phSampler = Sampler.release();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urSamplerGetInfo(ur_sampler_handle_t hSampler, ur_sampler_info_t propName,
                 size_t propValueSize, void *pPropValue, size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propValueSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_SAMPLER_INFO_REFERENCE_COUNT:
    return ReturnValue(hSampler->getReferenceCount());
  case UR_SAMPLER_INFO_CONTEXT:
    return ReturnValue(hSampler->Context);
  case UR_SAMPLER_INFO_NORMALIZED_COORDS: {
    bool NormCoordsProp = hSampler->isNormalizedCoords();
    return ReturnValue(NormCoordsProp);
  }
  case UR_SAMPLER_INFO_FILTER_MODE: {
    ur_sampler_filter_mode_t FilterProp = hSampler->getFilterMode();
    return ReturnValue(FilterProp);
  }
  case UR_SAMPLER_INFO_ADDRESSING_MODE: {
    ur_sampler_addressing_mode_t AddressingProp = hSampler->getAddressingMode();
    return ReturnValue(AddressingProp);
  }
  default:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL
urSamplerRetain(ur_sampler_handle_t hSampler) {
  hSampler->incrementReferenceCount();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urSamplerRelease(ur_sampler_handle_t hSampler) {
  // double delete or someone is messing with the ref count.
  // either way, cannot safely proceed.
  detail::ur::assertion(
      hSampler->getReferenceCount() != 0,
      "Reference count overflow detected in urSamplerRelease.");

  // decrement ref count. If it is 0, delete the sampler.
  if (hSampler->decrementReferenceCount() == 0) {
    delete hSampler;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urSamplerGetNativeHandle(ur_sampler_handle_t, ur_native_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urSamplerCreateWithNativeHandle(
    ur_native_handle_t, ur_context_handle_t,
    const ur_sampler_native_properties_t *, ur_sampler_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
