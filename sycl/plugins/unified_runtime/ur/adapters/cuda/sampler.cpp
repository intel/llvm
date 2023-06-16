//===--------- sampler.cpp - CUDA Adapter ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "sampler.hpp"
#include "common.hpp"

UR_APIEXPORT ur_result_t UR_APICALL
urSamplerCreate(ur_context_handle_t hContext, const ur_sampler_desc_t *pDesc,
                ur_sampler_handle_t *phSampler) {
  std::unique_ptr<ur_sampler_handle_t_> Sampler{
      new ur_sampler_handle_t_(hContext)};

  if (pDesc && pDesc->stype == UR_STRUCTURE_TYPE_SAMPLER_DESC) {
    Sampler->Props |= pDesc->normalizedCoords;
    Sampler->Props |= pDesc->filterMode << 1;
    Sampler->Props |= pDesc->addressingMode << 2;
  } else {
    // Set default values
    Sampler->Props |= true; // Normalized Coords
    Sampler->Props |= UR_SAMPLER_ADDRESSING_MODE_CLAMP << 2;
  }

  *phSampler = Sampler.release();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urSamplerGetInfo(ur_sampler_handle_t hSampler, ur_sampler_info_t propName,
                 size_t propValueSize, void *pPropValue, size_t *pPropSizeRet) {
  UR_ASSERT(hSampler, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UrReturnHelper ReturnValue(propValueSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_SAMPLER_INFO_REFERENCE_COUNT:
    return ReturnValue(hSampler->getReferenceCount());
  case UR_SAMPLER_INFO_CONTEXT:
    return ReturnValue(hSampler->Context);
  case UR_SAMPLER_INFO_NORMALIZED_COORDS: {
    bool NormCoordsProp = static_cast<bool>(hSampler->Props);
    return ReturnValue(NormCoordsProp);
  }
  case UR_SAMPLER_INFO_FILTER_MODE: {
    auto FilterProp =
        static_cast<ur_sampler_filter_mode_t>((hSampler->Props >> 1) & 0x1);
    return ReturnValue(FilterProp);
  }
  case UR_SAMPLER_INFO_ADDRESSING_MODE: {
    auto AddressingProp =
        static_cast<ur_sampler_addressing_mode_t>(hSampler->Props >> 2);
    return ReturnValue(AddressingProp);
  }
  default:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }
  return {};
}

UR_APIEXPORT ur_result_t UR_APICALL
urSamplerRetain(ur_sampler_handle_t hSampler) {
  UR_ASSERT(hSampler, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  hSampler->incrementReferenceCount();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urSamplerRelease(ur_sampler_handle_t hSampler) {
  UR_ASSERT(hSampler, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  // double delete or someone is messing with the ref count.
  // either way, cannot safely proceed.
  sycl::detail::ur::assertion(
      hSampler->getReferenceCount() != 0,
      "Reference count overflow detected in urSamplerRelease.");

  // decrement ref count. If it is 0, delete the sampler.
  if (hSampler->decrementReferenceCount() == 0) {
    delete hSampler;
  }

  return UR_RESULT_SUCCESS;
}
