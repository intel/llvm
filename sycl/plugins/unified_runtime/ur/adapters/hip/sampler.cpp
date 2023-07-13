//===--------- sampler.cpp - HIP Adapter ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "sampler.hpp"
#include "common.hpp"

ur_result_t urSamplerCreate(ur_context_handle_t hContext,
                            const ur_sampler_desc_t *pDesc,
                            ur_sampler_handle_t *phSampler) {
  std::unique_ptr<ur_sampler_handle_t_> RetImplSampl{
      new ur_sampler_handle_t_(hContext)};

  if (pDesc && pDesc->stype == UR_STRUCTURE_TYPE_SAMPLER_DESC) {
    RetImplSampl->Props |= pDesc->normalizedCoords;
    RetImplSampl->Props |= pDesc->filterMode << 1;
    RetImplSampl->Props |= pDesc->addressingMode << 2;
  } else {
    // Set default values
    RetImplSampl->Props |= true; // Normalized Coords
    RetImplSampl->Props |= UR_SAMPLER_ADDRESSING_MODE_CLAMP << 2;
  }

  *phSampler = RetImplSampl.release();
  return UR_RESULT_SUCCESS;
}

ur_result_t urSamplerGetInfo(ur_sampler_handle_t hSampler,
                             ur_sampler_info_t propName, size_t propValueSize,
                             void *pPropValue, size_t *pPropSizeRet) {
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

ur_result_t urSamplerRetain(ur_sampler_handle_t hSampler) {
  hSampler->incrementReferenceCount();
  return UR_RESULT_SUCCESS;
}

ur_result_t urSamplerRelease(ur_sampler_handle_t hSampler) {
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
