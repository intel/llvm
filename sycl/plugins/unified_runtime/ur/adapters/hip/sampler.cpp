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
  std::unique_ptr<ur_sampler_handle_t_> retImplSampl{
      new ur_sampler_handle_t_(hContext)};

  if (pDesc && pDesc->stype == UR_STRUCTURE_TYPE_SAMPLER_DESC) {
    retImplSampl->props_ |= pDesc->normalizedCoords;
    retImplSampl->props_ |= (pDesc->filterMode << 1);
    retImplSampl->props_ |= (pDesc->addressingMode << 2);
  } else {
    // Set default values
    retImplSampl->props_ |= true; // Normalized Coords
    retImplSampl->props_ |= UR_SAMPLER_ADDRESSING_MODE_CLAMP << 2;
  }

  *phSampler = retImplSampl.release();
  return UR_RESULT_SUCCESS;
}

ur_result_t urSamplerGetInfo(ur_sampler_handle_t hSampler,
                             ur_sampler_info_t propName, size_t propValueSize,
                             void *pPropValue, size_t *pPropSizeRet) {
  UR_ASSERT(hSampler, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UrReturnHelper ReturnValue(propValueSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_SAMPLER_INFO_REFERENCE_COUNT:
    return ReturnValue(hSampler->get_reference_count());
  case UR_SAMPLER_INFO_CONTEXT:
    return ReturnValue(hSampler->context_);
  case UR_SAMPLER_INFO_NORMALIZED_COORDS: {
    bool norm_coords_prop = static_cast<bool>(hSampler->props_);
    return ReturnValue(norm_coords_prop);
  }
  case UR_SAMPLER_INFO_FILTER_MODE: {
    auto filter_prop =
        static_cast<ur_sampler_filter_mode_t>(((hSampler->props_ >> 1) & 0x1));
    return ReturnValue(filter_prop);
  }
  case UR_SAMPLER_INFO_ADDRESSING_MODE: {
    auto addressing_prop =
        static_cast<ur_sampler_addressing_mode_t>(hSampler->props_ >> 2);
    return ReturnValue(addressing_prop);
  }
  default:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }
  return {};
}

ur_result_t urSamplerRetain(ur_sampler_handle_t hSampler) {
  UR_ASSERT(hSampler, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  hSampler->increment_reference_count();
  return UR_RESULT_SUCCESS;
}

ur_result_t urSamplerRelease(ur_sampler_handle_t hSampler) {
  UR_ASSERT(hSampler, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  // double delete or someone is messing with the ref count.
  // either way, cannot safely proceed.
  sycl::detail::ur::assertion(
      hSampler->get_reference_count() != 0,
      "Reference count overflow detected in urSamplerRelease.");

  // decrement ref count. If it is 0, delete the sampler.
  if (hSampler->decrement_reference_count() == 0) {
    delete hSampler;
  }

  return UR_RESULT_SUCCESS;
}
