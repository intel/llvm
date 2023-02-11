//===--------- ur_level_zero_sampler.cpp - Level Zero Adapter ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "ur_level_zero_sampler.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urSamplerCreate(
    ur_context_handle_t Context, ///< [in] handle of the context object
    const ur_sampler_property_t
        *Props, ///< [in] specifies a list of sampler property names and their
                ///< corresponding values.
    ur_sampler_handle_t
        *Sampler ///< [out] pointer to handle of sampler object created
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urSamplerRetain(
    ur_sampler_handle_t
        Sampler ///< [in] handle of the sampler object to get access
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urSamplerRelease(
    ur_sampler_handle_t
        Sampler ///< [in] handle of the sampler object to release
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urSamplerGetInfo(
    ur_sampler_handle_t Sampler, ///< [in] handle of the sampler object
    ur_sampler_info_t PropName,  ///< [in] name of the sampler property to query
    size_t PropValueSize, ///< [in] size in bytes of the sampler property value
                          ///< provided
    void *PropValue,      ///< [out] value of the sampler property
    size_t
        *PropSizeRet ///< [out] size in bytes returned in sampler property value
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urSamplerGetNativeHandle(
    ur_sampler_handle_t Sampler,      ///< [in] handle of the sampler.
    ur_native_handle_t *NativeSampler ///< [out] a pointer to the native
                                      ///< handle of the sampler.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urSamplerCreateWithNativeHandle(
    ur_native_handle_t
        NativeSampler,           ///< [in] the native handle of the sampler.
    ur_context_handle_t Context, ///< [in] handle of the context object
    ur_sampler_handle_t *Sampler ///< [out] pointer to the handle of the
                                 ///< sampler object created.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
