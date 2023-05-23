//===--------- ur_level_zero_sampler.cpp - Level Zero Adapter ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "ur_level_zero_sampler.hpp"
#include "ur_level_zero.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urSamplerCreate(
    ur_context_handle_t Context, ///< [in] handle of the context object
    const ur_sampler_desc_t
        *Props, ///< [in] specifies a list of sampler property names and their
                ///< corresponding values.
    ur_sampler_handle_t
        *Sampler ///< [out] pointer to handle of sampler object created
) {
  std::shared_lock<ur_shared_mutex> Lock(Context->Mutex);

  // Have the "0" device in context to own the sampler. Rely on Level-Zero
  // drivers to perform migration as necessary for sharing it across multiple
  // devices in the context.
  //
  // TODO: figure out if we instead need explicit copying for acessing
  // the sampler from other devices in the context.
  //
  ur_device_handle_t Device = Context->Devices[0];

  ze_sampler_handle_t ZeSampler;
  ZeStruct<ze_sampler_desc_t> ZeSamplerDesc;

  // Set the default values for the ZeSamplerDesc.
  ZeSamplerDesc.isNormalized = true;
  ZeSamplerDesc.addressMode = ZE_SAMPLER_ADDRESS_MODE_CLAMP;
  ZeSamplerDesc.filterMode = ZE_SAMPLER_FILTER_MODE_NEAREST;

  // Update the values of the ZeSamplerDesc from the pi_sampler_properties list.
  // Default values will be used if any of the following is true:
  //   a) SamplerProperties list is NULL
  //   b) SamplerProperties list is missing any properties

  if (Props) {
    ZeSamplerDesc.isNormalized = Props->normalizedCoords;

    // Level Zero runtime with API version 1.2 and lower has a bug:
    // ZE_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER is implemented as "clamp to
    // edge" and ZE_SAMPLER_ADDRESS_MODE_CLAMP is implemented as "clamp to
    // border", i.e. logic is flipped. Starting from API version 1.3 this
    // problem is going to be fixed. That's why check for API version to set
    // an address mode.
    ze_api_version_t ZeApiVersion = Context->getPlatform()->ZeApiVersion;
    // TODO: add support for PI_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE
    switch (Props->addressingMode) {
    case UR_SAMPLER_ADDRESSING_MODE_NONE:
      ZeSamplerDesc.addressMode = ZE_SAMPLER_ADDRESS_MODE_NONE;
      break;
    case UR_SAMPLER_ADDRESSING_MODE_REPEAT:
      ZeSamplerDesc.addressMode = ZE_SAMPLER_ADDRESS_MODE_REPEAT;
      break;
    case UR_SAMPLER_ADDRESSING_MODE_CLAMP:
      ZeSamplerDesc.addressMode = ZeApiVersion < ZE_MAKE_VERSION(1, 3)
                                      ? ZE_SAMPLER_ADDRESS_MODE_CLAMP
                                      : ZE_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
      break;
    case UR_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE:
      ZeSamplerDesc.addressMode = ZeApiVersion < ZE_MAKE_VERSION(1, 3)
                                      ? ZE_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER
                                      : ZE_SAMPLER_ADDRESS_MODE_CLAMP;
      break;
    case UR_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT:
      ZeSamplerDesc.addressMode = ZE_SAMPLER_ADDRESS_MODE_MIRROR;
      break;
    default:
      urPrint("urSamplerCreate: unsupported "
              "UR_SAMPLER_PROPERTIES_ADDRESSING_MODEE "
              "value\n");
      return UR_RESULT_ERROR_INVALID_VALUE;
    }

    if (Props->filterMode == UR_SAMPLER_FILTER_MODE_NEAREST)
      ZeSamplerDesc.filterMode = ZE_SAMPLER_FILTER_MODE_NEAREST;
    else if (Props->filterMode == UR_SAMPLER_FILTER_MODE_LINEAR)
      ZeSamplerDesc.filterMode = ZE_SAMPLER_FILTER_MODE_LINEAR;
    else {
      urPrint("urSamplerCreate: unsupported UR_SAMPLER_FILTER_MODE value\n");
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
  }

  ZE2UR_CALL(zeSamplerCreate, (Context->ZeContext, Device->ZeDevice,
                               &ZeSamplerDesc, // TODO: translate properties
                               &ZeSampler));

  try {
    ur_sampler_handle_t_ *UrSampler = new ur_sampler_handle_t_(ZeSampler);
    *Sampler = reinterpret_cast<ur_sampler_handle_t>(UrSampler);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urSamplerRetain(
    ur_sampler_handle_t
        Sampler ///< [in] handle of the sampler object to get access
) {
  Sampler->RefCount.increment();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urSamplerRelease(
    ur_sampler_handle_t
        Sampler ///< [in] handle of the sampler object to release
) {
  if (!Sampler->RefCount.decrementAndTest())
    return UR_RESULT_SUCCESS;

  auto ZeResult = ZE_CALL_NOCHECK(zeSamplerDestroy, (Sampler->ZeSampler));
  // Gracefully handle the case that L0 was already unloaded.
  if (ZeResult && ZeResult != ZE_RESULT_ERROR_UNINITIALIZED)
    return ze2urResult(ZeResult);
  delete Sampler;

  return UR_RESULT_SUCCESS;
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
  std::ignore = Sampler;
  std::ignore = PropName;
  std::ignore = PropValueSize;
  std::ignore = PropValue;
  std::ignore = PropSizeRet;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urSamplerGetNativeHandle(
    ur_sampler_handle_t Sampler,      ///< [in] handle of the sampler.
    ur_native_handle_t *NativeSampler ///< [out] a pointer to the native
                                      ///< handle of the sampler.
) {
  std::ignore = Sampler;
  std::ignore = NativeSampler;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urSamplerCreateWithNativeHandle(
    ur_native_handle_t
        NativeSampler,           ///< [in] the native handle of the sampler.
    ur_context_handle_t Context, ///< [in] handle of the context object
    const ur_sampler_native_properties_t
        *Properties, ///< [in][optional] pointer to native sampler properties
                     ///< struct.
    ur_sampler_handle_t *Sampler ///< [out] pointer to the handle of the
                                 ///< sampler object created.
) {
  std::ignore = NativeSampler;
  std::ignore = Context;
  std::ignore = Properties;
  std::ignore = Sampler;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
