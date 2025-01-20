//===--------- sampler.cpp - Level Zero Adapter ---------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sampler.hpp"
#include "logger/ur_logger.hpp"
#include "ur_level_zero.hpp"

namespace ur::level_zero {

ur_result_t urSamplerCreate(
    /// [in] handle of the context object
    ur_context_handle_t Context,
    /// [in] specifies a list of sampler property names and their
    /// corresponding values.
    const ur_sampler_desc_t *Props,
    /// [out] pointer to handle of sampler object created
    ur_sampler_handle_t *Sampler) {
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

  // Update the values of the ZeSamplerDesc from the sampler properties list.
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
      logger::error("urSamplerCreate: unsupported "
                    "UR_SAMPLER_PROPERTIES_ADDRESSING_MODEE "
                    "value");
      return UR_RESULT_ERROR_INVALID_VALUE;
    }

    if (Props->filterMode == UR_SAMPLER_FILTER_MODE_NEAREST)
      ZeSamplerDesc.filterMode = ZE_SAMPLER_FILTER_MODE_NEAREST;
    else if (Props->filterMode == UR_SAMPLER_FILTER_MODE_LINEAR)
      ZeSamplerDesc.filterMode = ZE_SAMPLER_FILTER_MODE_LINEAR;
    else {
      logger::error(
          "urSamplerCreate: unsupported UR_SAMPLER_FILTER_MODE value");
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
  }

  ZE2UR_CALL(zeSamplerCreate, (Context->ZeContext, Device->ZeDevice,
                               &ZeSamplerDesc, // TODO: translate properties
                               &ZeSampler));

  try {
    ur_sampler_handle_t_ *UrSampler = new ur_sampler_handle_t_(ZeSampler);
    UrSampler->ZeSamplerDesc = ZeSamplerDesc;
    *Sampler = reinterpret_cast<ur_sampler_handle_t>(UrSampler);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urSamplerRetain(
    /// [in] handle of the sampler object to get access
    ur_sampler_handle_t Sampler) {
  Sampler->RefCount.increment();
  return UR_RESULT_SUCCESS;
}

ur_result_t urSamplerRelease(
    /// [in] handle of the sampler object to release
    ur_sampler_handle_t Sampler) {
  if (!Sampler->RefCount.decrementAndTest())
    return UR_RESULT_SUCCESS;

  auto ZeResult = ZE_CALL_NOCHECK(zeSamplerDestroy, (Sampler->ZeSampler));
  // Gracefully handle the case that L0 was already unloaded.
  if (ZeResult && ZeResult != ZE_RESULT_ERROR_UNINITIALIZED)
    return ze2urResult(ZeResult);
  delete Sampler;

  return UR_RESULT_SUCCESS;
}

ur_result_t urSamplerGetInfo(
    /// [in] handle of the sampler object
    ur_sampler_handle_t Sampler,
    /// [in] name of the sampler property to query
    ur_sampler_info_t PropName,
    /// [in] size in bytes of the sampler property value provided
    size_t PropValueSize,
    /// [out] value of the sampler property
    void *PropValue,
    /// [out] size in bytes returned in sampler property value
    size_t *PropSizeRet) {
  std::ignore = Sampler;
  std::ignore = PropName;
  std::ignore = PropValueSize;
  std::ignore = PropValue;
  std::ignore = PropSizeRet;
  logger::error(logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urSamplerGetNativeHandle(
    /// [in] handle of the sampler.
    ur_sampler_handle_t Sampler,
    /// [out] a pointer to the native handle of the sampler.
    ur_native_handle_t *NativeSampler) {
  std::ignore = Sampler;
  std::ignore = NativeSampler;
  logger::error(logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urSamplerCreateWithNativeHandle(
    /// [in] the native handle of the sampler.
    ur_native_handle_t NativeSampler,
    /// [in] handle of the context object
    ur_context_handle_t Context,
    /// [in][optional] pointer to native sampler properties struct.
    const ur_sampler_native_properties_t *Properties,
    /// [out] pointer to the handle of the sampler object created.
    ur_sampler_handle_t *Sampler) {
  std::ignore = NativeSampler;
  std::ignore = Context;
  std::ignore = Properties;
  std::ignore = Sampler;
  logger::error(logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
} // namespace ur::level_zero
