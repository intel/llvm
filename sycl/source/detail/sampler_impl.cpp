//==----------------- sampler_impl.cpp - SYCL sampler ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/context_impl.hpp>
#include <detail/sampler_impl.hpp>
#include <sycl/property_list.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

sampler_impl::sampler_impl(coordinate_normalization_mode normalizationMode,
                           addressing_mode addressingMode,
                           filtering_mode filteringMode,
                           const property_list &propList)
    : MCoordNormMode(normalizationMode), MAddrMode(addressingMode),
      MFiltMode(filteringMode), MPropList(propList) {}

sampler_impl::sampler_impl(cl_sampler clSampler, const context &syclContext) {
  const PluginPtr &Plugin = getSyclObjImpl(syclContext)->getPlugin();
  ur_sampler_handle_t Sampler{};
  Plugin->call<UrApiKind::urSamplerCreateWithNativeHandle>(
      reinterpret_cast<ur_native_handle_t>(clSampler),
      getSyclObjImpl(syclContext)->getHandleRef(), nullptr, &Sampler);

  MContextToSampler[syclContext] = Sampler;
  bool NormalizedCoords;

  Plugin->call<UrApiKind::urSamplerGetInfo>(
      Sampler, UR_SAMPLER_INFO_NORMALIZED_COORDS, sizeof(ur_bool_t),
      &NormalizedCoords, nullptr);
  MCoordNormMode = NormalizedCoords
                       ? coordinate_normalization_mode::normalized
                       : coordinate_normalization_mode::unnormalized;

  ur_sampler_addressing_mode_t AddrMode;
  Plugin->call<UrApiKind::urSamplerGetInfo>(
      Sampler, UR_SAMPLER_INFO_ADDRESSING_MODE,
      sizeof(ur_sampler_addressing_mode_t), &AddrMode, nullptr);
  switch (AddrMode) {
  case UR_SAMPLER_ADDRESSING_MODE_CLAMP:
    MAddrMode = addressing_mode::clamp;
    break;
  case UR_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE:
    MAddrMode = addressing_mode::clamp_to_edge;
    break;
  case UR_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT:
    MAddrMode = addressing_mode::mirrored_repeat;
    break;
  case UR_SAMPLER_ADDRESSING_MODE_REPEAT:
    MAddrMode = addressing_mode::repeat;
    break;
  case UR_SAMPLER_ADDRESSING_MODE_NONE:
  default:
    MAddrMode = addressing_mode::none;
    break;
  }

  ur_sampler_filter_mode_t FiltMode;
  Plugin->call<UrApiKind::urSamplerGetInfo>(
      Sampler, UR_SAMPLER_INFO_FILTER_MODE, sizeof(ur_sampler_filter_mode_t),
      &FiltMode, nullptr);
  switch (FiltMode) {
  case UR_SAMPLER_FILTER_MODE_LINEAR:
    MFiltMode = filtering_mode::linear;
    break;
  case UR_SAMPLER_FILTER_MODE_NEAREST:
  default:
    MFiltMode = filtering_mode::nearest;
    break;
  }
}

sampler_impl::~sampler_impl() {
  try {
    std::lock_guard<std::mutex> Lock(MMutex);
    for (auto &Iter : MContextToSampler) {
      // TODO catch an exception and add it to the list of asynchronous
      // exceptions
      const PluginPtr &Plugin = getSyclObjImpl(Iter.first)->getPlugin();
      Plugin->call<UrApiKind::urSamplerRelease>(Iter.second);
    }
  } catch (std::exception &e) {
    __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~sample_impl", e);
  }
}

ur_sampler_handle_t sampler_impl::getOrCreateSampler(const context &Context) {
  {
    std::lock_guard<std::mutex> Lock(MMutex);
    auto It = MContextToSampler.find(Context);
    if (It != MContextToSampler.end())
      return It->second;
  }

  ur_sampler_desc_t desc{};
  desc.stype = UR_STRUCTURE_TYPE_SAMPLER_DESC;
  switch (MAddrMode) {
  case addressing_mode::clamp:
    desc.addressingMode = UR_SAMPLER_ADDRESSING_MODE_CLAMP;
    break;
  case addressing_mode::clamp_to_edge:
    desc.addressingMode = UR_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE;
    break;
  case addressing_mode::repeat:
    desc.addressingMode = UR_SAMPLER_ADDRESSING_MODE_REPEAT;
    break;
  case addressing_mode::none:
    desc.addressingMode = UR_SAMPLER_ADDRESSING_MODE_NONE;
    break;
  case addressing_mode::mirrored_repeat:
    desc.addressingMode = UR_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT;
    break;
  }
  switch (MFiltMode) {
  case filtering_mode::linear:
    desc.filterMode = UR_SAMPLER_FILTER_MODE_LINEAR;
    break;
  case filtering_mode::nearest:
    desc.filterMode = UR_SAMPLER_FILTER_MODE_NEAREST;
    break;
  }
  desc.normalizedCoords =
      (MCoordNormMode == coordinate_normalization_mode::normalized);

  ur_result_t errcode_ret = UR_RESULT_SUCCESS;
  ur_sampler_handle_t resultSampler = nullptr;
  const PluginPtr &Plugin = getSyclObjImpl(Context)->getPlugin();

  errcode_ret = Plugin->call_nocheck<UrApiKind::urSamplerCreate>(
      getSyclObjImpl(Context)->getHandleRef(), &desc, &resultSampler);

  if (errcode_ret == UR_RESULT_ERROR_UNSUPPORTED_FEATURE)
    throw sycl::exception(sycl::errc::feature_not_supported,
                          "Images are not supported by this device.");

  Plugin->checkUrResult(errcode_ret);
  std::lock_guard<std::mutex> Lock(MMutex);
  MContextToSampler[Context] = resultSampler;

  return resultSampler;
}

addressing_mode sampler_impl::get_addressing_mode() const { return MAddrMode; }

filtering_mode sampler_impl::get_filtering_mode() const { return MFiltMode; }

coordinate_normalization_mode
sampler_impl::get_coordinate_normalization_mode() const {
  return MCoordNormMode;
}

} // namespace detail
} // namespace _V1
} // namespace sycl
