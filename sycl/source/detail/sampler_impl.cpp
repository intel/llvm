//==----------------- sampler_impl.cpp - SYCL sampler ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/property_list.hpp>
#include <detail/context_impl.hpp>
#include <detail/sampler_impl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

sampler_impl::sampler_impl(coordinate_normalization_mode normalizationMode,
                           addressing_mode addressingMode,
                           filtering_mode filteringMode,
                           const property_list &propList)
    : MCoordNormMode(normalizationMode), MAddrMode(addressingMode),
      MFiltMode(filteringMode), MPropList(propList) {}

sampler_impl::sampler_impl(cl_sampler clSampler, const context &syclContext) {

  RT::PiSampler Sampler = pi::cast<RT::PiSampler>(clSampler);
  MContextToSampler[syclContext] = Sampler;
  const detail::plugin &Plugin = getSyclObjImpl(syclContext)->getPlugin();
  Plugin.call<PiApiKind::piSamplerRetain>(Sampler);
  Plugin.call<PiApiKind::piSamplerGetInfo>(
      Sampler, PI_SAMPLER_INFO_NORMALIZED_COORDS, sizeof(pi_bool),
      &MCoordNormMode, nullptr);
  Plugin.call<PiApiKind::piSamplerGetInfo>(
      Sampler, PI_SAMPLER_INFO_ADDRESSING_MODE,
      sizeof(pi_sampler_addressing_mode), &MAddrMode, nullptr);
  Plugin.call<PiApiKind::piSamplerGetInfo>(Sampler, PI_SAMPLER_INFO_FILTER_MODE,
                                           sizeof(pi_sampler_filter_mode),
                                           &MFiltMode, nullptr);
}

sampler_impl::~sampler_impl() {
  std::lock_guard<std::mutex> Lock(MMutex);
  for (auto &Iter : MContextToSampler) {
    // TODO catch an exception and add it to the list of asynchronous exceptions
    const detail::plugin &Plugin = getSyclObjImpl(Iter.first)->getPlugin();
    Plugin.call<PiApiKind::piSamplerRelease>(Iter.second);
  }
}

RT::PiSampler sampler_impl::getOrCreateSampler(const context &Context) {
  {
    std::lock_guard<std::mutex> Lock(MMutex);
    auto It = MContextToSampler.find(Context);
    if (It != MContextToSampler.end())
      return It->second;
  }

  const pi_sampler_properties sprops[] = {
      PI_SAMPLER_INFO_NORMALIZED_COORDS,
      static_cast<pi_sampler_properties>(MCoordNormMode),
      PI_SAMPLER_INFO_ADDRESSING_MODE,
      static_cast<pi_sampler_properties>(MAddrMode),
      PI_SAMPLER_INFO_FILTER_MODE,
      static_cast<pi_sampler_properties>(MFiltMode),
      0};

  RT::PiResult errcode_ret = PI_SUCCESS;
  RT::PiSampler resultSampler = nullptr;
  const detail::plugin &Plugin = getSyclObjImpl(Context)->getPlugin();

  errcode_ret = Plugin.call_nocheck<PiApiKind::piSamplerCreate>(
      getSyclObjImpl(Context)->getHandleRef(), sprops, &resultSampler);

  if (errcode_ret == PI_INVALID_OPERATION)
    throw feature_not_supported("Images are not supported by this device.",
                                errcode_ret);

  Plugin.checkPiResult(errcode_ret);
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
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
