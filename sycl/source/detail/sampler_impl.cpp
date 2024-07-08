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

  sycl::detail::pi::PiSampler Sampler =
      pi::cast<sycl::detail::pi::PiSampler>(clSampler);
  MContextToSampler[syclContext] = Sampler;
  const PluginPtr &Plugin = getSyclObjImpl(syclContext)->getPlugin();
  Plugin->call<PiApiKind::piSamplerRetain>(Sampler);
  Plugin->call<PiApiKind::piSamplerGetInfo>(
      Sampler, PI_SAMPLER_INFO_NORMALIZED_COORDS, sizeof(pi_bool),
      &MCoordNormMode, nullptr);
  Plugin->call<PiApiKind::piSamplerGetInfo>(
      Sampler, PI_SAMPLER_INFO_ADDRESSING_MODE,
      sizeof(pi_sampler_addressing_mode), &MAddrMode, nullptr);
  Plugin->call<PiApiKind::piSamplerGetInfo>(
      Sampler, PI_SAMPLER_INFO_FILTER_MODE, sizeof(pi_sampler_filter_mode),
      &MFiltMode, nullptr);
}

sampler_impl::~sampler_impl() {
  try {
    std::lock_guard<std::mutex> Lock(MMutex);
    for (auto &Iter : MContextToSampler) {
      // TODO catch an exception and add it to the list of asynchronous
      // exceptions
      const PluginPtr &Plugin = getSyclObjImpl(Iter.first)->getPlugin();
      Plugin->call<PiApiKind::piSamplerRelease>(Iter.second);
    }
  } catch (std::exception &e) {
    __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~sample_impl", e);
  }
}

sycl::detail::pi::PiSampler
sampler_impl::getOrCreateSampler(const context &Context) {
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

  sycl::detail::pi::PiResult errcode_ret = PI_SUCCESS;
  sycl::detail::pi::PiSampler resultSampler = nullptr;
  const PluginPtr &Plugin = getSyclObjImpl(Context)->getPlugin();

  errcode_ret = Plugin->call_nocheck<PiApiKind::piSamplerCreate>(
      getSyclObjImpl(Context)->getHandleRef(), sprops, &resultSampler);

  if (errcode_ret == PI_ERROR_UNSUPPORTED_FEATURE)
    throw sycl::exception(sycl::errc::feature_not_supported,
                          "Images are not supported by this device.");

  Plugin->checkPiResult(errcode_ret);
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
