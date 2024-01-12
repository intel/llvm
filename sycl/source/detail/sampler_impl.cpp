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

  sycl::detail::pi::PiResult Result =
      Plugin->call_nocheck<PiApiKind::piSamplerRetain>(Sampler);
  if (Result == PI_ERROR_INVALID_OPERATION) {
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Sampler retain command not supported by backend.");
  } else {
    Plugin->checkPiResult(Result);
  }

  Result = Plugin->call_nocheck<PiApiKind::piSamplerGetInfo>(
      Sampler, PI_SAMPLER_INFO_NORMALIZED_COORDS, sizeof(pi_bool),
      &MCoordNormMode, nullptr);
  if (Result == PI_ERROR_INVALID_OPERATION) {
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Sampler get info command not supported by backend.");
  }

  Result = Plugin->call_nocheck<PiApiKind::piSamplerGetInfo>(
      Sampler, PI_SAMPLER_INFO_ADDRESSING_MODE,
      sizeof(pi_sampler_addressing_mode), &MAddrMode, nullptr);
  if (Result == PI_ERROR_INVALID_OPERATION) {
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Sampler get info command not supported by backend.");
  }

  Result = Plugin->call_nocheck<PiApiKind::piSamplerGetInfo>(
      Sampler, PI_SAMPLER_INFO_FILTER_MODE, sizeof(pi_sampler_filter_mode),
      &MFiltMode, nullptr);
  if (Result == PI_ERROR_INVALID_OPERATION) {
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Sampler get info command not supported by backend.");
  }
}

sampler_impl::~sampler_impl() {
  std::lock_guard<std::mutex> Lock(MMutex);
  for (auto &Iter : MContextToSampler) {
    // TODO catch an exception and add it to the list of asynchronous exceptions
    const PluginPtr &Plugin = getSyclObjImpl(Iter.first)->getPlugin();
    sycl::detail::pi::PiResult Result =
        Plugin->call_nocheck<PiApiKind::piSamplerRelease>(Iter.second);
    if (Result == PI_ERROR_INVALID_OPERATION) {
      assert(!"Sampler release command not supported by backend.");
    } else {
      Plugin->checkPiResult(Result);
    }
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

  sycl::detail::pi::PiSampler resultSampler = nullptr;
  const PluginPtr &Plugin = getSyclObjImpl(Context)->getPlugin();

  sycl::detail::pi::PiResult Result =
      Plugin->call_nocheck<PiApiKind::piSamplerCreate>(
          getSyclObjImpl(Context)->getHandleRef(), sprops, &resultSampler);
  if (Result == PI_ERROR_INVALID_OPERATION) {
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Sampler create command not supported by backend.");
  } else {
    Plugin->checkPiResult(Result);
  }

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
