//==----------------- sampler_impl.cpp - SYCL sampler ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/sampler_impl.hpp>

namespace cl {
namespace sycl {
namespace detail {

sampler_impl::sampler_impl(coordinate_normalization_mode normalizationMode,
                           addressing_mode addressingMode,
                           filtering_mode filteringMode)
    : m_CoordNormMode(normalizationMode), m_AddrMode(addressingMode),
      m_FiltMode(filteringMode) {}

sampler_impl::sampler_impl(cl_sampler clSampler, const context &syclContext) {

  RT::PiSampler Sampler = pi::cast<RT::PiSampler>(clSampler);
  m_contextToSampler[syclContext] = Sampler;
  PI_CALL(RT::piSamplerRetain, Sampler);
  PI_CALL(RT::piSamplerGetInfo, Sampler, PI_SAMPLER_INFO_NORMALIZED_COORDS,
          sizeof(pi_bool), &m_CoordNormMode, nullptr);
  PI_CALL(RT::piSamplerGetInfo, Sampler, PI_SAMPLER_INFO_ADDRESSING_MODE,
          sizeof(pi_sampler_addressing_mode), &m_AddrMode, nullptr);
  PI_CALL(RT::piSamplerGetInfo, Sampler, PI_SAMPLER_INFO_FILTER_MODE,
          sizeof(pi_sampler_filter_mode), &m_FiltMode, nullptr);
}

sampler_impl::~sampler_impl() {
  for (auto &Iter : m_contextToSampler) {
    // TODO catch an exception and add it to the list of asynchronous exceptions
    PI_CALL(RT::piSamplerRelease, Iter.second);
  }
}

RT::PiSampler sampler_impl::getOrCreateSampler(const context &Context) {
  if (m_contextToSampler[Context])
    return m_contextToSampler[Context];

  const pi_sampler_properties sprops[] = {
      PI_SAMPLER_INFO_NORMALIZED_COORDS,
      static_cast<pi_sampler_properties>(m_CoordNormMode),
      PI_SAMPLER_INFO_ADDRESSING_MODE,
      static_cast<pi_sampler_properties>(m_AddrMode),
      PI_SAMPLER_INFO_FILTER_MODE,
      static_cast<pi_sampler_properties>(m_FiltMode),
      0};

  RT::PiResult errcode_ret = PI_SUCCESS;
  RT::PiSampler resultSampler = nullptr;
  errcode_ret = PI_CALL_RESULT(RT::piSamplerCreate,
           getSyclObjImpl(Context)->getHandleRef(), sprops, &resultSampler);

  if (errcode_ret == PI_INVALID_OPERATION)
    throw feature_not_supported("Images are not supported by this device.");

  RT::piCheckResult(errcode_ret);
  m_contextToSampler[Context] = resultSampler;

  return m_contextToSampler[Context];
}

addressing_mode sampler_impl::get_addressing_mode() const { return m_AddrMode; }

filtering_mode sampler_impl::get_filtering_mode() const { return m_FiltMode; }

coordinate_normalization_mode
sampler_impl::get_coordinate_normalization_mode() const {
  return m_CoordNormMode;
}

} // namespace detail
} // namespace sycl
} // namespace cl
