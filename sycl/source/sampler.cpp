//==------------------- sampler.cpp - SYCL standard header file ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/sampler_impl.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/property_list.hpp>
#include <sycl/sampler.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
sampler::sampler(coordinate_normalization_mode normalizationMode,
                 addressing_mode addressingMode, filtering_mode filteringMode,
                 const property_list &propList)
    : impl(std::make_shared<detail::sampler_impl>(
          normalizationMode, addressingMode, filteringMode, propList)) {}

sampler::sampler(cl_sampler clSampler, const context &syclContext)
    : impl(std::make_shared<detail::sampler_impl>(clSampler, syclContext)) {}

addressing_mode sampler::get_addressing_mode() const {
  return impl->get_addressing_mode();
}

filtering_mode sampler::get_filtering_mode() const {
  return impl->get_filtering_mode();
}

coordinate_normalization_mode
sampler::get_coordinate_normalization_mode() const {
  return impl->get_coordinate_normalization_mode();
}

bool sampler::operator==(const sampler &rhs) const {
  return (impl == rhs.impl);
}

bool sampler::operator!=(const sampler &rhs) const {
  return !(impl == rhs.impl);
}

#define __SYCL_PARAM_TRAITS_SPEC(param_type)                                   \
  template <>                                                                  \
  __SYCL_EXPORT bool sampler::has_property<param_type>() const noexcept {      \
    return impl->has_property<param_type>();                                   \
  }
#include <sycl/detail/properties_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

#define __SYCL_PARAM_TRAITS_SPEC(param_type)                                   \
  template <>                                                                  \
  __SYCL_EXPORT param_type sampler::get_property<param_type>() const {         \
    return impl->get_property<param_type>();                                   \
  }
#include <sycl/detail/properties_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
