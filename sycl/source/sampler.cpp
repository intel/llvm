//==------------------- sampler.cpp - SYCL standard header file ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/sampler_impl.hpp>
#include <sycl/image.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/property_list.hpp>
#include <sycl/sampler.hpp>

namespace sycl {
inline namespace _V1 {
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

const property_list &sampler::getPropList() const {
  return impl->getPropList();
}

#undef __SYCL_PARAM_TRAITS_SPEC

} // namespace _V1
} // namespace sycl
