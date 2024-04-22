//==- grf_size_properties.hpp - GRF size kernel properties for Intel GPUs -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/kernel_properties.hpp>
#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

#define SYCL_EXT_INTEL_GRF_SIZE 1

namespace sycl {
inline namespace _V1 {
namespace ext::intel::experimental {
struct grf_size_key : oneapi::experimental::detail::compile_time_property_key<
                          oneapi::experimental::detail::PropKind::GRFSize> {
  template <unsigned int Size>
  using value_t = oneapi::experimental::property_value<
      grf_size_key, std::integral_constant<unsigned int, Size>>;
};

struct grf_size_automatic_key
    : oneapi::experimental::detail::compile_time_property_key<
          oneapi::experimental::detail::PropKind::GRFSizeAutomatic> {
  using value_t = oneapi::experimental::property_value<grf_size_automatic_key>;
};

template <unsigned int Size>
inline constexpr grf_size_key::value_t<Size> grf_size;

inline constexpr grf_size_automatic_key::value_t grf_size_automatic;

} // namespace ext::intel::experimental
namespace ext::oneapi::experimental::detail {
template <unsigned int Size>
struct PropertyMetaInfo<
    sycl::ext::intel::experimental::grf_size_key::value_t<Size>> {
  static_assert(Size == 128 || Size == 256, "Unsupported GRF size");
  static constexpr const char *name = "sycl-grf-size";
  static constexpr unsigned int value = Size;
};
template <>
struct PropertyMetaInfo<
    sycl::ext::intel::experimental::grf_size_automatic_key::value_t> {
  static constexpr const char *name = "sycl-grf-size";
  static constexpr unsigned int value = 0;
};

template <typename Properties>
struct ConflictingProperties<sycl::ext::intel::experimental::grf_size_key,
                             Properties>
    : std::bool_constant<
          ContainsProperty<
              sycl::ext::intel::experimental::grf_size_automatic_key,
              Properties>::value ||
          ContainsProperty<sycl::detail::register_alloc_mode_key,
                           Properties>::value> {};

template <typename Properties>
struct ConflictingProperties<
    sycl::ext::intel::experimental::grf_size_automatic_key, Properties>
    : std::bool_constant<
          ContainsProperty<sycl::ext::intel::experimental::grf_size_key,
                           Properties>::value ||
          ContainsProperty<sycl::detail::register_alloc_mode_key,
                           Properties>::value> {};

template <typename Properties>
struct ConflictingProperties<sycl::detail::register_alloc_mode_key, Properties>
    : std::bool_constant<
          ContainsProperty<sycl::ext::intel::experimental::grf_size_key,
                           Properties>::value ||
          ContainsProperty<
              sycl::ext::intel::experimental::grf_size_automatic_key,
              Properties>::value> {};

} // namespace ext::oneapi::experimental::detail
} // namespace _V1
} // namespace sycl
