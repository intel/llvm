//==----- gpu_kernel_properties.hpp - Kernel properties for Intel GPUs ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::intel::experimental {

template <typename T, typename PropertyListT> class gpu_kernel_attribute;

enum class gpu_cache_config_enum : std::uint16_t { large_slm, large_data };

inline constexpr gpu_cache_config_enum large_slm =
    gpu_cache_config_enum::large_slm;
inline constexpr gpu_cache_config_enum large_data =
    gpu_cache_config_enum::large_data;

struct gpu_cache_config_key {
  template <gpu_cache_config_enum option>
  using value_t = ext::oneapi::experimental::property_value<
      gpu_cache_config_key,
      std::integral_constant<gpu_cache_config_enum, option>>;
};

template <gpu_cache_config_enum option>
inline constexpr gpu_cache_config_key::value_t<option> gpu_cache_config;

inline constexpr gpu_cache_config_key::value_t<gpu_cache_config_enum::large_slm>
    gpu_cache_config_large_slm;

inline constexpr gpu_cache_config_key::value_t<
    gpu_cache_config_enum::large_data>
    gpu_cache_config_large_data;

} // namespace ext::intel::experimental

namespace ext::oneapi::experimental {
template <>
struct is_property_key<intel::experimental::gpu_cache_config_key>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<
    intel::experimental::gpu_cache_config_key,
    intel::experimental::gpu_kernel_attribute<T, PropertyListT>>
    : std::true_type {};

namespace detail {
template <> struct PropertyToKind<intel::experimental::gpu_cache_config_key> {
  static constexpr PropKind Kind = GpuCacheConfig;
};

template <>
struct IsCompileTimeProperty<intel::experimental::gpu_cache_config_key>
    : std::true_type {};

template <intel::experimental::gpu_cache_config_enum Config>
struct PropertyMetaInfo<
    intel::experimental::gpu_cache_config_key::value_t<Config>> {
  static constexpr const char *name = "sycl-gpu-cache-config";
  static constexpr intel::experimental::gpu_cache_config_enum value = Config;
};

} // namespace detail
} // namespace ext::oneapi::experimental
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
