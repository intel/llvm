//==----- gpu_kernel_properties.hpp - Kernel properties for Intel GPUs ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/properties/property.hpp> // for PropKind, IsRunti...

#include <cstdint>     // for uint16_t
#include <type_traits> // for true_type

namespace sycl {
inline namespace _V1 {
namespace ext::intel::experimental {

template <typename T, typename PropertyListT> class gpu_kernel_attribute;

enum class cache_config_enum : std::uint16_t { large_slm, large_data };

inline constexpr cache_config_enum large_slm =
    cache_config_enum::large_slm;
inline constexpr cache_config_enum large_data =
    cache_config_enum::large_data;

struct cache_config : oneapi::experimental::detail::run_time_property_key<
                          oneapi::experimental::detail::PropKind::CacheConfig> {
  cache_config(cache_config_enum v) : value(v) {}
  cache_config_enum value;
};

using cache_config_key = cache_config;

inline bool operator==(const cache_config &lhs,
                       const cache_config &rhs) {
  return lhs.value == rhs.value;
}
inline bool operator!=(const cache_config &lhs,
                       const cache_config &rhs) {
  return !(lhs == rhs);
}

} // namespace ext::intel::experimental

namespace ext::oneapi::experimental {
template <typename T, typename PropertyListT>
struct is_property_key_of<
    intel::experimental::cache_config_key,
    intel::experimental::gpu_kernel_attribute<T, PropertyListT>>
    : std::true_type {};

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
