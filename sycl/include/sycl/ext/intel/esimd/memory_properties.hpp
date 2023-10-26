//==-- memory_properties.hpp - ESIMD memory properties ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/intel/experimental/fpga_utils.hpp>
#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

#define SYCL_EXT_INTEL_ESIMD_MEMORY_PROPERTIES 1

namespace sycl {
inline namespace _V1 {
namespace ext::intel::esimd {

template <typename PropertiesT>
class properties
    : public sycl::ext::oneapi::experimental::properties<PropertiesT> {
public:
  template <typename... PropertyValueTs>
  constexpr properties(PropertyValueTs... props)
      : sycl::ext::oneapi::experimental::properties<PropertiesT>(props...) {}
};

#ifdef __cpp_deduction_guides
// Deduction guides
template <typename... PropertyValueTs>
properties(PropertyValueTs... props)
    -> properties<typename sycl::ext::oneapi::experimental::detail::Sorted<
        PropertyValueTs...>::type>;
#endif

/// The 'alignment' property is used to specify the alignment of memory
/// accessed in ESIMD memory operation such as block_load().
//
/// Case1: ESIMD memory operation accepts only USM pointer parameter.
/// The 'alignment' property specifies the alignment of the memory accessed by
/// the USM pointer.
//
/// Case2: ESIMD memory operation accepts two parameters:
/// either (accessor + offset) or (USM pointer + offset). The 'alignment'
/// property specifies the alignment of the memory accesed by those two
/// parameters.

using alignment_key = sycl::ext::oneapi::experimental::alignment_key;

template <int K> inline constexpr alignment_key::value_t<K> alignment;

/// The 'cache_hint_L1', 'cache_hint_L2' and 'cache_hint_L3' properties
/// are used to specify L1, L2, L3 cache hints available in target device.
/// L1 cache is usually the fastest memory closest to the processor.
/// L2 is the next level cache (slower and farther from the processor), etc.
/// L2 cache hint property must be used for the old/experimental LSC L3 cache
/// hints.
/// L3 cache property is reserved for future devices.
struct cache_hint_L1_key {
  template <cache_hint Hint>
  using value_t = ext::oneapi::experimental::property_value<
      cache_hint_L1_key, std::integral_constant<cache_hint, Hint>>;
};
struct cache_hint_L2_key {
  template <cache_hint Hint>
  using value_t = ext::oneapi::experimental::property_value<
      cache_hint_L2_key, std::integral_constant<cache_hint, Hint>>;
};
struct cache_hint_L3_key {
  template <cache_hint Hint>
  using value_t = ext::oneapi::experimental::property_value<
      cache_hint_L3_key, std::integral_constant<cache_hint, Hint>>;
};

template <cache_hint Hint>
inline constexpr cache_hint_L1_key::value_t<Hint> cache_hint_L1;
template <cache_hint Hint>
inline constexpr cache_hint_L2_key::value_t<Hint> cache_hint_L2;
template <cache_hint Hint>
inline constexpr cache_hint_L3_key::value_t<Hint> cache_hint_L3;

#if 0
// TODO: Introduce the 2-parameter cache_hint property after looking
// for a better name for it. It cannot be 'esimd::cache_hint' because that
// may conflict with the enum name 'experimental::esimd::cache_hint' when
// both namespaces (esimd and experimental::esimd) are imported with 'using'
// statement. 
// Naming alternatives: 'esimd::esimd_cache_hint, esimd::cache_hint_L,
// esimd::cache_hint_property'.
template <cache_level Level, cache_hint Hint>
inline constexpr std::conditional_t<
    Level == cache_level::L1, cache_hint_L1_key::value_t<Hint>,
    std::conditional_t<Level == cache_level::L2,
                       cache_hint_L2_key::value_t<Hint>,
                       cache_hint_L3_key::value_t<Hint>>>
    cache_hint; // Get a non-conflicting name
#endif

using default_cache_hint_L1 = cache_hint_L1_key::value_t<cache_hint::none>;
using default_cache_hint_L2 = cache_hint_L2_key::value_t<cache_hint::none>;
using default_cache_hint_L3 = cache_hint_L3_key::value_t<cache_hint::none>;

namespace detail {
/// Helper-function that returns the value of the compile time property `KeyT`
/// if `PropertiesT` includes it. If it does not then the default value
/// \p DefaultValue is returned.
template <typename PropertiesT, typename KeyT, typename KeyValueT,
          typename = std::enable_if_t<
              ext::oneapi::experimental::is_property_list_v<PropertiesT>>>
constexpr auto getPropertyValue(KeyValueT DefaultValue) {
  if constexpr (!PropertiesT::template has_property<KeyT>()) {
    return DefaultValue;
  } else if constexpr (std::is_same_v<KeyT, cache_hint_L1_key> ||
                       std::is_same_v<KeyT, cache_hint_L2_key> ||
                       std::is_same_v<KeyT, cache_hint_L3_key>) {
    constexpr auto ValueT = PropertiesT::template get_property<KeyT>();
    return ValueT.hint;
  } else {
    constexpr auto ValueT = PropertiesT::template get_property<KeyT>();
    return ValueT.value;
  }
}
} // namespace detail

} // namespace ext::intel::esimd

namespace ext::oneapi::experimental {

template <__ESIMD_NS::cache_hint Hint>
struct property_value<__ESIMD_NS::cache_hint_L1_key,
                      std::integral_constant<__ESIMD_NS::cache_hint, Hint>> {
  using key_t = __ESIMD_NS::cache_hint_L1_key;
  static constexpr __ESIMD_NS::cache_level level = __ESIMD_NS::cache_level::L1;
  static constexpr __ESIMD_NS::cache_hint hint = Hint;
};
template <__ESIMD_NS::cache_hint Hint>
struct property_value<__ESIMD_NS::cache_hint_L2_key,
                      std::integral_constant<__ESIMD_NS::cache_hint, Hint>> {
  using key_t = __ESIMD_NS::cache_hint_L2_key;
  static constexpr __ESIMD_NS::cache_level level = __ESIMD_NS::cache_level::L2;
  static constexpr __ESIMD_NS::cache_hint hint = Hint;
};
template <__ESIMD_NS::cache_hint Hint>
struct property_value<__ESIMD_NS::cache_hint_L3_key,
                      std::integral_constant<__ESIMD_NS::cache_hint, Hint>> {
  using key_t = __ESIMD_NS::cache_hint_L3_key;
  static constexpr __ESIMD_NS::cache_level level = __ESIMD_NS::cache_level::L3;
  static constexpr __ESIMD_NS::cache_hint hint = Hint;
};

template <>
struct is_property_key<sycl::ext::intel::esimd::cache_hint_L1_key>
    : std::true_type {};
template <>
struct is_property_key<sycl::ext::intel::esimd::cache_hint_L2_key>
    : std::true_type {};
template <>
struct is_property_key<sycl::ext::intel::esimd::cache_hint_L3_key>
    : std::true_type {};

// Declare that esimd::properties is a property_list.
template <typename... PropertyValueTs>
struct is_property_list<__ESIMD_NS::properties<std::tuple<PropertyValueTs...>>>
    : is_property_list<properties<std::tuple<PropertyValueTs...>>> {};

namespace detail {
template <> struct PropertyToKind<sycl::ext::intel::esimd::cache_hint_L1_key> {
  static constexpr PropKind Kind = PropKind::ESIMDL1CacheHint;
};
template <> struct PropertyToKind<sycl::ext::intel::esimd::cache_hint_L2_key> {
  static constexpr PropKind Kind = PropKind::ESIMDL2CacheHint;
};
template <> struct PropertyToKind<sycl::ext::intel::esimd::cache_hint_L3_key> {
  static constexpr PropKind Kind = PropKind::ESIMDL3CacheHint;
};

template <>
struct IsCompileTimeProperty<__ESIMD_NS::cache_hint_L1_key> : std::true_type {};
template <>
struct IsCompileTimeProperty<__ESIMD_NS::cache_hint_L2_key> : std::true_type {};
template <>
struct IsCompileTimeProperty<__ESIMD_NS::cache_hint_L3_key> : std::true_type {};

// We do not override the class ConflictingProperties for cache_hint properties
// because that mechanism would only allow to verify few obvious restrictions
// without the knowledge of the context in which the cache_hint properties are
// used (load, store, prefetch, atomic). Thus the function
// __ESIMD_DNS::check_cache_hint() is used to verify correctness of properties.

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
