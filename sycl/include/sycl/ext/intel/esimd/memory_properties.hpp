//==-- memory_properties.hpp - ESIMD memory properties ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/intel/experimental/fpga_utils.hpp>
#include <sycl/ext/oneapi/experimental/common_annotated_properties/properties.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>
#include <utility>

#define SYCL_EXT_INTEL_ESIMD_MEMORY_PROPERTIES 1

namespace sycl {
inline namespace _V1 {
namespace ext::intel::esimd {

/// L1, L2 or L3 cache hint levels. L3 is reserved for future use.
enum class cache_level : uint8_t { L1 = 1, L2 = 2, L3 = 3 };

/// L1, L2 or L3 cache hints.
enum class cache_hint : uint8_t {
  none = 0,
  /// load/store/atomic: do not cache data to cache;
  uncached = 1,

  // load: cache data to cache;
  cached = 2,

  /// store: write data into cache level and mark the cache line as "dirty".
  /// Upon eviction, the "dirty" data will be written into the furthest
  /// subsequent cache;
  write_back = 3,

  /// store: immediately write data to the subsequent furthest cache, marking
  /// the cache line in the current cache as "not dirty";
  write_through = 4,

  /// load: cache data to cache using the evict-first policy to minimize cache
  /// pollution caused by temporary streaming data that may only be accessed
  /// once or twice;
  /// store/atomic: same as write-through, but use the evict-first policy
  /// to limit cache pollution by streaming;
  streaming = 5,

  /// load: asserts that the cache line containing the data will not be read
  /// again until it’s overwritten, therefore the load operation can invalidate
  /// the cache line and discard "dirty" data. If the assertion is violated
  /// (the cache line is read again) then behavior is undefined.
  read_invalidate = 6,

  // TODO: Implement the verification of this enum in check_cache_hint().
  /// load, L2 cache only, next gen GPU after Xe required: asserts that
  /// the L2 cache line containing the data will not be written until all
  /// invocations of the shader or kernel execution are finished.
  /// If the assertion is violated (the cache line is written), the behavior
  /// is undefined.
  const_cached = 7
};

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
struct cache_hint_L1_key
    : oneapi::experimental::detail::compile_time_property_key<
          oneapi::experimental::detail::PropKind::ESIMDL1CacheHint> {
  template <cache_hint Hint>
  using value_t = ext::oneapi::experimental::property_value<
      cache_hint_L1_key, std::integral_constant<cache_hint, Hint>>;
};
struct cache_hint_L2_key
    : oneapi::experimental::detail::compile_time_property_key<
          oneapi::experimental::detail::PropKind::ESIMDL2CacheHint> {
  template <cache_hint Hint>
  using value_t = ext::oneapi::experimental::property_value<
      cache_hint_L2_key, std::integral_constant<cache_hint, Hint>>;
};
struct cache_hint_L3_key
    : oneapi::experimental::detail::compile_time_property_key<
          oneapi::experimental::detail::PropKind::ESIMDL3CacheHint> {
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

template <typename PropsT>
using is_property_list = ext::oneapi::experimental::is_property_list<PropsT>;

template <typename PropsT>
inline constexpr bool is_property_list_v = is_property_list<PropsT>::value;

/// Helper-function that returns the value of the compile time property `KeyT`
/// if `PropertiesT` includes it. If it does not then the default value
/// \p DefaultValue is returned.
template <typename PropertiesT, typename KeyT, typename KeyValueT,
          typename = std::enable_if_t<is_property_list_v<PropertiesT>>>
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

/// This helper returns the ext::oneapi::experimental::properties class for
/// ext::oneapi::experimental::properties and it's child in esimd namespace.
template <typename PropertiesT> struct get_ext_oneapi_properties;
template <typename PropertiesT>
struct get_ext_oneapi_properties<
    ext::oneapi::experimental::properties<PropertiesT>> {
  using type = ext::oneapi::experimental::properties<PropertiesT>;
};
template <typename PropertiesT>
struct get_ext_oneapi_properties<properties<PropertiesT>> {
  using type = ext::oneapi::experimental::properties<PropertiesT>;
};

/// Simply returns 'PropertyListT' as it already has the alignment property.
template <typename PropertyListT, size_t Alignment, bool HasAlignment = true>
struct add_alignment_property_helper {
  using type = PropertyListT;
};
/// Returns a new property list type that contains the properties from
/// 'PropertyListT' and the newly added alignment property.
template <typename PropertyListT, size_t Alignment>
struct add_alignment_property_helper<PropertyListT, Alignment, false> {
  using ExpPropertyListT =
      typename get_ext_oneapi_properties<PropertyListT>::type;
  using AlignmentPropList =
      typename ext::oneapi::experimental::detail::properties_t<
          alignment_key::value_t<Alignment>>;

  using type =
      ext::oneapi::experimental::detail::merged_properties_t<ExpPropertyListT,
                                                             AlignmentPropList>;
};

// Creates and adds a compile-time property 'alignment<Alignment>' if
// the given property list 'PropertyListT' does not yet have the 'alignment'
// property in it.
template <typename PropertyListT, size_t Alignment>
class add_alignment_property {
  using ExpPropertyListT =
      typename get_ext_oneapi_properties<PropertyListT>::type;

public:
  using type = typename add_alignment_property_helper<
      ExpPropertyListT, Alignment,
      ExpPropertyListT::template has_property<alignment_key>()>::type;
};
template <typename PropertyListT, size_t Alignment>
using add_alignment_property_t =
    typename add_alignment_property<PropertyListT, Alignment>::type;

// Creates the type for the list of L1, L2, and alignment properties.
template <cache_hint L1H, cache_hint L2H, size_t Alignment>
struct make_L1_L2_alignment_properties {
  using type = ext::oneapi::experimental::detail::properties_t<
      alignment_key::value_t<Alignment>, cache_hint_L1_key::value_t<L1H>,
      cache_hint_L2_key::value_t<L2H>>;
};
template <cache_hint L1H, cache_hint L2H, size_t Alignment>
using make_L1_L2_alignment_properties_t =
    typename make_L1_L2_alignment_properties<L1H, L2H, Alignment>::type;

// Creates the type for the list of L1 and L2 properties.
template <cache_hint L1H, cache_hint L2H> struct make_L1_L2_properties {
  using type = ext::oneapi::experimental::detail::properties_t<
      cache_hint_L1_key::value_t<L1H>, cache_hint_L2_key::value_t<L2H>>;
};
template <cache_hint L1H, cache_hint L2H>
using make_L1_L2_properties_t = typename make_L1_L2_properties<L1H, L2H>::type;

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

// Declare that esimd::properties is a property_list.
template <typename... PropertyValueTs>
struct is_property_list<__ESIMD_NS::properties<std::tuple<PropertyValueTs...>>>
    : is_property_list<properties<std::tuple<PropertyValueTs...>>> {};

namespace detail {
// We do not override the class ConflictingProperties for cache_hint properties
// because that mechanism would only allow to verify few obvious restrictions
// without the knowledge of the context in which the cache_hint properties are
// used (load, store, prefetch, atomic). Thus the function
// __ESIMD_DNS::check_cache_hint() is used to verify correctness of properties.
} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
