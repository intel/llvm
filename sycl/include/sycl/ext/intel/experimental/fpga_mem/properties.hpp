//==----- properties.hpp - SYCL properties associated with fpga_mem ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/properties/property.hpp>       // for PropKind
#include <sycl/ext/oneapi/properties/property_value.hpp> // for property_value

#include <cstdint>     // for uint16_t
#include <iosfwd>      // for nullptr_t
#include <string_view> // for string_view
#include <type_traits> // for true_type

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace intel::experimental {

// Forward declare a class that these properties can be applied to
template <typename T, typename PropertyListT> class fpga_mem;

// Make sure that we are using the right namespace
template <typename PropertyT, typename... Ts>
using property_value =
    sycl::ext::oneapi::experimental::property_value<PropertyT, Ts...>;

// Property definitions
enum class resource_enum : std::uint16_t { mlab, block_ram };

struct resource_key {
  template <resource_enum Resource>
  using value_t =
      property_value<resource_key,
                     std::integral_constant<resource_enum, Resource>>;
};

struct num_banks_key {
  template <size_t Elements>
  using value_t =
      property_value<num_banks_key, std::integral_constant<size_t, Elements>>;
};

struct stride_size_key {
  template <size_t Elements>
  using value_t =
      property_value<stride_size_key, std::integral_constant<size_t, Elements>>;
};

struct word_size_key {
  template <size_t Elements>
  using value_t =
      property_value<word_size_key, std::integral_constant<size_t, Elements>>;
};

struct bi_directional_ports_key {
  template <bool Enable>
  using value_t =
      property_value<bi_directional_ports_key, std::bool_constant<Enable>>;
};

struct clock_2x_key {
  template <bool Enable>
  using value_t = property_value<clock_2x_key, std::bool_constant<Enable>>;
};

enum class ram_stitching_enum : std::uint16_t { min_ram, max_fmax };

struct ram_stitching_key {
  template <ram_stitching_enum RamStitching>
  using value_t =
      property_value<ram_stitching_key,
                     std::integral_constant<ram_stitching_enum, RamStitching>>;
};

struct max_private_copies_key {
  template <size_t N>
  using value_t =
      property_value<max_private_copies_key, std::integral_constant<size_t, N>>;
};

struct num_replicates_key {
  template <size_t N>
  using value_t =
      property_value<num_replicates_key, std::integral_constant<size_t, N>>;
};

// Convenience aliases
template <resource_enum R> inline constexpr resource_key::value_t<R> resource;
inline constexpr resource_key::value_t<resource_enum::mlab> resource_mlab;
inline constexpr resource_key::value_t<resource_enum::block_ram>
    resource_block_ram;

template <size_t E> inline constexpr num_banks_key::value_t<E> num_banks;

template <size_t E> inline constexpr stride_size_key::value_t<E> stride_size;

template <size_t E> inline constexpr word_size_key::value_t<E> word_size;

template <bool B>
inline constexpr bi_directional_ports_key::value_t<B> bi_directional_ports;
inline constexpr bi_directional_ports_key::value_t<false>
    bi_directional_ports_false;
inline constexpr bi_directional_ports_key::value_t<true>
    bi_directional_ports_true;

template <bool B> inline constexpr clock_2x_key::value_t<B> clock_2x;
inline constexpr clock_2x_key::value_t<true> clock_2x_true;
inline constexpr clock_2x_key::value_t<false> clock_2x_false;

template <ram_stitching_enum D>
inline constexpr ram_stitching_key::value_t<D> ram_stitching;
inline constexpr ram_stitching_key::value_t<ram_stitching_enum::min_ram>
    ram_stitching_min_ram;
inline constexpr ram_stitching_key::value_t<ram_stitching_enum::max_fmax>
    ram_stitching_max_fmax;

template <size_t N>
inline constexpr max_private_copies_key::value_t<N> max_private_copies;

template <size_t N>
inline constexpr num_replicates_key::value_t<N> num_replicates;

} // namespace intel::experimental

namespace oneapi::experimental {

template <>
struct is_property_key<intel::experimental::resource_key> : std::true_type {};
template <>
struct is_property_key<intel::experimental::num_banks_key> : std::true_type {};
template <>
struct is_property_key<intel::experimental::stride_size_key> : std::true_type {
};
template <>
struct is_property_key<intel::experimental::word_size_key> : std::true_type {};
template <>
struct is_property_key<intel::experimental::bi_directional_ports_key>
    : std::true_type {};
template <>
struct is_property_key<intel::experimental::clock_2x_key> : std::true_type {};
template <>
struct is_property_key<intel::experimental::ram_stitching_key>
    : std::true_type {};
template <>
struct is_property_key<intel::experimental::max_private_copies_key>
    : std::true_type {};
template <>
struct is_property_key<intel::experimental::num_replicates_key>
    : std::true_type {};

// Associate properties with fpga_mem
template <typename T, typename PropertyListT>
struct is_property_key_of<intel::experimental::resource_key,
                          intel::experimental::fpga_mem<T, PropertyListT>>
    : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<intel::experimental::num_banks_key,
                          intel::experimental::fpga_mem<T, PropertyListT>>
    : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<intel::experimental::stride_size_key,
                          intel::experimental::fpga_mem<T, PropertyListT>>
    : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<intel::experimental::word_size_key,
                          intel::experimental::fpga_mem<T, PropertyListT>>
    : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<intel::experimental::bi_directional_ports_key,
                          intel::experimental::fpga_mem<T, PropertyListT>>
    : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<intel::experimental::clock_2x_key,
                          intel::experimental::fpga_mem<T, PropertyListT>>
    : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<intel::experimental::ram_stitching_key,
                          intel::experimental::fpga_mem<T, PropertyListT>>
    : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<intel::experimental::max_private_copies_key,
                          intel::experimental::fpga_mem<T, PropertyListT>>
    : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<intel::experimental::num_replicates_key,
                          intel::experimental::fpga_mem<T, PropertyListT>>
    : std::true_type {};

namespace detail {
// Map Property to a PropKind enum
template <> struct PropertyToKind<intel::experimental::resource_key> {
  static constexpr PropKind Kind = PropKind::Resource;
};
template <> struct PropertyToKind<intel::experimental::num_banks_key> {
  static constexpr PropKind Kind = PropKind::NumBanks;
};
template <> struct PropertyToKind<intel::experimental::stride_size_key> {
  static constexpr PropKind Kind = PropKind::StrideSize;
};
template <> struct PropertyToKind<intel::experimental::word_size_key> {
  static constexpr PropKind Kind = PropKind::WordSize;
};
template <>
struct PropertyToKind<intel::experimental::bi_directional_ports_key> {
  static constexpr PropKind Kind = PropKind::BiDirectionalPorts;
};
template <> struct PropertyToKind<intel::experimental::clock_2x_key> {
  static constexpr PropKind Kind = PropKind::Clock2x;
};
template <> struct PropertyToKind<intel::experimental::ram_stitching_key> {
  static constexpr PropKind Kind = PropKind::RAMStitching;
};
template <> struct PropertyToKind<intel::experimental::max_private_copies_key> {
  static constexpr PropKind Kind = PropKind::MaxPrivateCopies;
};
template <> struct PropertyToKind<intel::experimental::num_replicates_key> {
  static constexpr PropKind Kind = PropKind::NumReplicates;
};

// Mark the properties as compile-time
template <>
struct IsCompileTimeProperty<intel::experimental::resource_key>
    : std::true_type {};
template <>
struct IsCompileTimeProperty<intel::experimental::num_banks_key>
    : std::true_type {};
template <>
struct IsCompileTimeProperty<intel::experimental::stride_size_key>
    : std::true_type {};
template <>
struct IsCompileTimeProperty<intel::experimental::word_size_key>
    : std::true_type {};
template <>
struct IsCompileTimeProperty<intel::experimental::bi_directional_ports_key>
    : std::true_type {};
template <>
struct IsCompileTimeProperty<intel::experimental::clock_2x_key>
    : std::true_type {};
template <>
struct IsCompileTimeProperty<intel::experimental::ram_stitching_key>
    : std::true_type {};
template <>
struct IsCompileTimeProperty<intel::experimental::max_private_copies_key>
    : std::true_type {};
template <>
struct IsCompileTimeProperty<intel::experimental::num_replicates_key>
    : std::true_type {};

// Map Property to MetaInfo
template <intel::experimental::resource_enum Value>
struct PropertyMetaInfo<intel::experimental::resource_key::value_t<Value>> {
  static constexpr const char *name = "sycl-resource";
  static constexpr const char *value =
      ((Value == intel::experimental::resource_enum::mlab) ? "MLAB"
                                                           : "BLOCK_RAM");
};
template <size_t Value>
struct PropertyMetaInfo<intel::experimental::num_banks_key::value_t<Value>> {
  static constexpr const char *name = "sycl-num-banks";
  static constexpr size_t value = Value;
};
template <size_t Value>
struct PropertyMetaInfo<intel::experimental::stride_size_key::value_t<Value>> {
  static constexpr const char *name = "sycl-stride-size";
  static constexpr size_t value = Value;
};
template <size_t Value>
struct PropertyMetaInfo<intel::experimental::word_size_key::value_t<Value>> {
  static constexpr const char *name = "sycl-word-size";
  static constexpr size_t value = Value;
};
template <bool Value>
struct PropertyMetaInfo<
    intel::experimental::bi_directional_ports_key::value_t<Value>> {
  // historical uglyness: single property maps to different SPIRV decorations
  static constexpr const char *name =
      (Value ? "sycl-bi-directional-ports-true"
             : "sycl-bi-directional-ports-false");
  static constexpr std::nullptr_t value = nullptr;
};
template <bool Value>
struct PropertyMetaInfo<intel::experimental::clock_2x_key::value_t<Value>> {
  // historical uglyness: single property maps to different SPIRV decorations
  static constexpr const char *name =
      (Value ? "sycl-clock-2x-true" : "sycl-clock-2x-false");
  static constexpr std::nullptr_t value = nullptr;
};
template <intel::experimental::ram_stitching_enum Value>
struct PropertyMetaInfo<
    intel::experimental::ram_stitching_key::value_t<Value>> {
  static constexpr const char *name = "sycl-ram-stitching";
  // enum to bool conversion to match with the SPIR-V decoration
  // ForcePow2DepthINTEL
  static constexpr size_t value = static_cast<size_t>(
      Value == intel::experimental::ram_stitching_enum::max_fmax);
};
template <size_t Value>
struct PropertyMetaInfo<
    intel::experimental::max_private_copies_key::value_t<Value>> {
  static constexpr const char *name = "sycl-max-private-copies";
  static constexpr size_t value = Value;
};
template <size_t Value>
struct PropertyMetaInfo<
    intel::experimental::num_replicates_key::value_t<Value>> {
  static constexpr const char *name = "sycl-num-replicates";
  static constexpr size_t value = Value;
};

} // namespace detail
} // namespace oneapi::experimental
} // namespace ext
} // namespace _V1
} // namespace sycl
