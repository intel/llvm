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
#include <type_traits> // for true_type

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

template <typename T, typename PropertyListT> class fpga_mem;

// Property definitions
enum class resource_enum : std::uint16_t { mlab, block_ram };

struct resource_key {
  template <resource_enum Resource>
  using value_t =
      property_value<resource_key,
                      std::integral_constant<resource_enum, Resource>>;
};

struct num_banks_key {
  template <size_t elements>
  using value_t =
      property_value<num_banks_key, std::integral_constant<size_t, elements>>;
};

struct stride_size_key {
  template <size_t elements>
  using value_t =
      property_value<stride_size_key, std::integral_constant<size_t, elements>>;
};

struct word_size_key {
  template <size_t elements>
  using value_t =
      property_value<word_size_key, std::integral_constant<size_t, elements>>;
};

struct bi_directional_ports_key {
  template <bool Enable>
  using value_t = property_value<
      bi_directional_ports_key, std::bool_constant<Enable>>;
};

struct clock_2x_key {
  template <bool Enable>
  using value_t = property_value<clock_2x_key, std::bool_constant<Enable>>;
};

enum class ram_stitching_enum : std::uint16_t { min_ram, max_fmax };

struct ram_stitching_key {
  template <ram_stitching_enum Ram_stritching>
  using value_t = property_value<
      ram_stitching_key,
      std::integral_constant<ram_stitching_enum, Ram_stritching>>;
};

struct max_private_copies_key {
  template <size_t n>
  using value_t =
      property_value<max_private_copies_key, std::integral_constant<size_t, n>>;
};

struct num_replicates_key {
  template <size_t n>
  using value_t =
      property_value<num_replicates_key, std::integral_constant<size_t, n>>;
};

// Convenience aliases
template <resource_enum r> inline constexpr resource_key::value_t<r> resource;
inline constexpr resource_key::value_t<resource_enum::mlab> resource_mlab;
inline constexpr resource_key::value_t<resource_enum::block_ram>
    resource_block_ram;

template <size_t e> inline constexpr num_banks_key::value_t<e> num_banks;

template <size_t e> inline constexpr stride_size_key::value_t<e> stride_size;

template <size_t e> inline constexpr word_size_key::value_t<e> word_size;

template <bool b>
inline constexpr bi_directional_ports_key::value_t<b> bi_directional_ports;
inline constexpr bi_directional_ports_key::value_t<false>
    bi_directional_ports_false;
inline constexpr bi_directional_ports_key::value_t<true>
    bi_directional_ports_true;

template <bool b>
inline constexpr clock_2x_key::value_t<b> clock_2x;
inline constexpr clock_2x_key::value_t<true> clock_2x_true;
inline constexpr clock_2x_key::value_t<false> clock_2x_false;

template <ram_stitching_enum d>
inline constexpr ram_stitching_key::value_t<d> ram_stitching;
inline constexpr ram_stitching_key::value_t<ram_stitching_enum::min_ram>
    ram_stitching_min_ram;
inline constexpr ram_stitching_key::value_t<ram_stitching_enum::max_fmax>
    ram_stitching_max_fmax;

template <size_t n>
inline constexpr max_private_copies_key::value_t<n> max_private_copies;

template <size_t n>
inline constexpr num_replicates_key::value_t<n> num_replicates;

// Associate properties with fpga_mem
template <typename T, typename PropertyListT>
struct is_property_key_of<resource_key,
                          fpga_mem<T, PropertyListT>> : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<num_banks_key,
                          fpga_mem<T, PropertyListT>> : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<stride_size_key,
                          fpga_mem<T, PropertyListT>> : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<word_size_key,
                          fpga_mem<T, PropertyListT>> : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<bi_directional_ports_key,
                          fpga_mem<T, PropertyListT>> : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<clock_2x_key,
                          fpga_mem<T, PropertyListT>> : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<ram_stitching_key,
                          fpga_mem<T, PropertyListT>> : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<max_private_copies_key,
                          fpga_mem<T, PropertyListT>> : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<num_replicates_key,
                          fpga_mem<T, PropertyListT>> : std::true_type {};

//Artem FIX below

// namespace detail {
// template <> struct PropertyToKind<device_image_scope_key> {
//   static constexpr PropKind Kind = PropKind::DeviceImageScope;
// };
// template <> struct PropertyToKind<host_access_key> {
//   static constexpr PropKind Kind = PropKind::HostAccess;
// };
// template <> struct PropertyToKind<init_mode_key> {
//   static constexpr PropKind Kind = PropKind::InitMode;
// };
// template <> struct PropertyToKind<implement_in_csr_key> {
//   static constexpr PropKind Kind = PropKind::ImplementInCSR;
// };

// template <>
// struct IsCompileTimeProperty<device_image_scope_key> : std::true_type {};
// template <> struct IsCompileTimeProperty<host_access_key> : std::true_type {};
// template <> struct IsCompileTimeProperty<init_mode_key> : std::true_type {};
// template <>
// struct IsCompileTimeProperty<implement_in_csr_key> : std::true_type {};

// template <> struct PropertyMetaInfo<device_image_scope_key::value_t> {
//   static constexpr const char *name = "sycl-device-image-scope";
//   static constexpr std::nullptr_t value = nullptr;
// };
// template <host_access_enum Access>
// struct PropertyMetaInfo<host_access_key::value_t<Access>> {
//   static constexpr const char *name = "sycl-host-access";
//   static constexpr host_access_enum value = Access;
// };
// template <init_mode_enum Trigger>
// struct PropertyMetaInfo<init_mode_key::value_t<Trigger>> {
//   static constexpr const char *name = "sycl-init-mode";
//   static constexpr init_mode_enum value = Trigger;
// };
// template <bool Enable>
// struct PropertyMetaInfo<implement_in_csr_key::value_t<Enable>> {
//   static constexpr const char *name = "sycl-implement-in-csr";
//   static constexpr bool value = Enable;
// };
// } // namespace detail

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
