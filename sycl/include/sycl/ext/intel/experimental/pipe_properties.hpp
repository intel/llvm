//==----- pipe_properties.hpp - SYCL properties associated with data flow pipe
//---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace intel {
namespace experimental {

struct min_capacity_key {
  template <int Capacity>
  using value_t = oneapi::experimental::property_value<
      min_capacity_key, std::integral_constant<int, Capacity>>;
};

struct ready_latency_key {
  template <int Latency>
  using value_t = oneapi::experimental::property_value<
      ready_latency_key, std::integral_constant<int, Latency>>;
};

struct bits_per_symbol_key {
  template <int Bits>
  using value_t =
      oneapi::experimental::property_value<bits_per_symbol_key,
                                           std::integral_constant<int, Bits>>;
};

struct uses_valid_key {
  template <bool Valid>
  using value_t =
      oneapi::experimental::property_value<uses_valid_key,
                                           sycl::detail::bool_constant<Valid>>;
};

struct uses_ready_key {
  template <bool Ready>
  using value_t =
      oneapi::experimental::property_value<uses_ready_key,
                                           sycl::detail::bool_constant<Ready>>;
};

struct in_csr_key {
  template <bool Enable>
  using value_t =
      oneapi::experimental::property_value<in_csr_key,
                                           sycl::detail::bool_constant<Enable>>;
};

struct first_symbol_in_high_order_bits_key {
  template <bool HighOrder>
  using value_t = oneapi::experimental::property_value<
      first_symbol_in_high_order_bits_key,
      sycl::detail::bool_constant<HighOrder>>;
};

enum class protocol_name : std::uint16_t { AVALON, AXI };
struct protocol_key {
  template <protocol_name Protocol>
  using value_t = oneapi::experimental::property_value<
      protocol_key, std::integral_constant<protocol_name, Protocol>>;
};

template <int Capacity>
inline constexpr min_capacity_key::value_t<Capacity> min_capacity;

template <int Latency>
inline constexpr ready_latency_key::value_t<Latency> ready_latency;

template <int Bits>
inline constexpr bits_per_symbol_key::value_t<Bits> bits_per_symbol;

template <bool Valid>
inline constexpr uses_valid_key::value_t<Valid> uses_valid;
inline constexpr uses_valid_key::value_t<true> uses_valid_on;
inline constexpr uses_valid_key::value_t<false> uses_valid_off;

template <bool Ready>
inline constexpr uses_ready_key::value_t<Ready> uses_ready;
inline constexpr uses_ready_key::value_t<true> uses_ready_on;
inline constexpr uses_ready_key::value_t<false> uses_ready_off;

template <bool Enable> inline constexpr in_csr_key::value_t<Enable> in_csr;
inline constexpr in_csr_key::value_t<true> in_csr_on;
inline constexpr in_csr_key::value_t<false> in_csr_off;

template <bool HighOrder>
inline constexpr first_symbol_in_high_order_bits_key::value_t<HighOrder>
    first_symbol_in_high_order_bits;
inline constexpr first_symbol_in_high_order_bits_key::value_t<true>
    first_symbol_in_high_order_bits_on;
inline constexpr first_symbol_in_high_order_bits_key::value_t<false>
    first_symbol_in_high_order_bits_off;

template <protocol_name Protocol>
inline constexpr protocol_key::value_t<Protocol> protocol;
inline constexpr protocol_key::value_t<protocol_name::AVALON> protocol_avalon;
inline constexpr protocol_key::value_t<protocol_name::AXI> protocol_axi;

} // namespace experimental
} // namespace intel

namespace oneapi {
namespace experimental {

template <>
struct is_property_key<intel::experimental::min_capacity_key> : std::true_type {
};
template <>
struct is_property_key<intel::experimental::ready_latency_key>
    : std::true_type {};
template <>
struct is_property_key<intel::experimental::bits_per_symbol_key>
    : std::true_type {};
template <>
struct is_property_key<intel::experimental::uses_valid_key> : std::true_type {};
template <>
struct is_property_key<intel::experimental::uses_ready_key> : std::true_type {};
template <>
struct is_property_key<intel::experimental::in_csr_key> : std::true_type {};
template <>
struct is_property_key<intel::experimental::first_symbol_in_high_order_bits_key>
    : std::true_type {};
template <>
struct is_property_key<intel::experimental::protocol_key> : std::true_type {};

namespace detail {
template <> struct PropertyToKind<intel::experimental::min_capacity_key> {
  static constexpr PropKind Kind = PropKind::MinCapacity;
};
template <> struct PropertyToKind<intel::experimental::ready_latency_key> {
  static constexpr PropKind Kind = PropKind::ReadyLatency;
};
template <> struct PropertyToKind<intel::experimental::bits_per_symbol_key> {
  static constexpr PropKind Kind = PropKind::BitsPerSymbol;
};
template <> struct PropertyToKind<intel::experimental::uses_valid_key> {
  static constexpr PropKind Kind = PropKind::UsesValid;
};
template <> struct PropertyToKind<intel::experimental::uses_ready_key> {
  static constexpr PropKind Kind = PropKind::UsesReady;
};
template <> struct PropertyToKind<intel::experimental::in_csr_key> {
  static constexpr PropKind Kind = PropKind::ImplementInCSR;
};
template <>
struct PropertyToKind<
    intel::experimental::first_symbol_in_high_order_bits_key> {
  static constexpr PropKind Kind = PropKind::FirstSymbolInHigherOrderBit;
};
template <> struct PropertyToKind<intel::experimental::protocol_key> {
  static constexpr PropKind Kind = PropKind::PipeProtocol;
};

template <>
struct IsCompileTimeProperty<intel::experimental::min_capacity_key>
    : std::true_type {};
template <>
struct IsCompileTimeProperty<intel::experimental::ready_latency_key>
    : std::true_type {};
template <>
struct IsCompileTimeProperty<intel::experimental::bits_per_symbol_key>
    : std::true_type {};
template <>
struct IsCompileTimeProperty<intel::experimental::uses_valid_key>
    : std::true_type {};
template <>
struct IsCompileTimeProperty<intel::experimental::uses_ready_key>
    : std::true_type {};
template <>
struct IsCompileTimeProperty<intel::experimental::in_csr_key> : std::true_type {
};
template <>
struct IsCompileTimeProperty<
    intel::experimental::first_symbol_in_high_order_bits_key> : std::true_type {
};
template <>
struct IsCompileTimeProperty<intel::experimental::protocol_key>
    : std::true_type {};

} // namespace detail
} // namespace experimental
} // namespace intel
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
