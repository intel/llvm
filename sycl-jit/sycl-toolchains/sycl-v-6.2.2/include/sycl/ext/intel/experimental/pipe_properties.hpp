//==----- pipe_properties.hpp - SYCL properties associated with data flow pipe
//---==//
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
#include <type_traits> // for true_type

namespace sycl {
inline namespace _V1 {
namespace ext::intel::experimental {

struct ready_latency_key
    : oneapi::experimental::detail::compile_time_property_key<
          oneapi::experimental::detail::PropKind::ReadyLatency> {
  template <int Latency>
  using value_t = oneapi::experimental::property_value<
      ready_latency_key, std::integral_constant<int, Latency>>;
};

struct bits_per_symbol_key
    : oneapi::experimental::detail::compile_time_property_key<
          oneapi::experimental::detail::PropKind::BitsPerSymbol> {
  template <int Bits>
  using value_t =
      oneapi::experimental::property_value<bits_per_symbol_key,
                                           std::integral_constant<int, Bits>>;
};

struct uses_valid_key : oneapi::experimental::detail::compile_time_property_key<
                            oneapi::experimental::detail::PropKind::UsesValid> {
  template <bool Valid>
  using value_t =
      oneapi::experimental::property_value<uses_valid_key,
                                           std::bool_constant<Valid>>;
};

struct first_symbol_in_high_order_bits_key
    : oneapi::experimental::detail::compile_time_property_key<
          oneapi::experimental::detail::PropKind::FirstSymbolInHigherOrderBit> {
  template <bool HighOrder>
  using value_t =
      oneapi::experimental::property_value<first_symbol_in_high_order_bits_key,
                                           std::bool_constant<HighOrder>>;
};

enum class protocol_name : std::uint16_t {
  avalon_streaming = 0,
  avalon_streaming_uses_ready = 1,
  avalon_mm = 2,
  avalon_mm_uses_ready = 3
};

struct protocol_key
    : oneapi::experimental::detail::compile_time_property_key<
          oneapi::experimental::detail::PropKind::PipeProtocol> {
  template <protocol_name Protocol>
  using value_t = oneapi::experimental::property_value<
      protocol_key, std::integral_constant<protocol_name, Protocol>>;
};

template <int Latency>
inline constexpr ready_latency_key::value_t<Latency> ready_latency;

template <int Bits>
inline constexpr bits_per_symbol_key::value_t<Bits> bits_per_symbol;

template <bool Valid>
inline constexpr uses_valid_key::value_t<Valid> uses_valid;
inline constexpr uses_valid_key::value_t<true> uses_valid_on;
inline constexpr uses_valid_key::value_t<false> uses_valid_off;

template <bool HighOrder>
inline constexpr first_symbol_in_high_order_bits_key::value_t<HighOrder>
    first_symbol_in_high_order_bits;
inline constexpr first_symbol_in_high_order_bits_key::value_t<true>
    first_symbol_in_high_order_bits_on;
inline constexpr first_symbol_in_high_order_bits_key::value_t<false>
    first_symbol_in_high_order_bits_off;

template <protocol_name Protocol>
inline constexpr protocol_key::value_t<Protocol> protocol;
inline constexpr protocol_key::value_t<protocol_name::avalon_streaming>
    protocol_avalon_streaming;
inline constexpr protocol_key::value_t<
    protocol_name::avalon_streaming_uses_ready>
    protocol_avalon_streaming_uses_ready;
inline constexpr protocol_key::value_t<protocol_name::avalon_mm>
    protocol_avalon_mm;
inline constexpr protocol_key::value_t<protocol_name::avalon_mm_uses_ready>
    protocol_avalon_mm_uses_ready;

} // namespace ext::intel::experimental
} // namespace _V1
} // namespace sycl
