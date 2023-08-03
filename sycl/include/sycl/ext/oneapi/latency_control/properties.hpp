//==--- properties.hpp - SYCL properties associated with latency_control ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/properties/property.hpp>       // for PropKind
#include <sycl/ext/oneapi/properties/property_value.hpp> // for property_value

#include <type_traits> // for true_type

namespace sycl {
inline namespace _V1 {
namespace ext::intel::experimental {

enum class latency_control_type {
  none, // default
  exact,
  max,
  min
};

struct latency_anchor_id_key {
  template <int Anchor>
  using value_t =
      oneapi::experimental::property_value<latency_anchor_id_key,
                                           std::integral_constant<int, Anchor>>;
};

struct latency_constraint_key {
  template <int Target, latency_control_type Type, int Cycle>
  using value_t = oneapi::experimental::property_value<
      latency_constraint_key, std::integral_constant<int, Target>,
      std::integral_constant<latency_control_type, Type>,
      std::integral_constant<int, Cycle>>;
};

template <int Anchor>
inline constexpr latency_anchor_id_key::value_t<Anchor> latency_anchor_id;
template <int Target, latency_control_type Type, int Cycle>
inline constexpr latency_constraint_key::value_t<Target, Type, Cycle>
    latency_constraint;

} // namespace ext::intel::experimental

namespace ext::oneapi::experimental {

template <int Target, intel::experimental::latency_control_type Type, int Cycle>
struct property_value<
    intel::experimental::latency_constraint_key,
    std::integral_constant<int, Target>,
    std::integral_constant<intel::experimental::latency_control_type, Type>,
    std::integral_constant<int, Cycle>> {
  using key_t = intel::experimental::latency_constraint_key;
  static constexpr int target = Target;
  static constexpr intel::experimental::latency_control_type type = Type;
  static constexpr int cycle = Cycle;
};

template <>
struct is_property_key<intel::experimental::latency_anchor_id_key>
    : std::true_type {};
template <>
struct is_property_key<intel::experimental::latency_constraint_key>
    : std::true_type {};

namespace detail {

template <> struct PropertyToKind<intel::experimental::latency_anchor_id_key> {
  static constexpr PropKind Kind = PropKind::LatencyAnchorID;
};
template <> struct PropertyToKind<intel::experimental::latency_constraint_key> {
  static constexpr PropKind Kind = PropKind::LatencyConstraint;
};

template <>
struct IsCompileTimeProperty<intel::experimental::latency_anchor_id_key>
    : std::true_type {};
template <>
struct IsCompileTimeProperty<intel::experimental::latency_constraint_key>
    : std::true_type {};

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
