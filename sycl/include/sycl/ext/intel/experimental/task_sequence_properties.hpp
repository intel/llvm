/* INTEL_CUSTOMIZATION */
//==-- task_sequence_properties.hpp - Specific properties of task_sequence -==//
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
// Forward declare a class that these properties can be applied to
template <auto &f, typename PropertyListT> class task_sequence;

// Make sure that we are using the right namespace
template <typename PropertyT, typename... Ts>
using property_value =
    sycl::ext::oneapi::experimental::property_value<PropertyT, Ts...>;

struct balanced_key {
  using value_t = property_value<balanced_key>;
};

struct invocation_capacity_key {
  template <unsigned int Size>
  using value_t = property_value<invocation_capacity_key,
                                 std::integral_constant<unsigned int, Size>>;
};

struct response_capacity_key {
  template <unsigned int Size>
  using value_t = property_value<response_capacity_key,
                                 std::integral_constant<unsigned int, Size>>;
};

inline constexpr balanced_key::value_t balanced;
template <unsigned int Size>
inline constexpr invocation_capacity_key::value_t<Size> invocation_capacity;
template <unsigned int Size>
inline constexpr response_capacity_key::value_t<Size> response_capacity;

} // namespace ext::intel::experimental

namespace ext::oneapi::experimental {
template <>
struct is_property_key<intel::experimental::balanced_key> : std::true_type {};
template <>
struct is_property_key<intel::experimental::invocation_capacity_key>
    : std::true_type {};
template <>
struct is_property_key<intel::experimental::response_capacity_key>
    : std::true_type {};

template <auto &f, typename PropertyListT>
struct is_property_key_of<intel::experimental::balanced_key,
                          intel::experimental::task_sequence<f, PropertyListT>>
    : std::true_type {};
template <auto &f, typename PropertyListT>
struct is_property_key_of<intel::experimental::invocation_capacity_key,
                          intel::experimental::task_sequence<f, PropertyListT>>
    : std::true_type {};
template <auto &f, typename PropertyListT>
struct is_property_key_of<intel::experimental::response_capacity_key,
                          intel::experimental::task_sequence<f, PropertyListT>>
    : std::true_type {};

namespace detail {
template <> struct PropertyToKind<intel::experimental::balanced_key> {
  static constexpr PropKind Kind = Balanced;
};
template <>
struct PropertyToKind<intel::experimental::invocation_capacity_key> {
  static constexpr PropKind Kind = InvocationCapacity;
};
template <> struct PropertyToKind<intel::experimental::response_capacity_key> {
  static constexpr PropKind Kind = ResponseCapacity;
};

template <>
struct IsCompileTimeProperty<intel::experimental::balanced_key>
    : std::true_type {};
template <>
struct IsCompileTimeProperty<intel::experimental::invocation_capacity_key>
    : std::true_type {};
template <>
struct IsCompileTimeProperty<intel::experimental::response_capacity_key>
    : std::true_type {};

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
