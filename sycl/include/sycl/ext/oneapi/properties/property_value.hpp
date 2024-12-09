//==------ property_value.hpp --- SYCL compile-time property values --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/properties/property.hpp>

#include <type_traits> // for enable_if_t

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
namespace detail {

// Base class for property values with a single non-type value
template <typename T, typename = void> struct SingleNontypePropertyValueBase {};

template <typename T>
struct SingleNontypePropertyValueBase<T, std::void_t<decltype(T::value)>> {
  static constexpr auto value = T::value;
};

// Helper base class for property_value.
template <typename... Ts> struct PropertyValueBase {};

template <typename T>
struct PropertyValueBase<T> : public detail::SingleNontypePropertyValueBase<T> {
  using value_t = T;
};

} // namespace detail

template <typename PropertyT, typename... Ts>
struct property_value
    : public detail::PropertyValueBase<Ts...>,
      public detail::property_base<property_value<PropertyT, Ts...>,
                                   detail::PropertyToKind<PropertyT>::Kind,
                                   PropertyT> {};

template <typename PropertyT, typename... A, typename... B>
constexpr std::enable_if_t<std::is_empty_v<property_value<PropertyT, A...>>,
                           bool>
operator==(const property_value<PropertyT, A...> &,
           const property_value<PropertyT, B...> &) {
  return (std::is_same<A, B>::value && ...);
}

template <typename PropertyT, typename... A, typename... B>
constexpr std::enable_if_t<std::is_empty_v<property_value<PropertyT, A...>>,
                           bool>
operator!=(const property_value<PropertyT, A...> &,
           const property_value<PropertyT, B...> &) {
  return (!std::is_same<A, B>::value || ...);
}

template <typename V>
struct is_property_value
    : std::bool_constant<!is_property_list_v<V> &&
                         std::is_base_of_v<detail::property_tag, V>> {};

template <typename V>
inline constexpr bool is_property_value_v = is_property_value<V>::value;

template <typename V, typename O> struct is_property_value_of {
  static constexpr bool value = []() constexpr {
    if constexpr (is_property_value_v<V>)
      return is_property_key_of<typename V::key_t, O>::value;
    else
      return false;
  }();
};

namespace detail {

// Specialization of PropertyID for propagating IDs through property_value.
template <typename PropertyT, typename... PropertyValueTs>
struct PropertyID<property_value<PropertyT, PropertyValueTs...>>
    : PropertyID<PropertyT> {};

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
