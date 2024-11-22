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

// Checks if a type T has a static value member variable.
template <typename T, typename U = int> struct HasValue : std::false_type {};
template <typename T>
struct HasValue<T, decltype((void)T::value, 0)> : std::true_type {};

// Base class for property values with a single non-type value
template <typename T, typename = void> struct SingleNontypePropertyValueBase {};

template <typename T>
struct SingleNontypePropertyValueBase<T, std::enable_if_t<HasValue<T>::value>> {
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
constexpr std::enable_if_t<detail::IsCompileTimeProperty<PropertyT>::value,
                           bool>
operator==(const property_value<PropertyT, A...> &,
           const property_value<PropertyT, B...> &) {
  return (std::is_same<A, B>::value && ...);
}

template <typename PropertyT, typename... A, typename... B>
constexpr std::enable_if_t<detail::IsCompileTimeProperty<PropertyT>::value,
                           bool>
operator!=(const property_value<PropertyT, A...> &,
           const property_value<PropertyT, B...> &) {
  return (!std::is_same<A, B>::value || ...);
}

template <typename V, typename = void> struct is_property_value {
  static constexpr bool value =
      detail::IsRuntimeProperty<V>::value && is_property_key<V>::value;
};
template <typename V, typename O, typename = void> struct is_property_value_of {
  static constexpr bool value =
      detail::IsRuntimeProperty<V>::value && is_property_key_of<V, O>::value;
};
// Specialization for compile-time-constant properties
template <typename V>
struct is_property_value<V, std::void_t<typename V::key_t>>
    : is_property_key<typename V::key_t> {};
template <typename V, typename O>
struct is_property_value_of<V, O, std::void_t<typename V::key_t>>
    : is_property_key_of<typename V::key_t, O> {};

namespace detail {

// Specialization of PropertyID for propagating IDs through property_value.
template <typename PropertyT, typename... PropertyValueTs>
struct PropertyID<property_value<PropertyT, PropertyValueTs...>>
    : PropertyID<PropertyT> {};

// Checks if a type is a compile-time property values.
template <typename PropertyT>
struct IsCompileTimePropertyValue : std::false_type {};
template <typename PropertyT, typename... PropertyValueTs>
struct IsCompileTimePropertyValue<property_value<PropertyT, PropertyValueTs...>>
    : IsCompileTimeProperty<PropertyT> {};

// Checks if a type is a valid property value, i.e either runtime property or
// property_value with a valid compile-time property
template <typename T> struct IsPropertyValue {
  static constexpr bool value =
      IsRuntimeProperty<T>::value || IsCompileTimePropertyValue<T>::value;
};
} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
