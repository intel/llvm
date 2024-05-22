//==------ property_value.hpp --- SYCL compile-time property values --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/properties/property.hpp>       // for IsCompileTi...
#include <sycl/ext/oneapi/properties/property_utils.hpp> // for HasValue

#include <type_traits> // for enable_if_t

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
namespace detail {

// Helper base class for property_value.
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
struct property_value : public detail::PropertyValueBase<Ts...> {};

template <typename PropertyT, typename... A, typename... B>
constexpr bool operator==(const property_value<PropertyT, A...> &,
                          const property_value<PropertyT, B...> &) {
  return (std::is_same<A, B>::value && ...);
}

template <typename PropertyT, typename... A, typename... B>
constexpr bool operator!=(const property_value<PropertyT, A...> &,
                          const property_value<PropertyT, B...> &) {
  return (!std::is_same<A, B>::value || ...);
}

template <typename V, typename O>
struct is_property_value_of : is_property_key_of<V, O> {};

template <typename V, typename O>
struct is_property_value_of<const V, O> : is_property_value_of<V, O> {};

template <typename K, typename... A, typename O>
struct is_property_value_of<property_value<K, A...>, O>
    : is_property_key_of<K, O> {};

namespace detail {
//******************************************************************************
// Property identification
//******************************************************************************

// Checks if a type is a compile-time property value.
template <typename PropertyT>
struct IsCompileTimePropertyValue : std::false_type {};
template <typename... PropertyTs>
struct IsCompileTimePropertyValue<property_value<PropertyTs...>>
    : std::true_type {};
} // namespace detail

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
