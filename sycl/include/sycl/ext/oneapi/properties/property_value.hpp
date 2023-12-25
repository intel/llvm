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

// Get property key from value. Key is always first template argument.
template <class T>
using key_from_value = sycl::detail::boost::mp11::mp_front<T>;

template <class T>
using key_from_value_ignore_const = key_from_value<std::remove_const_t<T>>;

// Return void if not a valid value
template <class T>
using key_from_value_or_void =
    sycl::detail::boost::mp11::mp_eval_or<void, key_from_value_ignore_const, T>;

} // namespace detail

template <typename PropertyT, typename... Ts>
struct property_value : public detail::PropertyValueBase<Ts...> {};

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
template <typename V>
using is_property_value = is_property_key<detail::key_from_value_or_void<V>>;

template <typename V, typename O>
using is_property_value_of =
    is_property_key_of<detail::key_from_value_or_void<V>, O>;

namespace detail {

// Specialization of PropertyID for propagating IDs through property_value.
template <typename PropertyT, typename... PropertyValueTs>
struct PropertyID<property_value<PropertyT, PropertyValueTs...>>
    : PropertyID<PropertyT> {};

// Specialization of IsCompileTimePropertyValue for property values.
template <typename PropertyT, typename... PropertyValueTs>
struct IsCompileTimePropertyValue<property_value<PropertyT, PropertyValueTs...>>
    : IsCompileTimeProperty<PropertyT> {};

template <typename PropertyT, typename... PropertyValueTs>
struct IsRuntimePropertyValue<property_value<PropertyT, PropertyValueTs...>>
    : IsRuntimeProperty<PropertyT> {};

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
