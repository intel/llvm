//==------ property_value.hpp --- SYCL compile-time property values --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_utils.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {
namespace experimental {
namespace detail {

// Base class for property values with a single type value.
struct SingleTypePropertyValueBase {};

// Base class for properties with 0 or more than 1 values.
struct EmptyPropertyValueBase {};

// Base class for property values with a single non-type value
template <typename T> struct SingleNontypePropertyValueBase {
  static constexpr auto value = T::value;
};

// Helper class for property values with a single value
template <typename T>
struct SinglePropertyValue
    : public sycl::detail::conditional_t<HasValue<T>::value,
                                         SingleNontypePropertyValueBase<T>,
                                         SingleTypePropertyValueBase> {
  using value_t = T;
};

} // namespace detail

template <typename PropertyT, typename T = void, typename... Ts>
struct property_value
    : public sycl::detail::conditional_t<
          sizeof...(Ts) == 0 && !std::is_same<T, void>::value,
          detail::SinglePropertyValue<T>, detail::EmptyPropertyValueBase> {
  using key_t = PropertyT;
};

template <typename PropertyT, typename... A, typename... B>
constexpr std::enable_if_t<detail::IsCompileTimeProperty<PropertyT>::value,
                           bool>
operator==(const property_value<PropertyT, A...> &LHS,
           const property_value<PropertyT, B...> &RHS) {
  return (std::is_same<A, B>::value && ...);
}

template <typename PropertyT, typename... A, typename... B>
constexpr std::enable_if_t<detail::IsCompileTimeProperty<PropertyT>::value,
                           bool>
operator!=(const property_value<PropertyT, A...> &LHS,
           const property_value<PropertyT, B...> &RHS) {
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
struct is_property_value<V, sycl::detail::void_t<typename V::key_t>>
    : is_property_key<typename V::key_t> {};
template <typename V, typename O>
struct is_property_value_of<V, O, sycl::detail::void_t<typename V::key_t>>
    : is_property_key_of<typename V::key_t, O> {};

namespace detail {

// Specialization of PropertyID for propagating IDs through property_value.
template <typename PropertyT, typename... PropertyValueTs>
struct PropertyID<property_value<PropertyT, PropertyValueTs...>>
    : PropertyID<PropertyT> {};

// Specialization of IsCompileTimePropertyValue for property values.
template <typename PropertyT, typename... PropertyValueTs>
struct IsCompileTimePropertyValue<property_value<PropertyT, PropertyValueTs...>>
    : IsCompileTimeProperty<PropertyT> {};

} // namespace detail
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
