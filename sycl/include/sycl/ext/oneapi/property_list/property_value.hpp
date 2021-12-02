//==------ property_value.hpp --- SYCL compile-time property values --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/property_list/properties.hpp>
#include <sycl/ext/oneapi/property_list/property_utils.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {
namespace detail {

// Base class for property values with a single type value.
struct SingleTypePropertyValueBase {};

// Base class for properties with 0 or more than 1 values.
struct EmptyPropertyValueBase {};

// Base class for property values with a single non-type value
template <class T> struct SingleNontypePropertyValueBase {
  static constexpr auto value = T::value;
};

// Helper class for property values with a single value
template <class T>
struct SinglePropertyValue
    : public detail::conditional_t<HasValue<T>::value,
                                   SingleNontypePropertyValueBase<T>,
                                   SingleTypePropertyValueBase> {
  using value_t = T;
};

} // namespace detail

template <class PropertyT, class T = void, class... Ts>
struct property_value
    : public detail::conditional_t<
          sizeof...(Ts) == 0 && !std::is_same<T, void>::value,
          detail::SinglePropertyValue<T>, detail::EmptyPropertyValueBase> {};

template <class PropertyT, class... A, class... B>
constexpr detail::enable_if_t<detail::IsCompileTimeProperty<PropertyT>::value,
                              bool>
operator==(const property_value<PropertyT, A...> &LHS,
           const property_value<PropertyT, B...> &RHS) {
  return (std::is_same<A, B>::value && ...);
}

template <class PropertyT, class... A, class... B>
constexpr detail::enable_if_t<detail::IsCompileTimeProperty<PropertyT>::value,
                              bool>
operator!=(const property_value<PropertyT, A...> &LHS,
           const property_value<PropertyT, B...> &RHS) {
  return (!std::is_same<A, B>::value || ...);
}

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
} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
