//==------ property_value.hpp --- SYCL compile-time property values --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// HOW-TO: Add new compile-time property
//  1. Define property class with `value_t` that must be `property_value` with
//     the first template argument being the property class itself.
//  2. Add a new enumerator to `sycl::ext::oneapi::detail::CompileTimePropKind`
//     representing the new property. Increment
//     `sycl::ext::oneapi::detail::CompileTimePropKind::CompileTimePropKindSize`
//  3. Specialize `sycl::ext::oneapi::detail::CompileTimePropertyToKind` for the
//     new property class. The specialization should have a `value` member
//     with the value equal to the enumerator added in 2.
//  4. Add an `inline constexpr` variable in the same namespace as the property.
//     The variable should have the same type as `value_t` of the property class
//     and should be named as the property class with `_v` appended, e.g. for a
//     property `foo`, there should be a definition
//     `inline constexpr foo::value_t foo_v`.
//  5. Specialize `sycl::is_property` and `sycl::is_property_of` for the
//     property class.

#pragma once

#include <CL/sycl/detail/property_helper.hpp>
#include <sycl/ext/oneapi/property_list/property_utils.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {
namespace detail {

// List of all compile-time properties
enum CompileTimePropKind {
  CompileTimePropKindSize = 0,
};

// This trait must be specialized for all compile-time properties and must have
// a unique constexpr CompileTimePropKind member named PropKind
template <typename PropertyT> struct CompileTimePropertyToKind {};

// Get unique ID for compile-time property
template <typename PropertyT> struct CompileTimePropertyID {
  static constexpr int value =
      static_cast<int>(CompileTimePropertyToKind<PropertyT>::PropKind) +
      static_cast<int>(PropWithDataKind::PropWithDataKindSize) +
      static_cast<int>(DataLessPropKind::DataLessPropKindSize);
};

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
    : public detail::IdentifyablePropertyBase<
          detail::CompileTimePropertyID<PropertyT>::value>,
      public detail::conditional_t<
          sizeof...(Ts) == 0 && !std::is_same<T, void>::value,
          detail::SinglePropertyValue<T>, detail::EmptyPropertyValueBase> {};

template <class PropertyT, class... A, class... B>
constexpr detail::enable_if_t<is_property<PropertyT>::value &&
                                  !detail::IsRuntimeProperty<PropertyT>::value,
                              bool>
operator==(const property_value<PropertyT, A...> &LHS,
           const property_value<PropertyT, B...> &RHS) {
  return (std::is_same<A, B>::value && ...);
}

template <class PropertyT, class... A, class... B>
constexpr detail::enable_if_t<is_property<PropertyT>::value &&
                                  !detail::IsRuntimeProperty<PropertyT>::value,
                              bool>
operator!=(const property_value<PropertyT, A...> &LHS,
           const property_value<PropertyT, B...> &RHS) {
  return (!std::is_same<A, B>::value || ...);
}

} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
