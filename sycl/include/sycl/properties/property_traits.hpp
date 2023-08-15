//==------------ property_traits.hpp --- SYCL property traits --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/property_helper.hpp> // for DataLessPropertyBase, Pro...

#include <type_traits> // for bool_constant, false_type

namespace sycl {
inline namespace _V1 {

// Property traits
template <typename propertyT>
struct is_property
    : public std::bool_constant<
          std::is_base_of_v<detail::DataLessPropertyBase, propertyT> ||
          std::is_base_of_v<detail::PropertyWithDataBase, propertyT>> {};

template <typename propertyT, typename syclObjectT>
struct is_property_of : public std::false_type {};

template <typename propertyT>
inline constexpr bool is_property_v = is_property<propertyT>::value;

template <typename propertyT, typename syclObjectT>
inline constexpr bool is_property_of_v =
    is_property_of<propertyT, syclObjectT>::value;

} // namespace _V1
} // namespace sycl
