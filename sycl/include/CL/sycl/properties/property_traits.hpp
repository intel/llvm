//==------------ property_traits.hpp --- SYCL property traits --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// Property traits
template <typename propertyT> struct is_property : public std::false_type {};

template <typename propertyT, typename syclObjectT>
struct is_property_of : public std::false_type {};

template <typename propertyT>
__SYCL_INLINE_CONSTEXPR bool is_property_v = is_property<propertyT>::value;

template <typename propertyT, typename syclObjectT>
__SYCL_INLINE_CONSTEXPR bool is_property_of_v =
    is_property_of<propertyT, syclObjectT>::value;

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
