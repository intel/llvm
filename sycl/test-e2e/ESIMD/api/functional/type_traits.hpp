//===-- type_traits.hpp - Define functions for iterating with datatypes. --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides function for iterating with data types.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/core.hpp>

#include <type_traits>
#include <utility>

namespace esimd_test::api::functional::type_traits {

template <typename T>
using is_sycl_floating_point =
    std::bool_constant<std::is_floating_point_v<T> ||
                       std::is_same_v<T, sycl::half>>;

template <typename T>
inline constexpr bool is_sycl_floating_point_v{
    is_sycl_floating_point<T>::value};

template <typename T>
using is_sycl_signed =
    std::bool_constant<std::is_signed_v<T> || is_sycl_floating_point_v<T>>;

template <typename T>
inline constexpr bool is_sycl_signed_v{is_sycl_signed<T>::value};

template <typename T>
using is_nonconst_rvalue_reference =
    std::bool_constant<std::is_rvalue_reference_v<T> &&
                       !std::is_const_v<typename std::remove_reference_t<T>>>;

template <typename T>
inline constexpr bool is_nonconst_rvalue_reference_v{
    is_nonconst_rvalue_reference<T>::value};

namespace has_static_member {

template <typename, typename = void> struct to_string : std::false_type {};

template <typename T>
struct to_string<T, std::void_t<decltype(T::to_string())>> : std::true_type {};

} // namespace has_static_member

} // namespace esimd_test::api::functional::type_traits
