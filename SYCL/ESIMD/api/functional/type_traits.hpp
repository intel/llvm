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

#include <type_traits>

namespace esimd_test {
namespace api {
namespace functional {
namespace type_traits {

template <typename T>
using is_sycl_floating_point =
    std::bool_constant<std::is_floating_point_v<T> ||
                       std::is_same_v<T, sycl::half>>;

template <typename T>
inline constexpr bool is_sycl_floating_point_v{
    is_sycl_floating_point<T>::value};

} // namespace type_traits
} // namespace functional
} // namespace api
} // namespace esimd_test
