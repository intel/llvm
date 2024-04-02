//==----------------- util.hpp - DPC++ Explicit SIMD API  ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Utility functions used for implementing experimental Explicit SIMD APIs.
//===----------------------------------------------------------------------===//

#pragma once

/// @cond ESIMD_DETAIL

#include <sycl/ext/intel/esimd/detail/util.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::intel::experimental::esimd::detail {

/// Compile-time checks if first template parameter is equal for any other
template <typename...> struct is_one_of {
  static constexpr bool value = false;
};

template <typename Checked, typename First, typename... Other>
struct is_one_of<Checked, First, Other...> {
  static constexpr bool value = std::is_same_v<std::remove_const_t<Checked>,
                                               std::remove_const_t<First>> ||
                                is_one_of<Checked, Other...>::value;
};
template <typename Checked, typename... T>
inline constexpr bool is_one_of_v = is_one_of<Checked, T...>::value;

/// Compile-time checks if compile-time known  element of enum class is equal
/// for any other compile-time known elements of enum
template <typename enumClass, enumClass... E> struct is_one_of_enum {
  static constexpr bool value = false;
};

template <typename enumClass, enumClass Checked, enumClass First,
          enumClass... Else>
struct is_one_of_enum<enumClass, Checked, First, Else...> {
  static constexpr bool value =
      (Checked == First) || is_one_of_enum<enumClass, Checked, Else...>::value;
};
template <typename enumClass, enumClass... T>
inline constexpr bool is_one_of_enum_v = is_one_of_enum<enumClass, T...>::value;

} // namespace ext::intel::experimental::esimd::detail
} // namespace _V1
} // namespace sycl

/// @endcond ESIMD_DETAIL
