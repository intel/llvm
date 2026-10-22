//==----------- functional.hpp --- SYCL functional -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/known_op_traits.hpp> // for known_op_traits

#include <functional>  // for logical_and, logical_or, bit_and, bit_or, bit...
#include <type_traits> // for common_type
#include <utility>     // for forward

namespace sycl {
inline namespace _V1 {

template <typename T = void> using plus = std::plus<T>;
template <typename T = void> using multiplies = std::multiplies<T>;
template <typename T = void> using bit_and = std::bit_and<T>;
template <typename T = void> using bit_or = std::bit_or<T>;
template <typename T = void> using bit_xor = std::bit_xor<T>;
template <typename T = void> struct logical_and : std::logical_and<T> {};
template <typename T = void> struct logical_or : std::logical_or<T> {};

// sycl::minimum definition should be consistent with std::min
template <typename T = void> struct minimum {
  T operator()(const T &lhs, const T &rhs) const {
    return (rhs < lhs) ? rhs : lhs;
  }
};

template <> struct minimum<void> {
  struct is_transparent {};
  template <typename T, typename U>
  auto operator()(T &&lhs, U &&rhs) const ->
      typename std::common_type<T &&, U &&>::type {
    return (std::forward<const U>(rhs) < std::forward<const T>(lhs))
               ? std::forward<U>(rhs)
               : std::forward<T>(lhs);
  }
};

// sycl::maximum definition should be consistent with std::max
template <typename T = void> struct maximum {
  T operator()(const T &lhs, const T &rhs) const {
    return (lhs < rhs) ? rhs : lhs;
  }
};

template <> struct maximum<void> {
  struct is_transparent {};
  template <typename T, typename U>
  auto operator()(T &&lhs, U &&rhs) const ->
      typename std::common_type<T &&, U &&>::type {
    return (std::forward<const T>(lhs) < std::forward<const U>(rhs))
               ? std::forward<U>(rhs)
               : std::forward<T>(lhs);
  }
};

namespace detail {
// Register the function objects recognized by sycl/known_identity.hpp. The
// std:: ones cover the sycl::plus, sycl::multiplies and sycl::bit_* aliases;
// std::logical_and/std::logical_or are recognized as a DPC++ extension.
#define __SYCL_KNOWN_OP(OP, KIND)                                              \
  template <typename T>                                                        \
  struct known_op_traits<OP<T>>                                                \
      : known_op_traits_base<known_op_kind::KIND, T> {};
__SYCL_KNOWN_OP(std::plus, plus)
__SYCL_KNOWN_OP(std::multiplies, multiplies)
__SYCL_KNOWN_OP(std::bit_and, bit_and)
__SYCL_KNOWN_OP(std::bit_or, bit_or)
__SYCL_KNOWN_OP(std::bit_xor, bit_xor)
__SYCL_KNOWN_OP(std::logical_and, logical_and)
__SYCL_KNOWN_OP(std::logical_or, logical_or)
__SYCL_KNOWN_OP(sycl::logical_and, logical_and)
__SYCL_KNOWN_OP(sycl::logical_or, logical_or)
__SYCL_KNOWN_OP(sycl::minimum, minimum)
__SYCL_KNOWN_OP(sycl::maximum, maximum)
#undef __SYCL_KNOWN_OP
} // namespace detail

} // namespace _V1
} // namespace sycl
