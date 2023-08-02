//==----------- functional.hpp --- SYCL functional -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

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

// std:logical_and/std::logical_or with a non-void type returns bool,
// sycl requires returning T.
template <typename T = void> struct logical_and {
  T operator()(const T &lhs, const T &rhs) const { return lhs && rhs; }
};

template <> struct logical_and<void> : std::logical_and<void> {};

template <typename T = void> struct logical_or {
  T operator()(const T &lhs, const T &rhs) const { return lhs || rhs; }
};

template <> struct logical_or<void> : std::logical_or<void> {};

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

} // namespace _V1
} // namespace sycl
