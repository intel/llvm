//==----------- functional.hpp --- SYCL functional -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <functional>

__SYCL_INLINE namespace cl {
namespace sycl {
namespace intel {

template <typename T = void> struct minimum {
  T operator()(const T &lhs, const T &rhs) const {
    return std::less<T>()(lhs, rhs) ? lhs : rhs;
  }
};

#if __cplusplus >= 201402L
template <> struct minimum<void> {
  struct is_transparent {};
  template <typename T, typename U>
  auto operator()(T &&lhs, U &&rhs) const ->
      typename std::common_type<T &&, U &&>::type {
    return std::less<>()(std::forward<const T>(lhs), std::forward<const U>(rhs))
               ? std::forward<T>(lhs)
               : std::forward<U>(rhs);
  }
};
#endif

template <typename T = void> struct maximum {
  T operator()(const T &lhs, const T &rhs) const {
    return std::greater<T>()(lhs, rhs) ? lhs : rhs;
  }
};

#if __cplusplus >= 201402L
template <> struct maximum<void> {
  struct is_transparent {};
  template <typename T, typename U>
  auto operator()(T &&lhs, U &&rhs) const ->
      typename std::common_type<T &&, U &&>::type {
    return std::greater<>()(std::forward<const T>(lhs), std::forward<const U>(rhs))
               ? std::forward<T>(lhs)
               : std::forward<U>(rhs);
  }
};
#endif

template <typename T = void> using plus = std::plus<T>;

} // namespace intel
} // namespace sycl
} // namespace cl
