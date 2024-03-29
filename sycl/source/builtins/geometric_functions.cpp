//==------------------- geometric_functions.cpp ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/builtins_preview.hpp>

#include "host_helper_macros.hpp"

#include <cmath>

namespace sycl {
inline namespace _V1 {
template <typename T> static inline T cross_host_impl(T p0, T p1) {
  T result(0);
  result.x() = p0.y() * p1.z() - p0.z() * p1.y();
  result.y() = p0.z() * p1.x() - p0.x() * p1.z();
  result.z() = p0.x() * p1.y() - p0.y() * p1.x();
  return result;
}
EXPORT_VEC_3_4(TWO_ARGS, cross, FP_TYPES)

template <typename T0, typename T1>
static inline auto dot_host_impl(T0 x, T1 y) {
  if constexpr (detail::is_scalar_arithmetic<T0>::value) {
    return x * y;
  } else {
    auto R = x[0] * y[0];
    for (size_t i = 1; i < detail::num_elements<T0>::value; ++i)
      R += x[i] * y[i];
    return R;
  }
}
EXPORT_SCALAR_AND_VEC_2_4(TWO_ARGS, dot, FP_TYPES)

#if defined(__GNUC__) && !defined(__clang__)
// GCC miscompiles if using dot (instead of dot_host_impl) *or* if
// optimizations aren't disabled here. Not sure if a bug in GCC or some UB in
// sycl::vec/sycl::half (like ansi-alias violations).
#pragma GCC push_options
#pragma GCC optimize("O0")
#endif
template <typename T> static inline auto length_host_impl(T x) {
  auto d = dot_host_impl(x, x);
  return static_cast<decltype(d)>(std::sqrt(d));
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
EXPORT_SCALAR_AND_VEC_2_4(ONE_ARG, length, FP_TYPES)

// fast_length on host is the same as just length.
template <typename T> static inline auto fast_length_host_impl(T x) {
  return length_host_impl(x);
}
EXPORT_SCALAR_AND_VEC_2_4(ONE_ARG, fast_length, float)

template <typename T0, typename T1>
static inline auto distance_host_impl(T0 x, T1 y) {
  return length(x - y);
}
EXPORT_SCALAR_AND_VEC_2_4(TWO_ARGS, distance, FP_TYPES)
// fast_distance on host is the same as just distance.
template <typename T0, typename T1>
static inline auto fast_distance_host_impl(T0 x, T1 y) {
  return distance_host_impl(x, y);
}
EXPORT_SCALAR_AND_VEC_2_4(TWO_ARGS, fast_distance, float)

template <typename T> static inline auto normalize_host_impl(T x) {
  auto len = length(x);
  if (len == 0)
    return x;
  return x / len;
}
EXPORT_SCALAR_AND_VEC_2_4(ONE_ARG, normalize, FP_TYPES)
// fast_normalize on host is the same as just normalize.
template <typename T> static inline auto fast_normalize_host_impl(T x) {
  return normalize_host_impl(x);
}
EXPORT_SCALAR_AND_VEC_2_4(ONE_ARG, fast_normalize, float)
} // namespace _V1
} // namespace sycl
