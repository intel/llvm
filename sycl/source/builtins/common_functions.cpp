//==------------------- common_functions.cpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Define _USE_MATH_DEFINES to enforce math defines of macros like M_PI in
// <cmath>. _USE_MATH_DEFINES is defined here before includes of SYCL header
// files to avoid include of <cmath> via those SYCL headers with unset
// _USE_MATH_DEFINES.
//
// Note that C++20 has std::numbers containing the constants but we're limited
// to C++17.
#define _USE_MATH_DEFINES
#include <cmath>

#include <sycl/builtins_preview.hpp>

#include "host_helper_macros.hpp"

namespace sycl {
inline namespace _V1 {
#define BUILTIN_COMMON(NUM_ARGS, NAME, IMPL)                                   \
  HOST_IMPL(NAME, IMPL)                                                        \
  EXPORT_SCALAR_AND_VEC_1_16(NUM_ARGS, NAME, FP_TYPES)

BUILTIN_COMMON(ONE_ARG, degrees,
               [](auto x) -> decltype(x) { return (180 / M_PI) * x; })

BUILTIN_COMMON(ONE_ARG, radians,
               [](auto x) -> decltype(x) { return (M_PI / 180) * x; })

BUILTIN_COMMON(ONE_ARG, sign, [](auto x) -> decltype(x) {
  using T = decltype(x);
  if (std::isnan(x))
    return T(0.0);
  if (x > 0)
    return T(1.0);
  if (x < 0)
    return T(-1.0);
  /* x is +0.0 or -0.0 */
  return x;
})

BUILTIN_COMMON(THREE_ARGS, mix, [](auto x, auto y, auto z) -> decltype(x) {
  return x + (y - x) * z;
})

BUILTIN_COMMON(TWO_ARGS, step,
               [](auto x, auto y) -> decltype(x) { return y < x ? 0.0 : 1.0; })

BUILTIN_COMMON(THREE_ARGS, smoothstep,
               [](auto x, auto y, auto z) -> decltype(x) {
                 using T = decltype(x);
                 auto t = sycl::clamp((z - x) / (y - x), T{0}, T{1});
                 return t * t * (3 - 2 * t);
               })

BUILTIN_COMMON(TWO_ARGS, max,
               [](auto x, auto y) -> decltype(x) { return (x < y ? y : x); })
BUILTIN_COMMON(TWO_ARGS, min,
               [](auto x, auto y) -> decltype(x) { return (y < x ? y : x); })

// clamp is implemented for INTEGER_TYPES as well, so expand/inline
// BUILTIN_COMMON manually.
HOST_IMPL(clamp, [](auto x, auto y, auto z) -> decltype(x) {
  using ElemTy = detail::get_elem_type_t<decltype(x)>;
  if constexpr (std::is_integral_v<ElemTy>) {
    return std::min(std::max(x, y), z);
  } else {
    return std::fmin(std::fmax(x, y), z);
  }
})
EXPORT_SCALAR_AND_VEC_1_16(THREE_ARGS, clamp, INTEGER_TYPES, FP_TYPES)
} // namespace _V1
} // namespace sycl
