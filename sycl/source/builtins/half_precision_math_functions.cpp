//==------------------- half_precision_math_functions.cpp ------------------==//
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
namespace half_precision {
#define BUILTIN_HALF_CUSTOM(NUM_ARGS, NAME, IMPL)                              \
  HOST_IMPL(NAME, IMPL)                                                        \
  EXPORT_SCALAR_AND_VEC_1_16_NS(NUM_ARGS, NAME, half_precision, float)

#define BUILTIN_HALF(NUM_ARGS, NAME)                                           \
  BUILTIN_HALF_CUSTOM(NUM_ARGS, NAME, std::NAME)

BUILTIN_HALF(ONE_ARG, cos)
BUILTIN_HALF_CUSTOM(TWO_ARGS, divide, [](auto x, auto y) { return x / y; })
BUILTIN_HALF(ONE_ARG, exp)
BUILTIN_HALF(ONE_ARG, exp2)
BUILTIN_HALF_CUSTOM(ONE_ARG, exp10, [](auto x) { return std::pow(10.0f, x); })
BUILTIN_HALF(ONE_ARG, log)
BUILTIN_HALF(ONE_ARG, log2)
BUILTIN_HALF(ONE_ARG, log10)
BUILTIN_HALF_CUSTOM(TWO_ARGS, powr, [](auto x, auto y) {
  return (x >= 0 ? std::pow(x, y) : x);
})
BUILTIN_HALF_CUSTOM(ONE_ARG, recip, [](auto x) { return 1.0f / x; })
BUILTIN_HALF_CUSTOM(ONE_ARG, rsqrt, [](auto x) { return 1.0f / std::sqrt(x); })
BUILTIN_HALF(ONE_ARG, sin)
BUILTIN_HALF(ONE_ARG, sqrt)
BUILTIN_HALF(ONE_ARG, tan)
} // namespace half_precision
} // namespace _V1
} // namespace sycl
