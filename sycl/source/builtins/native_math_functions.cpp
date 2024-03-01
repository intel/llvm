//==------------------- native_math_functions.cpp --------------------------==//
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
namespace native {
#define BUILTIN_NATIVE_CUSTOM(NUM_ARGS, NAME, IMPL)                            \
  HOST_IMPL(NAME, IMPL)                                                        \
  EXPORT_SCALAR_AND_VEC_1_16_NS(NUM_ARGS, NAME, native, float)

#define BUILTIN_NATIVE(NUM_ARGS, NAME)                                         \
  BUILTIN_NATIVE_CUSTOM(NUM_ARGS, NAME, std::NAME)

BUILTIN_NATIVE(ONE_ARG, cos)
BUILTIN_NATIVE_CUSTOM(TWO_ARGS, divide, [](auto x, auto y) { return x / y; })
BUILTIN_NATIVE(ONE_ARG, exp)
BUILTIN_NATIVE(ONE_ARG, exp2)
BUILTIN_NATIVE_CUSTOM(ONE_ARG, exp10, [](auto x) { return std::pow(10.0f, x); })
BUILTIN_NATIVE(ONE_ARG, log)
BUILTIN_NATIVE(ONE_ARG, log2)
BUILTIN_NATIVE(ONE_ARG, log10)
BUILTIN_NATIVE_CUSTOM(TWO_ARGS, powr, [](auto x, auto y) {
  return (x >= 0 ? std::pow(x, y) : x);
})
BUILTIN_NATIVE_CUSTOM(ONE_ARG, recip, [](auto x) { return 1.0f / x; })
BUILTIN_NATIVE_CUSTOM(ONE_ARG, rsqrt,
                      [](auto x) { return 1.0f / std::sqrt(x); })
BUILTIN_NATIVE(ONE_ARG, sin)
BUILTIN_NATIVE(ONE_ARG, sqrt)
BUILTIN_NATIVE(ONE_ARG, tan)
} // namespace native
} // namespace _V1
} // namespace sycl
