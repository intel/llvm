//==------------------- half_precision_math_functions.hpp ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Intentionally insufficient set of includes and no "#pragma once".

#include <sycl/detail/builtins/helper_macros.hpp>

namespace sycl {
inline namespace _V1 {
BUILTIN_CREATE_ENABLER(builtin_enable_half_precision_math, default_ret_type,
                       float_elem_type, non_scalar_only, same_elem_type)

#ifdef __SYCL_DEVICE_ONLY__
#define BUILTIN_HALF(NUM_ARGS, NAME)                                           \
  inline float NAME(NUM_ARGS##_TYPE_ARG(float)) {                              \
    return __spirv_ocl_half_##NAME(NUM_ARGS##_ARG);                            \
  }                                                                            \
  DEVICE_IMPL_TEMPLATE_CUSTOM_DELEGATE(                                        \
      NUM_ARGS, NAME, builtin_enable_half_precision_math_t,                    \
      builtin_marray_impl, half_precision, __spirv_ocl_half_##NAME)
#else
#define BUILTIN_HALF(NUM_ARGS, NAME)                                           \
  HOST_IMPL_SCALAR(NUM_ARGS, NAME, float)                                      \
  HOST_IMPL_TEMPLATE(NUM_ARGS, NAME, builtin_enable_half_precision_math_t,     \
                     half_precision, default_ret_type)
#endif

namespace half_precision {
BUILTIN_HALF(ONE_ARG, cos)
BUILTIN_HALF(TWO_ARGS, divide)
BUILTIN_HALF(ONE_ARG, exp)
BUILTIN_HALF(ONE_ARG, exp2)
BUILTIN_HALF(ONE_ARG, exp10)
BUILTIN_HALF(ONE_ARG, log)
BUILTIN_HALF(ONE_ARG, log2)
BUILTIN_HALF(ONE_ARG, log10)
BUILTIN_HALF(TWO_ARGS, powr)
BUILTIN_HALF(ONE_ARG, recip)
BUILTIN_HALF(ONE_ARG, rsqrt)
BUILTIN_HALF(ONE_ARG, sin)
BUILTIN_HALF(ONE_ARG, sqrt)
BUILTIN_HALF(ONE_ARG, tan)
} // namespace half_precision

#undef BUILTIN_HALF
} // namespace _V1
} // namespace sycl
