//==------------------- native_math_functions.hpp --------------------------==//
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
BUILTIN_CREATE_ENABLER(builtin_enable_native_math, default_ret_type,
                       float_elem_type, non_scalar_only, same_elem_type)

#ifdef __SYCL_DEVICE_ONLY__
#define BUILTIN_NATIVE(NUM_ARGS, NAME)                                         \
  inline float NAME(NUM_ARGS##_TYPE_ARG(float)) {                              \
    return __spirv_ocl_native_##NAME(NUM_ARGS##_ARG);                          \
  }                                                                            \
  DEVICE_IMPL_TEMPLATE_CUSTOM_DELEGATE(                                        \
      NUM_ARGS, NAME, builtin_enable_native_math_t, builtin_marray_impl,       \
      native, __spirv_ocl_native_##NAME)
#else
#define BUILTIN_NATIVE(NUM_ARGS, NAME)                                         \
  HOST_IMPL_SCALAR(NUM_ARGS, NAME, float)                                      \
  HOST_IMPL_TEMPLATE(NUM_ARGS, NAME, builtin_enable_native_math_t, native,     \
                     default_ret_type)
#endif

namespace native {
BUILTIN_NATIVE(ONE_ARG, cos)
BUILTIN_NATIVE(ONE_ARG, exp)
BUILTIN_NATIVE(ONE_ARG, exp10)
BUILTIN_NATIVE(ONE_ARG, exp2)
BUILTIN_NATIVE(ONE_ARG, log)
BUILTIN_NATIVE(ONE_ARG, log10)
BUILTIN_NATIVE(ONE_ARG, log2)
BUILTIN_NATIVE(ONE_ARG, recip)
BUILTIN_NATIVE(ONE_ARG, rsqrt)
BUILTIN_NATIVE(ONE_ARG, sin)
BUILTIN_NATIVE(ONE_ARG, sqrt)
BUILTIN_NATIVE(ONE_ARG, tan)
BUILTIN_NATIVE(TWO_ARGS, divide)
BUILTIN_NATIVE(TWO_ARGS, powr)
} // namespace native

#undef BUILTIN_NATIVE
#undef VEC_EXTERN
} // namespace _V1
} // namespace sycl
