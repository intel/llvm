//==------------------- common_functions.hpp -------------------------------==//
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
BUILTIN_CREATE_ENABLER(builtin_enable_common, default_ret_type, fp_elem_type,
                       any_shape, same_elem_type)
BUILTIN_CREATE_ENABLER(builtin_enable_common_non_scalar, default_ret_type,
                       fp_elem_type, non_scalar_only, same_elem_type)

#ifdef __SYCL_DEVICE_ONLY__
#define BUILTIN_COMMON(NUM_ARGS, NAME, SPIRV_IMPL)                             \
  DEVICE_IMPL_TEMPLATE(NUM_ARGS, NAME, builtin_enable_common_t, SPIRV_IMPL)
#else
#define BUILTIN_COMMON(NUM_ARGS, NAME, SPIRV_IMPL)                             \
  HOST_IMPL_TEMPLATE(NUM_ARGS, NAME, builtin_enable_common_t, common,          \
                     default_ret_type)
#endif

BUILTIN_COMMON(ONE_ARG, degrees, __spirv_ocl_degrees)
BUILTIN_COMMON(ONE_ARG, radians, __spirv_ocl_radians)
BUILTIN_COMMON(ONE_ARG, sign, __spirv_ocl_sign)

BUILTIN_COMMON(THREE_ARGS, mix, __spirv_ocl_mix)
template <typename T0, typename T1>
detail::builtin_enable_common_non_scalar_t<T0, T1>
mix(T0 x, T1 y, detail::get_elem_type_t<T0> z) {
  return mix(detail::simplify_if_swizzle_t<T0>{x},
             detail::simplify_if_swizzle_t<T0>{y},
             detail::simplify_if_swizzle_t<T0>{z});
}

BUILTIN_COMMON(TWO_ARGS, step, __spirv_ocl_step)
template <typename T>
detail::builtin_enable_common_non_scalar_t<T> step(detail::get_elem_type_t<T> x,
                                                   T y) {
  return step(detail::simplify_if_swizzle_t<T>{x},
              detail::simplify_if_swizzle_t<T>{y});
}

BUILTIN_COMMON(THREE_ARGS, smoothstep, __spirv_ocl_smoothstep)
template <typename T>
detail::builtin_enable_common_non_scalar_t<T>
smoothstep(detail::get_elem_type_t<T> x, detail::get_elem_type_t<T> y, T z) {
  return smoothstep(detail::simplify_if_swizzle_t<T>{x},
                    detail::simplify_if_swizzle_t<T>{y},
                    detail::simplify_if_swizzle_t<T>{z});
}

BUILTIN_COMMON(TWO_ARGS, max, __spirv_ocl_fmax_common)
template <typename T>
detail::builtin_enable_common_non_scalar_t<T>
max(T x, detail::get_elem_type_t<T> y) {
  return max(detail::simplify_if_swizzle_t<T>{x},
             detail::simplify_if_swizzle_t<T>{y});
}

BUILTIN_COMMON(TWO_ARGS, min, __spirv_ocl_fmin_common)
template <typename T>
detail::builtin_enable_common_non_scalar_t<T>
min(T x, detail::get_elem_type_t<T> y) {
  return min(detail::simplify_if_swizzle_t<T>{x},
             detail::simplify_if_swizzle_t<T>{y});
}

BUILTIN_COMMON(THREE_ARGS, clamp, __spirv_ocl_fclamp)
template <typename T>
detail::builtin_enable_common_non_scalar_t<T>
clamp(T x, detail::get_elem_type_t<T> y, detail::get_elem_type_t<T> z) {
  return clamp(detail::simplify_if_swizzle_t<T>{x},
               detail::simplify_if_swizzle_t<T>{y},
               detail::simplify_if_swizzle_t<T>{z});
}

#undef BUILTIN_COMMON
} // namespace _V1
} // namespace sycl
