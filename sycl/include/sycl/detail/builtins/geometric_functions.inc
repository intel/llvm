//==------------------- geometric_functions.hpp ----------------------------==//
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
namespace detail {
template <typename T>
struct shape_geo : std::bool_constant<is_valid_size_v<T, 2, 3, 4> ||
                                      is_scalar_arithmetic_v<T>> {};
template <typename T>
struct shape_geo3or4 : std::bool_constant<is_valid_size_v<T, 3, 4>> {};
} // namespace detail

BUILTIN_CREATE_ENABLER(builtin_enable_geo, default_ret_type, fp_elem_type,
                       shape_geo, same_elem_type)
BUILTIN_CREATE_ENABLER(builtin_enable_geo_fast, default_ret_type,
                       float_elem_type, shape_geo, same_elem_type)
BUILTIN_CREATE_ENABLER(builtin_enable_geo_scalar_ret, scalar_ret_type,
                       fp_elem_type, shape_geo, same_elem_type)
BUILTIN_CREATE_ENABLER(builtin_enable_geo_fast_scalar_ret, scalar_ret_type,
                       float_elem_type, shape_geo, same_elem_type)
BUILTIN_CREATE_ENABLER(builtin_enable_geo3or4, default_ret_type, fp_elem_type,
                       shape_geo3or4, same_elem_type)

namespace detail {
template <typename FuncTy, typename... Ts>
auto builtin_delegate_geo_impl(FuncTy F, const Ts &...x) {
  using T = typename first_type<Ts...>::type;
  if constexpr (is_marray_v<T>) {
    auto ret = F(to_vec(x)...);
    if constexpr (is_vec_v<decltype(ret)>)
      return to_marray(ret);
    else
      return ret;
  } else {
    return F(simplify_if_swizzle_t<T>{x}...);
  }
}
} // namespace detail

#ifdef __SYCL_DEVICE_ONLY__
#define BUILTIN_GEO(NUM_ARGS, NAME, ENABLER, RET_TYPE_TRAITS)                  \
  DEVICE_IMPL_TEMPLATE_CUSTOM_DELEGATE(NUM_ARGS, NAME, ENABLER,                \
                                       builtin_delegate_geo_impl, sycl,        \
                                       __spirv_ocl_##NAME)
#else
#define BUILTIN_GEO(NUM_ARGS, NAME, ENABLER, RET_TYPE_TRAITS)                  \
  HOST_IMPL_TEMPLATE_CUSTOM_DELEGATOR(NUM_ARGS, NAME, ENABLER, geo,            \
                                      RET_TYPE_TRAITS,                         \
                                      builtin_delegate_geo_impl)
#endif

#ifdef __SYCL_DEVICE_ONLY__
DEVICE_IMPL_TEMPLATE_CUSTOM_DELEGATE(TWO_ARGS, cross, builtin_enable_geo3or4_t,
                                     builtin_delegate_geo_impl, sycl,
                                     __spirv_ocl_cross)
#else
BUILTIN_GEO(TWO_ARGS, cross, builtin_enable_geo3or4_t, default_ret_type)
#endif

#ifdef __SYCL_DEVICE_ONLY__
DEVICE_IMPL_TEMPLATE_CUSTOM_DELEGATE(
    TWO_ARGS, dot, builtin_enable_geo_scalar_ret_t, builtin_delegate_geo_impl,
    sycl, [](auto x, auto y) {
      if constexpr (detail::is_scalar_arithmetic_v<decltype(x)>)
        return x * y;
      else {
        return __spirv_Dot(x, y);
      }
    })
#else
BUILTIN_GEO(TWO_ARGS, dot, builtin_enable_geo_scalar_ret_t, scalar_ret_type)
#endif

// FIXME: fast_* should use *fast*_t enablers.
BUILTIN_GEO(ONE_ARG, length, builtin_enable_geo_scalar_ret_t, scalar_ret_type)
BUILTIN_GEO(ONE_ARG, fast_length, builtin_enable_geo_scalar_ret_t,
            scalar_ret_type)
BUILTIN_GEO(TWO_ARGS, distance, builtin_enable_geo_scalar_ret_t,
            scalar_ret_type)
BUILTIN_GEO(TWO_ARGS, fast_distance, builtin_enable_geo_scalar_ret_t,
            scalar_ret_type)
BUILTIN_GEO(ONE_ARG, normalize, builtin_enable_geo_t, default_ret_type)
BUILTIN_GEO(ONE_ARG, fast_normalize, builtin_enable_geo_t, default_ret_type)

#undef BUILTIN_GEO
} // namespace _V1
} // namespace sycl
