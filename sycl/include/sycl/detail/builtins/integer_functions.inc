//==------------------- integer_functions.hpp ------------------------------==//
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
struct integer_elem_type
    : std::bool_constant<
          (is_vec_or_swizzle_v<T> &&
           check_type_in_v<get_elem_type_t<T>, FIXED_WIDTH_INTEGER_TYPES>) ||
          (!is_vec_or_swizzle_v<T> &&
           check_type_in_v<get_elem_type_t<T>, INTEGER_TYPES>)> {};
template <typename T>
struct suint32_elem_type
    : std::bool_constant<
          check_type_in_v<get_elem_type_t<T>, int32_t, uint32_t>> {};
} // namespace detail
BUILTIN_CREATE_ENABLER(builtin_enable_integer, default_ret_type,
                       integer_elem_type, any_shape, same_elem_type)
BUILTIN_CREATE_ENABLER(builtin_enable_integer_non_scalar, default_ret_type,
                       integer_elem_type, non_scalar_only, same_elem_type)
BUILTIN_CREATE_ENABLER(builtin_enable_suint32, default_ret_type,
                       suint32_elem_type, any_shape, same_elem_type)

#ifdef __SYCL_DEVICE_ONLY__
#define BUILTIN_GENINT(NUM_ARGS, NAME)                                         \
  DEVICE_IMPL_TEMPLATE(                                                        \
      NUM_ARGS, NAME, builtin_enable_integer_t, [](auto... xs) {               \
        using ret_ty =                                                         \
            detail::builtin_enable_integer_t<NUM_ARGS##_TEMPLATE_TYPE>;        \
        return detail::builtins::convert_result<ret_ty>(                       \
            __spirv_ocl_##NAME(xs...));                                        \
      })
#else
#define BUILTIN_GENINT(NUM_ARGS, NAME)                                         \
  HOST_IMPL_TEMPLATE(NUM_ARGS, NAME, builtin_enable_integer_t, integer,        \
                     default_ret_type)
#endif

// Integer builtin with separate signed/unsigned implementations.
#ifdef __SYCL_DEVICE_ONLY__
#define BUILTIN_GENINT_SU(NUM_ARGS, NAME)                                      \
  DEVICE_IMPL_TEMPLATE(                                                        \
      NUM_ARGS, NAME, builtin_enable_integer_t, [](auto... xs) {               \
        using ret_ty =                                                         \
            detail::builtin_enable_integer_t<NUM_ARGS##_TEMPLATE_TYPE>;        \
        using detail::builtins::convert_result;                                \
        if constexpr (std::is_signed_v<detail::get_elem_type_t<T0>>)           \
          return convert_result<ret_ty>(__spirv_ocl_s_##NAME(xs...));          \
        else                                                                   \
          return convert_result<ret_ty>(__spirv_ocl_u_##NAME(xs...));          \
      })
#else
#define BUILTIN_GENINT_SU(NUM_ARGS, NAME) BUILTIN_GENINT(NUM_ARGS, NAME)
#endif

#if __SYCL_DEVICE_ONLY__
DEVICE_IMPL_TEMPLATE(ONE_ARG, abs, builtin_enable_integer_t, [](auto x) {
  using ret_ty = detail::builtin_enable_integer_t<T0>;
  using detail::builtins::convert_result;
  if constexpr (std::is_signed_v<detail::get_elem_type_t<T0>>)
    // SPIR-V builtin returns unsigned type, SYCL's return type is signed
    // with the following restriction:
    //   > The behavior is undefined if the result cannot be represented by
    //   > the return type
    return convert_result<ret_ty>(bit_cast<T0>(__spirv_ocl_s_abs(x)));
  else
    return convert_result<ret_ty>(__spirv_ocl_u_abs(x));
})
#else
BUILTIN_GENINT_SU(ONE_ARG, abs)
#endif

BUILTIN_GENINT_SU(TWO_ARGS, add_sat)

#if __SYCL_DEVICE_ONLY__
DEVICE_IMPL_TEMPLATE(
    TWO_ARGS, abs_diff, builtin_enable_integer_t, [](auto... xs) {
      using ret_ty = detail::builtin_enable_integer_t<T0>;
      using detail::builtins::convert_result;
      if constexpr (std::is_signed_v<detail::get_elem_type_t<T0>>) {
        // SPIRV built-in returns [vector of] unsigned type(s).
        auto ret = __spirv_ocl_s_abs_diff(xs...);
        if constexpr (detail::is_vec_v<T0>) {
          // SYCL 2020 revision 8's abs_diff returns T0 (or corresponding vec in
          // case of a swizzle). The only way to produce signed ext_vector_type
          // from unsigned is with C-style case. Also note that element type of
          // sycl::vec and ext_vector_type might be different, e.g.
          // sycl::vec<char, N>::vector_t is
          // signed char __attribute__((ext_vector_type(N))).
          //
          // TODO: Shouldn't be different from "abs" above.
          return convert_result<ret_ty>((typename T0::vector_t)(ret));
        } else {
          return convert_result<ret_ty>(ret);
        }
      } else {
        return convert_result<ret_ty>(__spirv_ocl_u_abs_diff(xs...));
      }
    })
#else
BUILTIN_GENINT_SU(TWO_ARGS, abs_diff)
#endif

BUILTIN_GENINT_SU(TWO_ARGS, hadd)
BUILTIN_GENINT_SU(TWO_ARGS, mul_hi)
BUILTIN_GENINT_SU(TWO_ARGS, rhadd)
BUILTIN_GENINT_SU(TWO_ARGS, sub_sat)
BUILTIN_GENINT_SU(THREE_ARGS, mad_hi)
BUILTIN_GENINT_SU(THREE_ARGS, mad_sat)

BUILTIN_GENINT_SU(TWO_ARGS, max)
template <typename T>
detail::builtin_enable_integer_non_scalar_t<T>
max(T x, detail::get_elem_type_t<T> y) {
  return max(detail::simplify_if_swizzle_t<T>{x},
             detail::simplify_if_swizzle_t<T>{y});
}

BUILTIN_GENINT_SU(TWO_ARGS, min)
template <typename T>
detail::builtin_enable_integer_non_scalar_t<T>
min(T x, detail::get_elem_type_t<T> y) {
  return min(detail::simplify_if_swizzle_t<T>{x},
             detail::simplify_if_swizzle_t<T>{y});
}

BUILTIN_GENINT_SU(THREE_ARGS, clamp)
template <typename T>
detail::builtin_enable_integer_non_scalar_t<T>
clamp(T x, detail::get_elem_type_t<T> y, detail::get_elem_type_t<T> z) {
  return clamp(detail::simplify_if_swizzle_t<T>{x},
               detail::simplify_if_swizzle_t<T>{y},
               detail::simplify_if_swizzle_t<T>{z});
}

BUILTIN_GENINT(ONE_ARG, clz)
BUILTIN_GENINT(ONE_ARG, ctz)
BUILTIN_GENINT(ONE_ARG, popcount)
BUILTIN_GENINT(TWO_ARGS, rotate)

#undef BUILTIN_GENINT
#undef BUILTIN_GENINT_SU

#ifdef __SYCL_DEVICE_ONLY__
DEVICE_IMPL_TEMPLATE(
    THREE_ARGS, mad24, builtin_enable_suint32_t, [](auto... xs) {
      if constexpr (std::is_same_v<int32_t, detail::get_elem_type_t<T0>>) {
        return __spirv_ocl_s_mad24(xs...);
      } else {
        return __spirv_ocl_u_mad24(xs...);
      }
    })
#else
template <typename T0, typename T1, typename T2>
detail::builtin_enable_suint32_t<T0, T1, T2> mad24(T0 x, T1 y, T2 z) {
  if constexpr (detail::is_scalar_arithmetic_v<T0>) {
    return x * y + z;
  } else {
    detail::simplify_if_swizzle_t<T0> res;
    for (int i = 0; i < detail::num_elements<T0>::value; ++i)
      res[i] = x[i] * y[i] + z[i];
    return res;
  }
}
#endif

#ifdef __SYCL_DEVICE_ONLY__
DEVICE_IMPL_TEMPLATE(TWO_ARGS, mul24, builtin_enable_suint32_t, [](auto... xs) {
  if constexpr (std::is_same_v<int32_t, detail::get_elem_type_t<T0>>) {
    return __spirv_ocl_s_mul24(xs...);
  } else {
    return __spirv_ocl_u_mul24(xs...);
  }
})
#else
template <typename T0, typename T1>
detail::builtin_enable_suint32_t<T0, T1> mul24(T0 x, T1 y) {
  if constexpr (detail::is_scalar_arithmetic_v<T0>) {
    return x * y;
  } else {
    detail::simplify_if_swizzle_t<T0> res;
    for (int i = 0; i < detail::num_elements<T0>::value; ++i)
      res[i] = x[i] * y[i];
    return res;
  }
}
#endif

namespace detail {
// clang-format off
template <typename T>
using upsample_ret_type_t = change_elements_t<
    typename map_type<get_elem_type_t<T>,
                      int8_t, /*->*/ int16_t, uint8_t, /*->*/ uint16_t,
                      int16_t, /*->*/ int32_t, uint16_t, /*->*/ uint32_t,
                      int32_t, /*->*/ int64_t, uint32_t, /*->*/ uint64_t>::type,
    T>;
// clang-format on

template <typename T0, typename T1>
inline constexpr bool enable_upsample_v =
    builtin_same_shape_v<T0, T1> &&
    check_type_in_v<get_elem_type_t<T0>, int8_t, uint8_t, int16_t, uint16_t,
                    int32_t, uint32_t> &&
    check_type_in_v<get_elem_type_t<T1>, uint8_t, uint16_t, uint32_t> &&
    sizeof(get_elem_type_t<T0>) == sizeof(get_elem_type_t<T1>);

template <typename T0, typename T1>
using enable_upsample_t =
    std::enable_if_t<enable_upsample_v<T0, T1>, upsample_ret_type_t<T0>>;
} // namespace detail

#if __SYCL_DEVICE_ONLY
DEVICE_IMPL_TEMPLATE(TWO_ARGS, mul24, enable_upsample_t, [](auto... xs) {
  if constexpr (std::is_same_v<int32_t, detail::get_elem_type_t<T0>>) {
    return __spirv_ocl_s_upsample(xs...);
  } else {
    return __spirv_ocl_u_upsample(xs...);
  }
})
#else
template <typename T0, typename T1>
std::enable_if_t<detail::enable_upsample_v<T0, T1>,
                 detail::upsample_ret_type_t<T0>>
upsample(T0 x, T1 y) {
  using namespace detail;
  if constexpr (is_vec_or_swizzle_v<T0> || is_marray_v<T0>) {
    return builtin_delegate_to_scalar(
        [](auto... xs) { return upsample(xs...); }, x, y);
  } else {
    return upsample_ret_type_t<T0>{x} << (sizeof(T0) * 8) | y;
  }
}
#endif
} // namespace _V1
} // namespace sycl
