//==------------------- math_functions.hpp ---------------------------------==//
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
template <typename... Ts> struct last_int_rest_same {
  static constexpr bool value = []() constexpr {
    constexpr auto N = sizeof...(Ts);
    using first_type = typename first_type<Ts...>::type;
    if (!builtin_same_shape_v<first_type>)
      return false;
    int i = 0;
    using int_type =
        std::conditional_t<is_vec_or_swizzle_v<first_type>, int32_t, int>;
    return (
        (... &&
         (++i == N
              ? /* last */ builtin_same_shape_v<Ts> // filter out "bad" types,
                                                    // e.g. multi-ptr
                    && std::is_same_v<get_elem_type_t<Ts>, int_type>
              : /* not last  */ builtin_same_or_swizzle_v<first_type, Ts>)));
  }();
};
template <typename... Ts> struct last_intptr_rest_same {
  static constexpr bool value = []() constexpr {
    constexpr auto N = sizeof...(Ts);
    using first_type = typename first_type<Ts...>::type;
    if (!builtin_same_shape_v<first_type>)
      return false;
    int i = 0;
    using int_type =
        std::conditional_t<is_vec_or_swizzle_v<first_type>, int32_t, int>;
    return (
        (... &&
         (++i == N
              ? /* last */ (
                    std::is_pointer_v<Ts> ||
                    (is_multi_ptr_v<Ts> && has_writeable_addr_space_v<Ts>)) &&
                    builtin_same_shape_v<first_type, get_elem_type_t<Ts>> &&
                    !is_swizzle_v<get_elem_type_t<Ts>> &&
                    std::is_same_v<get_elem_type_t<get_elem_type_t<Ts>>,
                                   int_type>
              : /* not last  */ builtin_same_or_swizzle_v<first_type, Ts>)));
  }();
};
} // namespace detail
BUILTIN_CREATE_ENABLER(builtin_enable_math, default_ret_type, fp_elem_type,
                       non_scalar_only, same_elem_type)
BUILTIN_CREATE_ENABLER(builtin_enable_math_allow_scalar, default_ret_type,
                       fp_elem_type, any_shape, same_elem_type)
BUILTIN_CREATE_ENABLER(builtin_enable_last_int, default_ret_type, fp_elem_type,
                       non_scalar_only, last_int_rest_same)
BUILTIN_CREATE_ENABLER(builtin_enable_last_intptr_scalar, default_ret_type,
                       fp_elem_type, scalar_only, last_intptr_rest_same)
BUILTIN_CREATE_ENABLER(builtin_enable_last_intptr_non_scalar, default_ret_type,
                       fp_elem_type, non_scalar_only, last_intptr_rest_same)

namespace detail {
// FIXME: get rid of these.
template <typename... Ts>
inline constexpr bool builtin_enable_math_allow_scalar_v =
    builtin_same_or_swizzle_v<Ts...> &&
    check_type_in_v<get_elem_type_t<typename first_type<Ts...>::type>, float,
                    double, half>;
} // namespace detail

#ifdef __SYCL_DEVICE_ONLY__
// Common between generic case and fast math optimized path. Note that vector
// case is template with a single implementation between all three types, so we
// have to introduce this VEC_IMPL parameter to be able to use native version
// for floatN.
#define BUILTIN_GENF_DEVICE_COMMON(NUM_ARGS, NAME, VEC_IMPL)                   \
  inline double NAME(NUM_ARGS##_TYPE_ARG(double)) {                            \
    return __spirv_ocl_##NAME(NUM_ARGS##_ARG);                                 \
  }                                                                            \
  inline half NAME(NUM_ARGS##_TYPE_ARG(half)) {                                \
    return __spirv_ocl_##NAME(NUM_ARGS##_CONVERTED_ARG);                       \
  }                                                                            \
  DEVICE_IMPL_TEMPLATE(NUM_ARGS, NAME, builtin_enable_math_t, VEC_IMPL)

#define BUILTIN_GENF(NUM_ARGS, NAME)                                           \
  inline float NAME(NUM_ARGS##_TYPE_ARG(float)) {                              \
    return __spirv_ocl_##NAME(NUM_ARGS##_ARG);                                 \
  }                                                                            \
  BUILTIN_GENF_DEVICE_COMMON(NUM_ARGS, NAME, __spirv_ocl_##NAME)

#define BUILTIN_GENF_NATIVE_OPT(NUM_ARGS, NAME)                                \
  namespace detail {                                                           \
  template <typename T>                                                        \
  decltype(auto) maybe_fmf_##NAME(NUM_ARGS##_TYPE_ARG(T)) {                    \
    if constexpr (use_fast_math_v<T>) {                                        \
      return __spirv_ocl_native_##NAME(NUM_ARGS##_ARG);                        \
    } else {                                                                   \
      return __spirv_ocl_##NAME(NUM_ARGS##_ARG);                               \
    }                                                                          \
  }                                                                            \
  }                                                                            \
  inline float NAME(NUM_ARGS##_TYPE_ARG(float)) {                              \
    return detail::maybe_fmf_##NAME(NUM_ARGS##_CONVERTED_ARG);                 \
  }                                                                            \
  BUILTIN_GENF_DEVICE_COMMON(NUM_ARGS, NAME, detail::maybe_fmf_##NAME)

#else
#define BUILTIN_GENF(NUM_ARGS, NAME)                                           \
  FOR_EACH2(HOST_IMPL_SCALAR, NUM_ARGS, NAME, FP_TYPES)                        \
  HOST_IMPL_TEMPLATE(NUM_ARGS, NAME, builtin_enable_math_t, math,              \
                     default_ret_type)

// Optimization only affects device code.
#define BUILTIN_GENF_NATIVE_OPT(NUM_ARGS, NAME) BUILTIN_GENF(NUM_ARGS, NAME)
#endif

BUILTIN_GENF(ONE_ARG, acos)
BUILTIN_GENF(ONE_ARG, acosh)
BUILTIN_GENF(ONE_ARG, acospi)
BUILTIN_GENF(ONE_ARG, asin)
BUILTIN_GENF(ONE_ARG, asinh)
BUILTIN_GENF(ONE_ARG, asinpi)
BUILTIN_GENF(ONE_ARG, atan)
BUILTIN_GENF(ONE_ARG, atanh)
BUILTIN_GENF(ONE_ARG, atanpi)
BUILTIN_GENF(ONE_ARG, cbrt)
BUILTIN_GENF(ONE_ARG, ceil)
BUILTIN_GENF(ONE_ARG, cosh)
BUILTIN_GENF(ONE_ARG, cospi)
BUILTIN_GENF(ONE_ARG, erf)
BUILTIN_GENF(ONE_ARG, erfc)
BUILTIN_GENF(ONE_ARG, expm1)
BUILTIN_GENF(ONE_ARG, fabs)
BUILTIN_GENF(ONE_ARG, floor)
BUILTIN_GENF(ONE_ARG, lgamma)
BUILTIN_GENF(ONE_ARG, log1p)
BUILTIN_GENF(ONE_ARG, logb)
BUILTIN_GENF(ONE_ARG, rint)
BUILTIN_GENF(ONE_ARG, round)
BUILTIN_GENF(ONE_ARG, sinh)
BUILTIN_GENF(ONE_ARG, sinpi)
BUILTIN_GENF(ONE_ARG, tanh)
BUILTIN_GENF(ONE_ARG, tanpi)
BUILTIN_GENF(ONE_ARG, tgamma)
BUILTIN_GENF(ONE_ARG, trunc)
BUILTIN_GENF(TWO_ARGS, atan2)
BUILTIN_GENF(TWO_ARGS, atan2pi)
BUILTIN_GENF(TWO_ARGS, copysign)
BUILTIN_GENF(TWO_ARGS, fdim)
BUILTIN_GENF(TWO_ARGS, fmod)
BUILTIN_GENF(TWO_ARGS, hypot)
BUILTIN_GENF(TWO_ARGS, maxmag)
BUILTIN_GENF(TWO_ARGS, minmag)
BUILTIN_GENF(TWO_ARGS, nextafter)
BUILTIN_GENF(TWO_ARGS, pow)
BUILTIN_GENF(TWO_ARGS, remainder)
BUILTIN_GENF(THREE_ARGS, fma)
BUILTIN_GENF(THREE_ARGS, mad)

#define BUILTIN_GENF_SCALAR_2ND(NAME)                                          \
  BUILTIN_GENF(TWO_ARGS, NAME)                                                 \
  template <typename T>                                                        \
  detail::builtin_enable_math_t<T> NAME(T x, detail::get_elem_type_t<T> y) {   \
    return NAME(detail::simplify_if_swizzle_t<T>{x},                           \
                detail::simplify_if_swizzle_t<T>{y});                          \
  }

BUILTIN_GENF_SCALAR_2ND(fmax)
BUILTIN_GENF_SCALAR_2ND(fmin)

#undef BUILTIN_GENF_SCALAR_2ND

BUILTIN_GENF_NATIVE_OPT(ONE_ARG, cos)
BUILTIN_GENF_NATIVE_OPT(ONE_ARG, exp)
BUILTIN_GENF_NATIVE_OPT(ONE_ARG, exp10)
BUILTIN_GENF_NATIVE_OPT(ONE_ARG, exp2)
BUILTIN_GENF_NATIVE_OPT(ONE_ARG, log)
BUILTIN_GENF_NATIVE_OPT(ONE_ARG, log10)
BUILTIN_GENF_NATIVE_OPT(ONE_ARG, log2)
BUILTIN_GENF_NATIVE_OPT(ONE_ARG, rsqrt)
BUILTIN_GENF_NATIVE_OPT(ONE_ARG, sin)
BUILTIN_GENF_NATIVE_OPT(ONE_ARG, sqrt)
BUILTIN_GENF_NATIVE_OPT(ONE_ARG, tan)
BUILTIN_GENF_NATIVE_OPT(TWO_ARGS, powr)

#undef BUILTIN_GENF_NATIVE_OPT
#undef BUILTIN_GENF
#undef BUILTIN_GENF_DEVICE_COMMON

namespace detail {
template <typename T0, typename T1>
inline constexpr bool builtin_ptr_check_v =
    (std::is_pointer_v<T1> ||
     (is_multi_ptr_v<T1> && has_writeable_addr_space_v<T1>)) &&
    is_valid_elem_type_v<T1, simplify_if_swizzle_t<T0>>;

template <typename T0, typename T1>
inline constexpr bool builtin_enable_ptr_v =
    builtin_enable_math_allow_scalar_v<T0> && builtin_ptr_check_v<T0, T1>;

template <typename T0, typename T1>
using builtin_enable_ptr_scalar_t =
    std::enable_if_t<builtin_enable_ptr_v<T0, T1> && is_scalar_arithmetic_v<T0>,
                     T0>;

template <typename T0, typename T1>
using builtin_enable_ptr_non_scalar_t =
    std::enable_if_t<builtin_enable_ptr_v<T0, T1> &&
                         !is_scalar_arithmetic_v<T0>,
                     simplify_if_swizzle_t<T0>>;

template <typename FuncTy, typename PtrTy, typename... Ts>
auto builtin_delegate_ptr_impl(FuncTy F, PtrTy p, Ts... xs) {
  using T0 = typename first_type<Ts...>::type;
  // Simplify just incase, although most callers seem to do that on their own.
  simplify_if_swizzle_t<T0> r{};

  // TODO: Optimize for sizes. Make not to violate ANSI-aliasing rules for the
  // pointer argument.
  auto p0 = [&]() {
    if constexpr (is_multi_ptr_v<PtrTy>)
      return address_space_cast<PtrTy::address_space,
                                get_multi_ptr_decoration_v<PtrTy>>(&(*p)[0]);
    else
      return &(*p)[0];
  }();

  constexpr auto N = T0::size();
  if constexpr (N <= 16)
    loop<N>([&](auto i) { r[i] = F(xs[i]..., p0 + i); });
  else
    for (size_t i = 0; i < N; ++i)
      r[i] = F(xs[i]..., p0 + i);
  return r;
}
} // namespace detail

#define LAST_PTR_SCALAR(NUM_ARGS, NAME, SCALAR_ENABLER, TYPE)                  \
  template <typename PtrTy>                                                    \
  detail::SCALAR_ENABLER<TYPE, PtrTy> NAME(                                    \
      SYCL_CONCAT(LESS_ONE(NUM_ARGS), _TYPE_ARG)(TYPE), PtrTy p) {             \
    return detail::NAME##_impl(SYCL_CONCAT(LESS_ONE(NUM_ARGS), _ARG), p);      \
  }
#define BUILTIN_LAST_PTR_COMMON(NUM_ARGS, NAME, SCALAR_ENABLER,                \
                                NON_SCALAR_ENABLER)                            \
  FOR_EACH3(LAST_PTR_SCALAR, NUM_ARGS, NAME, SCALAR_ENABLER, FP_TYPES)         \
  template <SYCL_CONCAT(LESS_ONE(NUM_ARGS), _TYPENAME_TYPE), typename PtrTy>   \
  detail::NON_SCALAR_ENABLER<SYCL_CONCAT(LESS_ONE(NUM_ARGS), _TEMPLATE_TYPE),  \
                             PtrTy>                                            \
  NAME(SYCL_CONCAT(LESS_ONE(NUM_ARGS), _TEMPLATE_TYPE_ARG), PtrTy p) {         \
    return detail::NAME##_impl(SYCL_CONCAT(LESS_ONE(NUM_ARGS), _ARG), p);      \
  }

#if __SYCL_DEVICE_ONLY__
#define BUILTIN_LAST_PTR(NUM_ARGS, NAME, SCALAR_ENABLER, NON_SCALAR_ENABLER)   \
  namespace detail {                                                           \
  template <NUM_ARGS##_TYPENAME_TYPE>                                          \
  auto NAME##_impl(NUM_ARGS##_TEMPLATE_TYPE_ARG_REF) {                         \
    if constexpr (is_marray_v<T0>) {                                           \
      return builtin_delegate_ptr_impl(                                        \
          [](auto... xs) { return NAME##_impl(xs...); },                       \
          NUM_ARGS##_ARG_ROTATED);                                             \
    } else {                                                                   \
      return __spirv_ocl_##NAME(NUM_ARGS##_CONVERTED_ARG);                     \
    }                                                                          \
  }                                                                            \
  } /* namespace detail */                                                     \
  BUILTIN_LAST_PTR_COMMON(NUM_ARGS, NAME, SCALAR_ENABLER, NON_SCALAR_ENABLER)
#else
#define BUILTIN_LAST_PTR(NUM_ARGS, NAME, SCALAR_ENABLER, NON_SCALAR_ENABLER)   \
  BUILTIN_LAST_PTR_COMMON(NUM_ARGS, NAME, SCALAR_ENABLER, NON_SCALAR_ENABLER)
#endif

namespace detail {
template <typename T>
using builtin_last_raw_intptr_t =
    // FIXME: Should we allow marray here, or limit just to vec/swizzle/ scalar?
    //        If not, "enabler" has to be changed as well.
    change_elements_t<std::conditional_t<is_marray_v<T>, int, int32_t>,
                      simplify_if_swizzle_t<T>> *;
}

#ifdef __SYCL_DEVICE_ONLY__
#define BUILTIN_LAST_INTPTR(NUM_ARGS, NAME)                                    \
  BUILTIN_LAST_PTR(NUM_ARGS, NAME, builtin_enable_last_intptr_scalar_t,        \
                   builtin_enable_last_intptr_non_scalar_t)
#else
#define LAST_INT_PTR_DECLARE_SCALAR(NUM_ARGS, NAME, TYPE)                      \
  __SYCL_EXPORT TYPE NAME##_impl(SYCL_CONCAT(LESS_ONE(NUM_ARGS), _TYPE)(TYPE), \
                                 int *);
#define BUILTIN_LAST_INTPTR(NUM_ARGS, NAME)                                    \
  namespace detail {                                                           \
  FOR_EACH2(LAST_INT_PTR_DECLARE_SCALAR, NUM_ARGS, NAME, FP_TYPES)             \
  template <SYCL_CONCAT(LESS_ONE(NUM_ARGS), _TYPENAME_TYPE), typename PtrTy>   \
  auto NAME##_impl(SYCL_CONCAT(LESS_ONE(NUM_ARGS), _TEMPLATE_TYPE_ARG_REF),    \
                   PtrTy p) {                                                  \
    if constexpr (is_multi_ptr_v<PtrTy>) {                                     \
      /* TODO: Can't really create multi_ptr on host... */                     \
      return NAME##_impl(SYCL_CONCAT(LESS_ONE(NUM_ARGS), _ARG), p.get_raw());  \
    } else {                                                                   \
      return builtin_delegate_ptr_impl(                                        \
          [](auto... xs) { return NAME##_impl(xs...); }, p,                    \
          SYCL_CONCAT(LESS_ONE(NUM_ARGS), _SIMPLIFIED_ARG));                   \
    }                                                                          \
  }                                                                            \
  } /* namespace detail */                                                     \
  BUILTIN_LAST_PTR(NUM_ARGS, NAME, builtin_enable_last_intptr_scalar_t,        \
                   builtin_enable_last_intptr_non_scalar_t)
#endif

BUILTIN_LAST_INTPTR(TWO_ARGS, frexp)
BUILTIN_LAST_INTPTR(TWO_ARGS, lgamma_r)
BUILTIN_LAST_INTPTR(THREE_ARGS, remquo)

#undef BUILTIN_LAST_INTPTR
#undef LAST_INT_PTR_DECLARE_SCALAR

#ifndef __SYCL_DEVICE_ONLY__
namespace detail {
template <typename T0, typename T1> auto fract_impl(T0 &x, T1 &y) {
  auto flr = floor(simplify_if_swizzle_t<T0>{x});
  *y = flr;
  return fmin(x - flr, nextafter(simplify_if_swizzle_t<T0>{1.0},
                                 simplify_if_swizzle_t<T0>{0.0}));
}
} // namespace detail
#endif
BUILTIN_LAST_PTR(TWO_ARGS, fract, builtin_enable_ptr_scalar_t,
                 builtin_enable_ptr_non_scalar_t)

#ifndef __SYCL_DEVICE_ONLY__
namespace detail {
__SYCL_EXPORT float modf_impl(float, float *);
__SYCL_EXPORT double modf_impl(double, double *);
__SYCL_EXPORT half modf_impl(half, half *);
template <typename T0, typename T1> auto modf_impl(T0 &x, T1 &&y) {
  if constexpr (is_multi_ptr_v<std::remove_reference_t<T1>>) {
    // TODO: Spec needs to be clarified, multi_ptr shouldn't be possible on
    // host.
    return modf_impl(x, y.get_raw());
  } else {
    return builtin_delegate_ptr_impl(
        [](auto x, auto y) { return modf_impl(x, y); }, y,
        simplify_if_swizzle_t<T0>{x});
  }
}
} // namespace detail
#endif
BUILTIN_LAST_PTR(TWO_ARGS, modf, builtin_enable_ptr_scalar_t,
                 builtin_enable_ptr_non_scalar_t)

#undef BUILTIN_LAST_PTR

#ifdef __SYCL_DEVICE_ONLY__
#define BUILTIN_MATH_LAST_INT(NAME)                                            \
  float NAME(float x, int y) { return __spirv_ocl_##NAME(x, y); }              \
  double NAME(double x, int y) { return __spirv_ocl_##NAME(x, y); }            \
  half NAME(half x, int y) {                                                   \
    return __spirv_ocl_##NAME(detail::builtins::convert_arg(x), y);            \
  }                                                                            \
  DEVICE_IMPL_TEMPLATE(TWO_ARGS, NAME, builtin_enable_last_int_t,              \
                       __spirv_ocl_##NAME)
#else
#define SCALAR_EXTERN_LAST_INT(NAME, TYPE)                                     \
  inline TYPE NAME(TYPE x, int y) {                                            \
    extern SYCL_BUILTIN_EXPORT TYPE __##NAME##_impl(TYPE, int);                \
    return __##NAME##_impl(x, y);                                              \
  }
#define BUILTIN_MATH_LAST_INT(NAME)                                            \
  FOR_EACH1(SCALAR_EXTERN_LAST_INT, NAME, FP_TYPES)                            \
  HOST_IMPL_TEMPLATE(TWO_ARGS, NAME, builtin_enable_last_int_t, math,          \
                     default_ret_type)
#endif

BUILTIN_MATH_LAST_INT(pown)
BUILTIN_MATH_LAST_INT(rootn)
BUILTIN_MATH_LAST_INT(ldexp)
template <typename T> detail::builtin_enable_math_t<T> ldexp(T x, int y) {
  return ldexp(
      detail::simplify_if_swizzle_t<T>{x},
      detail::change_elements_t<int, detail::simplify_if_swizzle_t<T>>{y});
}

#undef BUILTIN_MATH_LAST_INT
#undef SCALAR_EXTERN_LAST_INT

namespace detail {
#ifdef __SYCL_DEVICE_ONLY__
template <typename T0, typename T1> auto sincos_impl(T0 &x, T1 &&y) {
  if constexpr (is_marray_v<T0>) {
    return builtin_delegate_ptr_impl(
        [](auto... xs) { return sincos_impl(xs...); }, y, x);
  } else {
    using detail::builtins::convert_arg;
    if constexpr (use_fast_math_v<T0>) {
      // This is a performance optimization to ensure that sincos isn't slower
      // than a pair of sin/cos executed separately. Theoretically, calling
      // non-native sincos might be faster than calling native::sin plus
      // native::cos separately and we'd need some kind of cost model to make
      // the right decision (and move this entirely to the JIT/AOT compilers).
      // However, in practice, this simpler solution seems to work just fine and
      // matches how sin/cos above are optimized for the fast math path.
      *y = __spirv_ocl_native_cos(convert_arg(x));
      return __spirv_ocl_native_sin(convert_arg(x));
    } else {
      return __spirv_ocl_sincos(convert_arg(x), convert_arg(y));
    }
  }
}
#else
__SYCL_EXPORT float sincos_impl(float, float *);
__SYCL_EXPORT double sincos_impl(double, double *);
__SYCL_EXPORT half sincos_impl(half, half *);
template <typename T0, typename T1> auto sincos_impl(T0 &x, T1 &&y) {
  if constexpr (is_multi_ptr_v<std::remove_reference_t<T1>>) {
    // TODO: Spec needs to be clarified, multi_ptr shouldn't be possible on
    // host.
    return sincos_impl(x, y.get_raw());
  } else {
    return builtin_delegate_ptr_impl(
        [](auto... xs) { return sincos_impl(xs...); }, y,
        simplify_if_swizzle_t<T0>{x});
  }
}
#endif
} // namespace detail
BUILTIN_LAST_PTR_COMMON(TWO_ARGS, sincos, builtin_enable_ptr_scalar_t,
                        builtin_enable_ptr_non_scalar_t)

#undef BUILTIN_LAST_PTR_COMMON
#undef LAST_PTR_SCALAR

namespace detail {
template <typename T>
struct ilogb_ret_traits
    : change_elements<std::conditional_t<is_vec_or_swizzle_v<T>, int32_t, int>,
                      T> {};
template <typename T>
using builtin_enable_ilogb_t =
    std::enable_if_t<builtin_enable_math_allow_scalar_v<T>,
                     typename ilogb_ret_traits<T>::type>;
} // namespace detail

#ifdef __SYCL_DEVICE_ONLY__
inline int ilogb(float x) { return __spirv_ocl_ilogb(x); }
inline int ilogb(double x) { return __spirv_ocl_ilogb(x); }
inline int ilogb(half x) {
  return __spirv_ocl_ilogb(detail::builtins::convert_arg(x));
}
DEVICE_IMPL_TEMPLATE(ONE_ARG, ilogb, builtin_enable_ilogb_t, __spirv_ocl_ilogb)
#else
inline int ilogb(float x) {
  extern SYCL_BUILTIN_EXPORT int __ilogb_impl(float);
  return __ilogb_impl(x);
}
inline int ilogb(double x) {
  extern SYCL_BUILTIN_EXPORT int __ilogb_impl(double);
  return __ilogb_impl(x);
}
inline int ilogb(half x) {
  extern SYCL_BUILTIN_EXPORT int __ilogb_impl(half);
  return __ilogb_impl(x);
}
HOST_IMPL_TEMPLATE(ONE_ARG, ilogb, builtin_enable_ilogb_t, math,
                   ilogb_ret_traits)
#endif

// nan implementation, as per
// https://github.com/KhronosGroup/SYCL-Docs/pull/519.
namespace detail {
template <typename T>
// clang-format off
using nan_elem_result_type = change_elements_t<
    typename map_type<get_elem_type_t<T>,
                      uint32_t, /*->*/ float,
                      uint64_t, /*->*/ double,
                      uint16_t, /*->*/ half>::type,
    T>;
// clang-format on

template <typename T>
using builtin_enable_nan_t = std::enable_if_t<
    (((is_vec_or_swizzle_v<T> || is_marray_v<T>)) &&
     check_type_in_v<get_elem_type_t<T>, uint32_t, uint64_t, uint16_t>),
    nan_elem_result_type<T>>;
} // namespace detail

#ifdef __SYCL_DEVICE_ONLY__
inline float nan(uint32_t x) {
  return __spirv_ocl_nan(detail::builtins::convert_arg(x));
}
inline double nan(uint64_t x) {
  return __spirv_ocl_nan(detail::builtins::convert_arg(x));
}
inline half nan(uint16_t x) {
  return __spirv_ocl_nan(detail::builtins::convert_arg(x));
}
DEVICE_IMPL_TEMPLATE(ONE_ARG, nan, builtin_enable_nan_t, __spirv_ocl_nan)
#else
inline float nan(uint32_t) { return std::numeric_limits<float>::quiet_NaN(); }
inline double nan(uint64_t) { return std::numeric_limits<float>::quiet_NaN(); }
// NOTE: half_type.hpp provides partial specialization for std::numeric_limits.
inline half nan(uint16_t) { return std::numeric_limits<half>::quiet_NaN(); }
template <typename T> detail::builtin_enable_nan_t<T> nan(T x) {
  return detail::builtin_delegate_to_scalar([](auto x) { return nan(x); }, x);
}
#endif

} // namespace _V1
} // namespace sycl
