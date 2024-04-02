//==--- builtins_legacy_marray_vec.hpp - Old SYCL built-in nd definitions --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
#error "Legacy builtins must not be used in preview."
#endif

#pragma once

#include <sycl/access/access.hpp>              // for address_space, decorated
#include <sycl/aliases.hpp>                    // for half
#include <sycl/builtins_legacy_scalar.hpp>     // for scalar builtin variants
#include <sycl/builtins_utils_vec.hpp>         // for to_vec, to_marray...
#include <sycl/detail/boolean.hpp>             // for Boolean
#include <sycl/detail/builtins.hpp>            // for __invoke_select, __in...
#include <sycl/detail/defines_elementary.hpp>  // for __SYCL_ALWAYS_INLINE
#include <sycl/detail/generic_type_traits.hpp> // for is_svgenfloat, is_sge...
#include <sycl/detail/type_list.hpp>           // for is_contained, type_list
#include <sycl/detail/type_traits.hpp>         // for make_larger_t, marray...
#include <sycl/half_type.hpp>                  // for half, intel
#include <sycl/marray.hpp>                     // for marray
#include <sycl/multi_ptr.hpp>                  // for address_space_cast
#include <sycl/types.hpp>                      // for vec

namespace sycl {
inline namespace _V1 {

#ifdef __SYCL_DEVICE_ONLY__
#define __sycl_std
#else
namespace __sycl_std = __host_std;
#endif

#ifdef __FAST_MATH__
#define __FAST_MATH_GENFLOAT(T)                                                \
  (detail::is_svgenfloatd_v<T> || detail::is_svgenfloath_v<T>)
#define __FAST_MATH_SGENFLOAT(T)                                               \
  (std::is_same_v<T, double> || std::is_same_v<T, half>)
#else
#define __FAST_MATH_GENFLOAT(T) (detail::is_svgenfloat_v<T>)
#define __FAST_MATH_SGENFLOAT(T) (detail::is_sgenfloat_v<T>)
#endif

/* ------------------ 4.13.3 Math functions. ---------------------------------*/
// These macros for marray math function implementations use vectorizations of
// size two as a simple general optimization. A more complex implementation
// using larger vectorizations for large marray sizes is possible; however more
// testing is required in order to ascertain the performance implications for
// all backends.
#define __SYCL_MATH_FUNCTION_OVERLOAD_IMPL(NAME)                               \
  marray<T, N> res;                                                            \
  for (size_t i = 0; i < N / 2; i++) {                                         \
    vec<T, 2> partial_res =                                                    \
        __sycl_std::__invoke_##NAME<vec<T, 2>>(detail::to_vec2(x, i * 2));     \
    std::memcpy(&res[i * 2], &partial_res, sizeof(vec<T, 2>));                 \
  }                                                                            \
  if (N % 2) {                                                                 \
    res[N - 1] = __sycl_std::__invoke_##NAME<T>(x[N - 1]);                     \
  }                                                                            \
  return res;

#define __SYCL_MATH_FUNCTION_OVERLOAD(NAME)                                    \
  template <typename T, size_t N>                                              \
  inline __SYCL_ALWAYS_INLINE                                                  \
      std::enable_if_t<detail::is_sgenfloat_v<T>, marray<T, N>>                \
      NAME(marray<T, N> x) {                                                   \
    __SYCL_MATH_FUNCTION_OVERLOAD_IMPL(NAME)                                   \
  }

__SYCL_MATH_FUNCTION_OVERLOAD(cospi)
__SYCL_MATH_FUNCTION_OVERLOAD(sinpi)
__SYCL_MATH_FUNCTION_OVERLOAD(tanpi)
__SYCL_MATH_FUNCTION_OVERLOAD(sinh)
__SYCL_MATH_FUNCTION_OVERLOAD(cosh)
__SYCL_MATH_FUNCTION_OVERLOAD(tanh)
__SYCL_MATH_FUNCTION_OVERLOAD(asin)
__SYCL_MATH_FUNCTION_OVERLOAD(acos)
__SYCL_MATH_FUNCTION_OVERLOAD(atan)
__SYCL_MATH_FUNCTION_OVERLOAD(asinpi)
__SYCL_MATH_FUNCTION_OVERLOAD(acospi)
__SYCL_MATH_FUNCTION_OVERLOAD(atanpi)
__SYCL_MATH_FUNCTION_OVERLOAD(asinh)
__SYCL_MATH_FUNCTION_OVERLOAD(acosh)
__SYCL_MATH_FUNCTION_OVERLOAD(atanh)
__SYCL_MATH_FUNCTION_OVERLOAD(cbrt)
__SYCL_MATH_FUNCTION_OVERLOAD(ceil)
__SYCL_MATH_FUNCTION_OVERLOAD(floor)
__SYCL_MATH_FUNCTION_OVERLOAD(erfc)
__SYCL_MATH_FUNCTION_OVERLOAD(erf)
__SYCL_MATH_FUNCTION_OVERLOAD(expm1)
__SYCL_MATH_FUNCTION_OVERLOAD(tgamma)
__SYCL_MATH_FUNCTION_OVERLOAD(lgamma)
__SYCL_MATH_FUNCTION_OVERLOAD(log1p)
__SYCL_MATH_FUNCTION_OVERLOAD(logb)
__SYCL_MATH_FUNCTION_OVERLOAD(rint)
__SYCL_MATH_FUNCTION_OVERLOAD(round)
__SYCL_MATH_FUNCTION_OVERLOAD(trunc)
__SYCL_MATH_FUNCTION_OVERLOAD(fabs)

#undef __SYCL_MATH_FUNCTION_OVERLOAD

// __SYCL_MATH_FUNCTION_OVERLOAD_FM cases are replaced by corresponding native
// implementations when the -ffast-math flag is used with float.
#define __SYCL_MATH_FUNCTION_OVERLOAD_FM(NAME)                                 \
  template <typename T, size_t N>                                              \
  inline __SYCL_ALWAYS_INLINE                                                  \
      std::enable_if_t<__FAST_MATH_SGENFLOAT(T), marray<T, N>>                 \
      NAME(marray<T, N> x) {                                                   \
    __SYCL_MATH_FUNCTION_OVERLOAD_IMPL(NAME)                                   \
  }

__SYCL_MATH_FUNCTION_OVERLOAD_FM(sin)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(cos)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(tan)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(exp)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(exp2)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(exp10)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(log)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(log2)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(log10)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(sqrt)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(rsqrt)

#undef __SYCL_MATH_FUNCTION_OVERLOAD_FM
#undef __SYCL_MATH_FUNCTION_OVERLOAD_IMPL

template <typename T, size_t N>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<detail::is_sgenfloat_v<T>, marray<int, N>>
    ilogb(marray<T, N> x) {
  marray<int, N> res;
  for (size_t i = 0; i < N / 2; i++) {
    vec<int, 2> partial_res =
        __sycl_std::__invoke_ilogb<vec<int, 2>>(detail::to_vec2(x, i * 2));
    std::memcpy(&res[i * 2], &partial_res, sizeof(vec<int, 2>));
  }
  if (N % 2) {
    res[N - 1] = __sycl_std::__invoke_ilogb<int>(x[N - 1]);
  }
  return res;
}

#define __SYCL_MATH_FUNCTION_2_OVERLOAD_IMPL(NAME)                             \
  marray<T, N> res;                                                            \
  for (size_t i = 0; i < N / 2; i++) {                                         \
    auto partial_res = __sycl_std::__invoke_##NAME<vec<T, 2>>(                 \
        detail::to_vec2(x, i * 2), detail::to_vec2(y, i * 2));                 \
    std::memcpy(&res[i * 2], &partial_res, sizeof(vec<T, 2>));                 \
  }                                                                            \
  if (N % 2) {                                                                 \
    res[N - 1] = __sycl_std::__invoke_##NAME<T>(x[N - 1], y[N - 1]);           \
  }                                                                            \
  return res;

#define __SYCL_MATH_FUNCTION_2_OVERLOAD(NAME)                                  \
  template <typename T, size_t N>                                              \
  inline __SYCL_ALWAYS_INLINE                                                  \
      std::enable_if_t<detail::is_sgenfloat_v<T>, marray<T, N>>                \
      NAME(marray<T, N> x, marray<T, N> y) {                                   \
    __SYCL_MATH_FUNCTION_2_OVERLOAD_IMPL(NAME)                                 \
  }

__SYCL_MATH_FUNCTION_2_OVERLOAD(atan2)
__SYCL_MATH_FUNCTION_2_OVERLOAD(atan2pi)
__SYCL_MATH_FUNCTION_2_OVERLOAD(copysign)
__SYCL_MATH_FUNCTION_2_OVERLOAD(fdim)
__SYCL_MATH_FUNCTION_2_OVERLOAD(fmin)
__SYCL_MATH_FUNCTION_2_OVERLOAD(fmax)
__SYCL_MATH_FUNCTION_2_OVERLOAD(fmod)
__SYCL_MATH_FUNCTION_2_OVERLOAD(hypot)
__SYCL_MATH_FUNCTION_2_OVERLOAD(maxmag)
__SYCL_MATH_FUNCTION_2_OVERLOAD(minmag)
__SYCL_MATH_FUNCTION_2_OVERLOAD(nextafter)
__SYCL_MATH_FUNCTION_2_OVERLOAD(pow)
__SYCL_MATH_FUNCTION_2_OVERLOAD(remainder)

#undef __SYCL_MATH_FUNCTION_2_OVERLOAD

template <typename T, size_t N>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<__FAST_MATH_SGENFLOAT(T), marray<T, N>>
    powr(marray<T, N> x, marray<T, N> y) {
  __SYCL_MATH_FUNCTION_2_OVERLOAD_IMPL(powr)
}

#undef __SYCL_MATH_FUNCTION_2_OVERLOAD_IMPL

#define __SYCL_MATH_FUNCTION_2_SGENFLOAT_Y_OVERLOAD(NAME)                      \
  template <typename T, size_t N>                                              \
  inline __SYCL_ALWAYS_INLINE                                                  \
      std::enable_if_t<detail::is_sgenfloat_v<T>, marray<T, N>>                \
      NAME(marray<T, N> x, T y) {                                              \
    marray<T, N> res;                                                          \
    sycl::vec<T, 2> y_vec{y, y};                                               \
    for (size_t i = 0; i < N / 2; i++) {                                       \
      auto partial_res = __sycl_std::__invoke_##NAME<vec<T, 2>>(               \
          detail::to_vec2(x, i * 2), y_vec);                                   \
      std::memcpy(&res[i * 2], &partial_res, sizeof(vec<T, 2>));               \
    }                                                                          \
    if (N % 2) {                                                               \
      res[N - 1] = __sycl_std::__invoke_##NAME<T>(x[N - 1], y_vec[0]);         \
    }                                                                          \
    return res;                                                                \
  }

__SYCL_MATH_FUNCTION_2_SGENFLOAT_Y_OVERLOAD(fmax)
// clang-format off
__SYCL_MATH_FUNCTION_2_SGENFLOAT_Y_OVERLOAD(fmin)

#undef __SYCL_MATH_FUNCTION_2_SGENFLOAT_Y_OVERLOAD

template <typename T, size_t N>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<detail::is_sgenfloat_v<T>, marray<T, N>>
    ldexp(marray<T, N> x, marray<int, N> k)  {
  // clang-format on
  marray<T, N> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = __sycl_std::__invoke_ldexp<T>(x[i], k[i]);
  }
  return res;
}

template <typename T, size_t N>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<detail::is_sgenfloat_v<T>, marray<T, N>>
    ldexp(marray<T, N> x, int k) {
  marray<T, N> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = __sycl_std::__invoke_ldexp<T>(x[i], k);
  }
  return res;
}

#define __SYCL_MATH_FUNCTION_2_GENINT_Y_OVERLOAD_IMPL(NAME)                    \
  marray<T, N> res;                                                            \
  for (size_t i = 0; i < N; i++) {                                             \
    res[i] = __sycl_std::__invoke_##NAME<T>(x[i], y[i]);                       \
  }                                                                            \
  return res;

template <typename T, size_t N>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<detail::is_sgenfloat_v<T>, marray<T, N>>
    pown(marray<T, N> x, marray<int, N> y) {
  __SYCL_MATH_FUNCTION_2_GENINT_Y_OVERLOAD_IMPL(pown)
}

template <typename T, size_t N>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<detail::is_sgenfloat_v<T>, marray<T, N>>
    rootn(marray<T, N> x, marray<int, N> y){
        __SYCL_MATH_FUNCTION_2_GENINT_Y_OVERLOAD_IMPL(rootn)}

#undef __SYCL_MATH_FUNCTION_2_GENINT_Y_OVERLOAD_IMPL

#define __SYCL_MATH_FUNCTION_2_INT_Y_OVERLOAD_IMPL(NAME)                       \
  marray<T, N> res;                                                            \
  for (size_t i = 0; i < N; i++) {                                             \
    res[i] = __sycl_std::__invoke_##NAME<T>(x[i], y);                          \
  }                                                                            \
  return res;

template <typename T, size_t N>
inline __SYCL_ALWAYS_INLINE std::enable_if_t<detail::is_sgenfloat_v<T>,
                                             marray<T, N>> pown(marray<T, N> x,
                                                                int y) {
  __SYCL_MATH_FUNCTION_2_INT_Y_OVERLOAD_IMPL(pown)
}

template <typename T, size_t N>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<detail::is_sgenfloat_v<T>, marray<T, N>>
    rootn(marray<T, N> x, int y) {
  __SYCL_MATH_FUNCTION_2_INT_Y_OVERLOAD_IMPL(rootn)
}

#undef __SYCL_MATH_FUNCTION_2_INT_Y_OVERLOAD_IMPL

#define __SYCL_MATH_FUNCTION_3_OVERLOAD(NAME)                                  \
  template <typename T, size_t N>                                              \
  inline __SYCL_ALWAYS_INLINE                                                  \
      std::enable_if_t<detail::is_sgenfloat_v<T>, marray<T, N>>                \
      NAME(marray<T, N> x, marray<T, N> y, marray<T, N> z) {                   \
    marray<T, N> res;                                                          \
    for (size_t i = 0; i < N / 2; i++) {                                       \
      auto partial_res = __sycl_std::__invoke_##NAME<vec<T, 2>>(               \
          detail::to_vec2(x, i * 2), detail::to_vec2(y, i * 2),                \
          detail::to_vec2(z, i * 2));                                          \
      std::memcpy(&res[i * 2], &partial_res, sizeof(vec<T, 2>));               \
    }                                                                          \
    if (N % 2) {                                                               \
      res[N - 1] =                                                             \
          __sycl_std::__invoke_##NAME<T>(x[N - 1], y[N - 1], z[N - 1]);        \
    }                                                                          \
    return res;                                                                \
  }

__SYCL_MATH_FUNCTION_3_OVERLOAD(mad)
__SYCL_MATH_FUNCTION_3_OVERLOAD(mix)
__SYCL_MATH_FUNCTION_3_OVERLOAD(fma)

#undef __SYCL_MATH_FUNCTION_3_OVERLOAD

// svgenfloat fmax (svgenfloat x, sgenfloat y)
template <typename T>
std::enable_if_t<detail::is_vgenfloat_v<T>, T>
fmax(T x, typename T::element_type y) {
  return __sycl_std::__invoke_fmax<T>(x, T(y));
}

// svgenfloat fmin (svgenfloat x, sgenfloat y)
template <typename T>
std::enable_if_t<detail::is_vgenfloat_v<T>, T>
fmin(T x, typename T::element_type y) {
  return __sycl_std::__invoke_fmin<T>(x, T(y));
}

// vgenfloat ldexp (vgenfloat x, int k)
template <typename T>
std::enable_if_t<detail::is_vgenfloat_v<T>, T> ldexp(T x, int k) {
  return __sycl_std::__invoke_ldexp<T>(x, vec<int, T::size()>(k));
}

// vgenfloat ldexp (vgenfloat x, genint k)
template <typename T, typename T2>
std::enable_if_t<detail::is_vgenfloat_v<T> && detail::is_intn_v<T2>, T>
ldexp(T x, T2 k) {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_ldexp<T>(x, k);
}

// other marray math functions

// TODO: can be optimized in the way marray math functions above are optimized
// (usage of vec<T, 2>)
#define __SYCL_MARRAY_MATH_FUNCTION_W_GENPTR_ARG_OVERLOAD_IMPL(NAME, ARGPTR,   \
                                                               ...)            \
  marray<T, N> res;                                                            \
  for (int j = 0; j < N; j++) {                                                \
    res[j] =                                                                   \
        NAME(__VA_ARGS__,                                                      \
             address_space_cast<AddressSpace, IsDecorated,                     \
                                detail::marray_element_t<T2>>(&(*ARGPTR)[j])); \
  }                                                                            \
  return res;

#define __SYCL_MARRAY_MATH_FUNCTION_BINOP_2ND_ARG_GENFLOATPTR_OVERLOAD(        \
    NAME, ARG1, ARG2, ...)                                                     \
  template <typename T, size_t N, typename T2,                                 \
            access::address_space AddressSpace, access::decorated IsDecorated> \
  std::enable_if_t<                                                            \
      detail::is_svgenfloat_v<T> &&                                            \
          detail::is_genfloatptr_marray_v<T2, AddressSpace, IsDecorated>,      \
      marray<T, N>>                                                            \
  NAME(marray<T, N> ARG1, multi_ptr<T2, AddressSpace, IsDecorated> ARG2) {     \
    __SYCL_MARRAY_MATH_FUNCTION_W_GENPTR_ARG_OVERLOAD_IMPL(NAME, ARG2,         \
                                                           __VA_ARGS__)        \
  }

__SYCL_MARRAY_MATH_FUNCTION_BINOP_2ND_ARG_GENFLOATPTR_OVERLOAD(fract, x, iptr,
                                                               x[j])
__SYCL_MARRAY_MATH_FUNCTION_BINOP_2ND_ARG_GENFLOATPTR_OVERLOAD(modf, x, iptr,
                                                               x[j])
__SYCL_MARRAY_MATH_FUNCTION_BINOP_2ND_ARG_GENFLOATPTR_OVERLOAD(sincos, x,
                                                               cosval, x[j])

#undef __SYCL_MARRAY_MATH_FUNCTION_BINOP_2ND_GENFLOATPTR_OVERLOAD

#define __SYCL_MARRAY_MATH_FUNCTION_BINOP_2ND_ARG_GENINTPTR_OVERLOAD(          \
    NAME, ARG1, ARG2, ...)                                                     \
  template <typename T, size_t N, typename T2,                                 \
            access::address_space AddressSpace, access::decorated IsDecorated> \
  std::enable_if_t<                                                            \
      detail::is_svgenfloat_v<T> &&                                            \
          detail::is_genintptr_marray_v<T2, AddressSpace, IsDecorated>,        \
      marray<T, N>>                                                            \
  NAME(marray<T, N> ARG1, multi_ptr<T2, AddressSpace, IsDecorated> ARG2) {     \
    __SYCL_MARRAY_MATH_FUNCTION_W_GENPTR_ARG_OVERLOAD_IMPL(NAME, ARG2,         \
                                                           __VA_ARGS__)        \
  }

__SYCL_MARRAY_MATH_FUNCTION_BINOP_2ND_ARG_GENINTPTR_OVERLOAD(frexp, x, exp,
                                                             x[j])
__SYCL_MARRAY_MATH_FUNCTION_BINOP_2ND_ARG_GENINTPTR_OVERLOAD(lgamma_r, x, signp,
                                                             x[j])

#undef __SYCL_MARRAY_MATH_FUNCTION_BINOP_2ND_GENINTPTR_OVERLOAD

#define __SYCL_MARRAY_MATH_FUNCTION_REMQUO_OVERLOAD(NAME, ...)                 \
  template <typename T, size_t N, typename T2,                                 \
            access::address_space AddressSpace, access::decorated IsDecorated> \
  std::enable_if_t<                                                            \
      detail::is_svgenfloat_v<T> &&                                            \
          detail::is_genintptr_marray_v<T2, AddressSpace, IsDecorated>,        \
      marray<T, N>>                                                            \
  NAME(marray<T, N> x, marray<T, N> y,                                         \
       multi_ptr<T2, AddressSpace, IsDecorated> quo) {                         \
    __SYCL_MARRAY_MATH_FUNCTION_W_GENPTR_ARG_OVERLOAD_IMPL(NAME, quo,          \
                                                           __VA_ARGS__)        \
  }

__SYCL_MARRAY_MATH_FUNCTION_REMQUO_OVERLOAD(remquo, x[j], y[j])

#undef __SYCL_MARRAY_MATH_FUNCTION_REMQUO_OVERLOAD

#undef __SYCL_MARRAY_MATH_FUNCTION_W_GENPTR_ARG_OVERLOAD_IMPL

template <typename T, size_t N>
std::enable_if_t<detail::is_nan_type_v<T> &&
                     detail::is_non_deprecated_nan_type_v<T>,
                 marray<detail::nan_return_t<T>, N>>
nan(marray<T, N> nancode) {
  marray<detail::nan_return_t<T>, N> res;
  for (int j = 0; j < N; j++) {
    res[j] = nan(nancode[j]);
  }
  return res;
}
template <typename T, size_t N>
__SYCL_DEPRECATED(
    "This is a deprecated argument type for SYCL nan built-in function.")
std::enable_if_t<detail::is_nan_type_v<T> &&
                     !detail::is_non_deprecated_nan_type_v<T>,
                 marray<detail::nan_return_t<T>, N>> nan(marray<T, N> nancode) {
  marray<detail::nan_return_t<T>, N> res;
  for (int j = 0; j < N; j++) {
    res[j] = nan(nancode[j]);
  }
  return res;
}

/* --------------- 4.13.5 Common functions. ---------------------------------*/
// vgenfloath clamp (vgenfloath x, half minval, half maxval)
// vgenfloatf clamp (vgenfloatf x, float minval, float maxval)
// vgenfloatd clamp (vgenfloatd x, double minval, double maxval)
template <typename T>
std::enable_if_t<detail::is_vgenfloat_v<T>, T>
clamp(T x, typename T::element_type minval, typename T::element_type maxval) {
  return __sycl_std::__invoke_fclamp<T>(x, T(minval), T(maxval));
}

// vgenfloatf max (vgenfloatf x, float y)
// vgenfloatd max (vgenfloatd x, double y)
// vgenfloath max (vgenfloath x, half y)
template <typename T>
std::enable_if_t<detail::is_vgenfloat_v<T>, T>(max)(
    T x, typename T::element_type y) {
  return __sycl_std::__invoke_fmax_common<T>(x, T(y));
}

// vgenfloatf min (vgenfloatf x, float y)
// vgenfloatd min (vgenfloatd x, double y)
// vgenfloath min (vgenfloath x, half y)
template <typename T>
std::enable_if_t<detail::is_vgenfloat_v<T>, T>(min)(
    T x, typename T::element_type y) {
  return __sycl_std::__invoke_fmin_common<T>(x, T(y));
}

// vgenfloatf mix (vgenfloatf x, vgenfloatf y, float a)
// vgenfloatd mix (vgenfloatd x, vgenfloatd y, double a)
// vgenfloatd mix (vgenfloath x, vgenfloath y, half a)
template <typename T>
std::enable_if_t<detail::is_vgenfloat_v<T>, T> mix(T x, T y,
                                                   typename T::element_type a) {
  return __sycl_std::__invoke_mix<T>(x, y, T(a));
}

// vgenfloatf step (float edge, vgenfloatf x)
// vgenfloatd step (double edge, vgenfloatd x)
// vgenfloatd step (half edge, vgenfloath x)
template <typename T>
std::enable_if_t<detail::is_vgenfloat_v<T>, T>
step(typename T::element_type edge, T x) {
  return __sycl_std::__invoke_step<T>(T(edge), x);
}

// vgenfloatf smoothstep (float edge0, float edge1, vgenfloatf x)
// vgenfloatd smoothstep (double edge0, double edge1, vgenfloatd x)
// vgenfloath smoothstep (half edge0, half edge1, vgenfloath x)
template <typename T>
std::enable_if_t<detail::is_vgenfloat_v<T>, T>
smoothstep(typename T::element_type edge0, typename T::element_type edge1,
           T x) {
  return __sycl_std::__invoke_smoothstep<T>(T(edge0), T(edge1), x);
}

// marray common functions

// TODO: can be optimized in the way math functions are optimized (usage of
// vec<T, 2>)
#define __SYCL_MARRAY_COMMON_FUNCTION_OVERLOAD_IMPL(NAME, ...)                 \
  T res;                                                                       \
  for (int i = 0; i < T::size(); i++) {                                        \
    res[i] = NAME(__VA_ARGS__);                                                \
  }                                                                            \
  return res;

#define __SYCL_MARRAY_COMMON_FUNCTION_UNOP_OVERLOAD(NAME, ARG, ...)            \
  template <typename T,                                                        \
            typename = std::enable_if_t<detail::is_mgenfloat_v<T>>>            \
  T NAME(ARG) {                                                                \
    __SYCL_MARRAY_COMMON_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)             \
  }

__SYCL_MARRAY_COMMON_FUNCTION_UNOP_OVERLOAD(degrees, T radians, radians[i])
__SYCL_MARRAY_COMMON_FUNCTION_UNOP_OVERLOAD(radians, T degrees, degrees[i])
__SYCL_MARRAY_COMMON_FUNCTION_UNOP_OVERLOAD(sign, T x, x[i])

#undef __SYCL_MARRAY_COMMON_FUNCTION_UNOP_OVERLOAD

#define __SYCL_MARRAY_COMMON_FUNCTION_BINOP_OVERLOAD(NAME, ARG1, ARG2, ...)    \
  template <typename T,                                                        \
            typename = std::enable_if_t<detail::is_mgenfloat_v<T>>>            \
  T NAME(ARG1, ARG2) {                                                         \
    __SYCL_MARRAY_COMMON_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)             \
  }

// min and max may be defined as macros, so we wrap them in parentheses to avoid
// errors.
__SYCL_MARRAY_COMMON_FUNCTION_BINOP_OVERLOAD((min), T x, T y, x[i], y[i])
__SYCL_MARRAY_COMMON_FUNCTION_BINOP_OVERLOAD((min), T x,
                                             detail::marray_element_t<T> y,
                                             x[i], y)
__SYCL_MARRAY_COMMON_FUNCTION_BINOP_OVERLOAD((max), T x, T y, x[i], y[i])
__SYCL_MARRAY_COMMON_FUNCTION_BINOP_OVERLOAD((max), T x,
                                             detail::marray_element_t<T> y,
                                             x[i], y)
__SYCL_MARRAY_COMMON_FUNCTION_BINOP_OVERLOAD(step, T edge, T x, edge[i], x[i])
__SYCL_MARRAY_COMMON_FUNCTION_BINOP_OVERLOAD(step,
                                             detail::marray_element_t<T> edge,
                                             T x, edge, x[i])

#undef __SYCL_MARRAY_COMMON_FUNCTION_BINOP_OVERLOAD

#define __SYCL_MARRAY_COMMON_FUNCTION_TEROP_OVERLOAD(NAME, ARG1, ARG2, ARG3,   \
                                                     ...)                      \
  template <typename T,                                                        \
            typename = std::enable_if_t<detail::is_mgenfloat_v<T>>>            \
  T NAME(ARG1, ARG2, ARG3) {                                                   \
    __SYCL_MARRAY_COMMON_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)             \
  }

__SYCL_MARRAY_COMMON_FUNCTION_TEROP_OVERLOAD(clamp, T x, T minval, T maxval,
                                             x[i], minval[i], maxval[i])
__SYCL_MARRAY_COMMON_FUNCTION_TEROP_OVERLOAD(clamp, T x,
                                             detail::marray_element_t<T> minval,
                                             detail::marray_element_t<T> maxval,
                                             x[i], minval, maxval)
__SYCL_MARRAY_COMMON_FUNCTION_TEROP_OVERLOAD(mix, T x, T y, T a, x[i], y[i],
                                             a[i])
__SYCL_MARRAY_COMMON_FUNCTION_TEROP_OVERLOAD(mix, T x, T y,
                                             detail::marray_element_t<T> a,
                                             x[i], y[i], a)
__SYCL_MARRAY_COMMON_FUNCTION_TEROP_OVERLOAD(smoothstep, T edge0, T edge1, T x,
                                             edge0[i], edge1[i], x[i])
__SYCL_MARRAY_COMMON_FUNCTION_TEROP_OVERLOAD(smoothstep,
                                             detail::marray_element_t<T> edge0,
                                             detail::marray_element_t<T> edge1,
                                             T x, edge0, edge1, x[i])

#undef __SYCL_MARRAY_COMMON_FUNCTION_TEROP_OVERLOAD
#undef __SYCL_MARRAY_COMMON_FUNCTION_OVERLOAD_IMPL

/* --------------- 4.13.4 Integer functions. --------------------------------*/
// igeninteger abs (geninteger x)
template <typename T>
std::enable_if_t<detail::is_igeninteger_v<T>, T> abs(T x) {
  auto res = __sycl_std::__invoke_s_abs<detail::make_unsigned_t<T>>(x);
  if constexpr (detail::is_vigeninteger_v<T>) {
    return res.template convert<detail::vector_element_t<T>>();
  } else
    return detail::make_signed_t<decltype(res)>(res);
}

// geninteger clamp (geninteger x, sgeninteger minval, sgeninteger maxval)
template <typename T>
std::enable_if_t<detail::is_vigeninteger_v<T>, T>
clamp(T x, typename T::element_type minval, typename T::element_type maxval) {
  return __sycl_std::__invoke_s_clamp<T>(x, T(minval), T(maxval));
}

// geninteger clamp (geninteger x, sgeninteger minval, sgeninteger maxval)
template <typename T>
std::enable_if_t<detail::is_vugeninteger_v<T>, T>
clamp(T x, typename T::element_type minval, typename T::element_type maxval) {
  return __sycl_std::__invoke_u_clamp<T>(x, T(minval), T(maxval));
}

// igeninteger max (vigeninteger x, sigeninteger y)
template <typename T>
std::enable_if_t<detail::is_vigeninteger_v<T>, T>(max)(
    T x, typename T::element_type y) {
  return __sycl_std::__invoke_s_max<T>(x, T(y));
}

// vugeninteger max (vugeninteger x, sugeninteger y)
template <typename T>
std::enable_if_t<detail::is_vugeninteger_v<T>, T>(max)(
    T x, typename T::element_type y) {
  return __sycl_std::__invoke_u_max<T>(x, T(y));
}

// vigeninteger min (vigeninteger x, sigeninteger y)
template <typename T>
std::enable_if_t<detail::is_vigeninteger_v<T>, T>(min)(
    T x, typename T::element_type y) {
  return __sycl_std::__invoke_s_min<T>(x, T(y));
}

// vugeninteger min (vugeninteger x, sugeninteger y)
template <typename T>
std::enable_if_t<detail::is_vugeninteger_v<T>, T>(min)(
    T x, typename T::element_type y) {
  return __sycl_std::__invoke_u_min<T>(x, T(y));
}

// marray integer functions

// TODO: can be optimized in the way math functions are optimized (usage of
// vec<T, 2>)
#define __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, ...)                \
  marray<T, N> res;                                                            \
  for (int j = 0; j < N; j++) {                                                \
    res[j] = NAME(__VA_ARGS__);                                                \
  }                                                                            \
  return res;

// Keep NAME for readability
#define __SYCL_MARRAY_INTEGER_FUNCTION_ABS_U_OVERLOAD(NAME, ARG, ...)          \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_ugeninteger_v<T>, marray<T, N>> NAME(            \
      marray<T, N> ARG) {                                                      \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_ABS_I_OVERLOAD(NAME, ARG, ...)          \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_igeninteger_v<T>, marray<T, N>> NAME(            \
      marray<T, N> ARG) {                                                      \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

__SYCL_MARRAY_INTEGER_FUNCTION_ABS_U_OVERLOAD(abs, x, x[j])
__SYCL_MARRAY_INTEGER_FUNCTION_ABS_I_OVERLOAD(abs, x, x[j])

#undef __SYCL_MARRAY_INTEGER_FUNCTION_ABS_I_OVERLOAD
#undef __SYCL_MARRAY_INTEGER_FUNCTION_ABS_U_OVERLOAD

#define __SYCL_MARRAY_INTEGER_FUNCTION_UNOP_OVERLOAD(NAME, ARG, ...)           \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_geninteger_v<T>, marray<T, N>> NAME(             \
      marray<T, N> ARG) {                                                      \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

__SYCL_MARRAY_INTEGER_FUNCTION_UNOP_OVERLOAD(clz, x, x[j])
__SYCL_MARRAY_INTEGER_FUNCTION_UNOP_OVERLOAD(ctz, x, x[j])
__SYCL_MARRAY_INTEGER_FUNCTION_UNOP_OVERLOAD(popcount, x, x[j])

#undef __SYCL_MARRAY_INTEGER_FUNCTION_UNOP_OVERLOAD

#define __SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_OVERLOAD(NAME, ARG1, ARG2, ...) \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_ugeninteger_v<T>, marray<T, N>> NAME(            \
      marray<T, N> ARG1, marray<T, N> ARG2) {                                  \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_RET_U_OVERLOAD(NAME, ARG1,      \
                                                              ARG2, ...)       \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_igeninteger_v<T>,                                \
                   marray<detail::make_unsigned_t<T>, N>>                      \
  NAME(marray<T, N> ARG1, marray<T, N> ARG2) {                                 \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_OVERLOAD(NAME, ARG1, ARG2, ...) \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_igeninteger_v<T>, marray<T, N>> NAME(            \
      marray<T, N> ARG1, marray<T, N> ARG2) {                                  \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_2ND_ARG_SCALAR_OVERLOAD(        \
    NAME, ARG1, ARG2, ...)                                                     \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_ugeninteger_v<T>, marray<T, N>> NAME(            \
      marray<T, N> ARG1, T ARG2) {                                             \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_2ND_ARG_SCALAR_OVERLOAD(        \
    NAME, ARG1, ARG2, ...)                                                     \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_igeninteger_v<T>, marray<T, N>> NAME(            \
      marray<T, N> ARG1, T ARG2) {                                             \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_OVERLOAD(abs_diff, x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_RET_U_OVERLOAD(abs_diff, x, y, x[j],
                                                      y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_OVERLOAD(add_sat, x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_OVERLOAD(add_sat, x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_OVERLOAD(hadd, x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_OVERLOAD(hadd, x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_OVERLOAD(rhadd, x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_OVERLOAD(rhadd, x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_OVERLOAD((max), x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_OVERLOAD((max), x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_2ND_ARG_SCALAR_OVERLOAD((max), x, y,
                                                               x[j], y)
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_2ND_ARG_SCALAR_OVERLOAD((max), x, y,
                                                               x[j], y)
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_OVERLOAD((min), x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_OVERLOAD((min), x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_2ND_ARG_SCALAR_OVERLOAD((min), x, y,
                                                               x[j], y)
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_2ND_ARG_SCALAR_OVERLOAD((min), x, y,
                                                               x[j], y)
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_OVERLOAD(mul_hi, x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_OVERLOAD(mul_hi, x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_OVERLOAD(rotate, v, i, v[j], i[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_OVERLOAD(rotate, v, i, v[j], i[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_OVERLOAD(sub_sat, x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_OVERLOAD(sub_sat, x, y, x[j], y[j])

#undef __SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_2ND_ARG_SCALAR_OVERLOAD
#undef __SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_2ND_ARG_SCALAR_OVERLOAD
#undef __SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_OVERLOAD
#undef __SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_RET_U_OVERLOAD
#undef __SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_OVERLOAD

#define __SYCL_MARRAY_INTEGER_FUNCTION_TEROP_U_OVERLOAD(NAME, ARG1, ARG2,      \
                                                        ARG3, ...)             \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_ugeninteger_v<T>, marray<T, N>> NAME(            \
      marray<T, N> ARG1, marray<T, N> ARG2, marray<T, N> ARG3) {               \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_TEROP_I_OVERLOAD(NAME, ARG1, ARG2,      \
                                                        ARG3, ...)             \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_igeninteger_v<T>, marray<T, N>> NAME(            \
      marray<T, N> ARG1, marray<T, N> ARG2, marray<T, N> ARG3) {               \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_TEROP_U_2ND_3RD_ARGS_SCALAR_OVERLOAD(   \
    NAME, ARG1, ARG2, ARG3, ...)                                               \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_ugeninteger_v<T>, marray<T, N>> NAME(            \
      marray<T, N> ARG1, T ARG2, T ARG3) {                                     \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_TEROP_I_2ND_3RD_ARGS_SCALAR_OVERLOAD(   \
    NAME, ARG1, ARG2, ARG3, ...)                                               \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_igeninteger_v<T>, marray<T, N>> NAME(            \
      marray<T, N> ARG1, T ARG2, T ARG3) {                                     \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

__SYCL_MARRAY_INTEGER_FUNCTION_TEROP_U_OVERLOAD(clamp, x, minval, maxval, x[j],
                                                minval[j], maxval[j])
__SYCL_MARRAY_INTEGER_FUNCTION_TEROP_I_OVERLOAD(clamp, x, minval, maxval, x[j],
                                                minval[j], maxval[j])
__SYCL_MARRAY_INTEGER_FUNCTION_TEROP_U_2ND_3RD_ARGS_SCALAR_OVERLOAD(
    clamp, x, minval, maxval, x[j], minval, maxval)
__SYCL_MARRAY_INTEGER_FUNCTION_TEROP_I_2ND_3RD_ARGS_SCALAR_OVERLOAD(
    clamp, x, minval, maxval, x[j], minval, maxval)
__SYCL_MARRAY_INTEGER_FUNCTION_TEROP_U_OVERLOAD(mad_hi, a, b, c, a[j], b[j],
                                                c[j])
__SYCL_MARRAY_INTEGER_FUNCTION_TEROP_I_OVERLOAD(mad_hi, a, b, c, a[j], b[j],
                                                c[j])
__SYCL_MARRAY_INTEGER_FUNCTION_TEROP_U_OVERLOAD(mad_sat, a, b, c, a[j], b[j],
                                                c[j])
__SYCL_MARRAY_INTEGER_FUNCTION_TEROP_I_OVERLOAD(mad_sat, a, b, c, a[j], b[j],
                                                c[j])

#undef __SYCL_MARRAY_INTEGER_FUNCTION_TEROP_U_2ND_3RD_ARGS_SCALAR_OVERLOAD
#undef __SYCL_MARRAY_INTEGER_FUNCTION_TEROP_I_2ND_3RD_ARGS_SCALAR_OVERLOAD
#undef __SYCL_MARRAY_INTEGER_FUNCTION_TEROP_I_OVERLOAD
#undef __SYCL_MARRAY_INTEGER_FUNCTION_TEROP_U_OVERLOAD

// Keep NAME for readability
#define __SYCL_MARRAY_INTEGER_FUNCTION_MAD24_U_OVERLOAD(NAME, ARG1, ARG2,      \
                                                        ARG3, ...)             \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_ugeninteger32bit_v<T>, marray<T, N>> NAME(       \
      marray<T, N> ARG1, marray<T, N> ARG2, marray<T, N> ARG3) {               \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_MAD24_I_OVERLOAD(NAME, ARG1, ARG2,      \
                                                        ARG3, ...)             \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_igeninteger32bit_v<T>, marray<T, N>> NAME(       \
      marray<T, N> ARG1, marray<T, N> ARG2, marray<T, N> ARG3) {               \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

__SYCL_MARRAY_INTEGER_FUNCTION_MAD24_U_OVERLOAD(mad24, x, y, z, x[j], y[j],
                                                z[j])
__SYCL_MARRAY_INTEGER_FUNCTION_MAD24_I_OVERLOAD(mad24, x, y, z, x[j], y[j],
                                                z[j])

#undef __SYCL_MARRAY_INTEGER_FUNCTION_MAD24_I_OVERLOAD
#undef __SYCL_MARRAY_INTEGER_FUNCTION_MAD24_U_OVERLOAD

// Keep NAME for readability
#define __SYCL_MARRAY_INTEGER_FUNCTION_MUL24_U_OVERLOAD(NAME, ARG1, ARG2, ...) \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_ugeninteger32bit_v<T>, marray<T, N>> NAME(       \
      marray<T, N> ARG1, marray<T, N> ARG2) {                                  \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_MUL24_I_OVERLOAD(NAME, ARG1, ARG2, ...) \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_igeninteger32bit_v<T>, marray<T, N>> NAME(       \
      marray<T, N> ARG1, marray<T, N> ARG2) {                                  \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

__SYCL_MARRAY_INTEGER_FUNCTION_MUL24_U_OVERLOAD(mul24, x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_MUL24_I_OVERLOAD(mul24, x, y, x[j], y[j])

#undef __SYCL_MARRAY_INTEGER_FUNCTION_MUL24_I_OVERLOAD
#undef __SYCL_MARRAY_INTEGER_FUNCTION_MUL24_U_OVERLOAD
#undef __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL

// TODO: can be optimized in the way math functions are optimized (usage of
// vec<T, 2>)
#define __SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_OVERLOAD_IMPL(NAME)            \
  detail::make_larger_t<marray<T, N>> res;                                     \
  for (int j = 0; j < N; j++) {                                                \
    res[j] = NAME(hi[j], lo[j]);                                               \
  }                                                                            \
  return res;

// Keep NAME for readability
#define __SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_UU_OVERLOAD(NAME, KBIT)        \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_ugeninteger##KBIT##_v<T>,                        \
                   detail::make_larger_t<marray<T, N>>>                        \
  NAME(marray<T, N> hi, marray<T, N> lo) {                                     \
    __SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_OVERLOAD_IMPL(NAME)                \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_IU_OVERLOAD(NAME, KBIT)        \
  template <typename T, typename T2, size_t N>                                 \
  std::enable_if_t<detail::is_igeninteger##KBIT##_v<T> &&                      \
                       detail::is_ugeninteger##KBIT##_v<T2>,                   \
                   detail::make_larger_t<marray<T, N>>>                        \
  NAME(marray<T, N> hi, marray<T2, N> lo) {                                    \
    __SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_OVERLOAD_IMPL(NAME)                \
  }

__SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_UU_OVERLOAD(upsample, 8bit)
__SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_IU_OVERLOAD(upsample, 8bit)
__SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_UU_OVERLOAD(upsample, 16bit)
__SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_IU_OVERLOAD(upsample, 16bit)
__SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_UU_OVERLOAD(upsample, 32bit)
__SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_IU_OVERLOAD(upsample, 32bit)

#undef __SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_IU_OVERLOAD
#undef __SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_UU_OVERLOAD
#undef __SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_OVERLOAD_IMPL

/* --------------- 4.13.6 Geometric Functions. ------------------------------*/
// float3 cross (float3 p0, float3 p1)
// float4 cross (float4 p0, float4 p1)
// double3 cross (double3 p0, double3 p1)
// double4 cross (double4 p0, double4 p1)
// half3 cross (half3 p0, half3 p1)
// half4 cross (half4 p0, half4 p1)
template <typename T>
std::enable_if_t<detail::is_gencross_v<T>, T> cross(T p0, T p1) {
  return __sycl_std::__invoke_cross<T>(p0, p1);
}

// float dot (vgengeofloat p0, vgengeofloat p1)
template <typename T>
std::enable_if_t<detail::is_vgengeofloat_v<T>, float> dot(T p0, T p1) {
  return __sycl_std::__invoke_Dot<float>(p0, p1);
}

// double dot (vgengeodouble p0, vgengeodouble p1)
template <typename T>
std::enable_if_t<detail::is_vgengeodouble_v<T>, double> dot(T p0, T p1) {
  return __sycl_std::__invoke_Dot<double>(p0, p1);
}

// half dot (vgengeohalf p0, vgengeohalf p1)
template <typename T>
std::enable_if_t<detail::is_vgengeohalf_v<T>, half> dot(T p0, T p1) {
  return __sycl_std::__invoke_Dot<half>(p0, p1);
}

// float distance (gengeofloat p0, gengeofloat p1)
template <typename T,
          typename = std::enable_if_t<detail::is_gengeofloat_v<T>, T>>
float distance(T p0, T p1) {
  return __sycl_std::__invoke_distance<float>(p0, p1);
}

// double distance (gengeodouble p0, gengeodouble p1)
template <typename T,
          typename = std::enable_if_t<detail::is_gengeodouble_v<T>, T>>
double distance(T p0, T p1) {
  return __sycl_std::__invoke_distance<double>(p0, p1);
}

// half distance (gengeohalf p0, gengeohalf p1)
template <typename T,
          typename = std::enable_if_t<detail::is_gengeohalf_v<T>, T>>
half distance(T p0, T p1) {
  return __sycl_std::__invoke_distance<half>(p0, p1);
}

// float length (gengeofloat p)
template <typename T,
          typename = std::enable_if_t<detail::is_gengeofloat_v<T>, T>>
float length(T p) {
  return __sycl_std::__invoke_length<float>(p);
}

// double length (gengeodouble p)
template <typename T,
          typename = std::enable_if_t<detail::is_gengeodouble_v<T>, T>>
double length(T p) {
  return __sycl_std::__invoke_length<double>(p);
}

// half length (gengeohalf p)
template <typename T,
          typename = std::enable_if_t<detail::is_gengeohalf_v<T>, T>>
half length(T p) {
  return __sycl_std::__invoke_length<half>(p);
}

// gengeofloat normalize (gengeofloat p)
template <typename T>
std::enable_if_t<detail::is_gengeofloat_v<T>, T> normalize(T p) {
  return __sycl_std::__invoke_normalize<T>(p);
}

// gengeodouble normalize (gengeodouble p)
template <typename T>
std::enable_if_t<detail::is_gengeodouble_v<T>, T> normalize(T p) {
  return __sycl_std::__invoke_normalize<T>(p);
}

// gengeohalf normalize (gengeohalf p)
template <typename T>
std::enable_if_t<detail::is_gengeohalf_v<T>, T> normalize(T p) {
  return __sycl_std::__invoke_normalize<T>(p);
}

// float fast_distance (gengeofloat p0, gengeofloat p1)
template <typename T,
          typename = std::enable_if_t<detail::is_gengeofloat_v<T>, T>>
float fast_distance(T p0, T p1) {
  return __sycl_std::__invoke_fast_distance<float>(p0, p1);
}

// double fast_distance (gengeodouble p0, gengeodouble p1)
template <typename T,
          typename = std::enable_if_t<detail::is_gengeodouble_v<T>, T>>
__SYCL_DEPRECATED("fast_distance for double precision types is non-standard "
                  "and has been deprecated")
double fast_distance(T p0, T p1) {
  return __sycl_std::__invoke_fast_distance<double>(p0, p1);
}

// float fast_length (gengeofloat p)
template <typename T,
          typename = std::enable_if_t<detail::is_gengeofloat_v<T>, T>>
float fast_length(T p) {
  return __sycl_std::__invoke_fast_length<float>(p);
}

// double fast_length (gengeodouble p)
template <typename T,
          typename = std::enable_if_t<detail::is_gengeodouble_v<T>, T>>
__SYCL_DEPRECATED("fast_length for double precision types is non-standard "
                  "and has been deprecated")
double fast_length(T p) {
  return __sycl_std::__invoke_fast_length<double>(p);
}

// gengeofloat fast_normalize (gengeofloat p)
template <typename T>
std::enable_if_t<detail::is_gengeofloat_v<T>, T> fast_normalize(T p) {
  return __sycl_std::__invoke_fast_normalize<T>(p);
}

// gengeodouble fast_normalize (gengeodouble p)
template <typename T>
__SYCL_DEPRECATED("fast_normalize for double precision types is non-standard "
                  "and has been deprecated")
std::enable_if_t<detail::is_gengeodouble_v<T>, T> fast_normalize(T p) {
  return __sycl_std::__invoke_fast_normalize<T>(p);
}

// marray geometric functions

#define __SYCL_MARRAY_GEOMETRIC_FUNCTION_OVERLOAD_IMPL(NAME, ...)              \
  vec<detail::marray_element_t<T>, T::size()> result_v;                        \
  result_v = NAME(__VA_ARGS__);                                                \
  return detail::to_marray(result_v);

template <typename T>
std::enable_if_t<detail::is_gencrossmarray_v<T>, T> cross(T p0, T p1) {
  __SYCL_MARRAY_GEOMETRIC_FUNCTION_OVERLOAD_IMPL(cross, detail::to_vec(p0),
                                                 detail::to_vec(p1))
}

template <typename T>
std::enable_if_t<detail::is_gengeomarray_v<T>, T> normalize(T p) {
  __SYCL_MARRAY_GEOMETRIC_FUNCTION_OVERLOAD_IMPL(normalize, detail::to_vec(p))
}

template <typename T>
std::enable_if_t<detail::is_gengeomarrayfloat_v<T>, T> fast_normalize(T p) {
  __SYCL_MARRAY_GEOMETRIC_FUNCTION_OVERLOAD_IMPL(fast_normalize,
                                                 detail::to_vec(p))
}

#undef __SYCL_MARRAY_GEOMETRIC_FUNCTION_OVERLOAD_IMPL

#define __SYCL_MARRAY_GEOMETRIC_FUNCTION_IS_GENGEOMARRAY_BINOP_OVERLOAD(NAME)  \
  template <typename T>                                                        \
  std::enable_if_t<detail::is_gengeomarray_v<T>, detail::marray_element_t<T>>  \
  NAME(T p0, T p1) {                                                           \
    return NAME(detail::to_vec(p0), detail::to_vec(p1));                       \
  }

// clang-format off
__SYCL_MARRAY_GEOMETRIC_FUNCTION_IS_GENGEOMARRAY_BINOP_OVERLOAD(dot)
__SYCL_MARRAY_GEOMETRIC_FUNCTION_IS_GENGEOMARRAY_BINOP_OVERLOAD(distance)
// clang-format on

#undef __SYCL_MARRAY_GEOMETRIC_FUNCTION_IS_GENGEOMARRAY_BINOP_OVERLOAD

template <typename T>
std::enable_if_t<detail::is_gengeomarray_v<T>, detail::marray_element_t<T>>
length(T p) {
  return __sycl_std::__invoke_length<detail::marray_element_t<T>>(
      detail::to_vec(p));
}

template <typename T>
std::enable_if_t<detail::is_gengeomarrayfloat_v<T>, detail::marray_element_t<T>>
fast_distance(T p0, T p1) {
  return fast_distance(detail::to_vec(p0), detail::to_vec(p1));
}

template <typename T>
std::enable_if_t<detail::is_gengeomarrayfloat_v<T>, detail::marray_element_t<T>>
fast_length(T p) {
  return fast_length(detail::to_vec(p));
}

/* SYCL 1.2.1 ---- 4.13.7 Relational functions. -----------------------------*/
/* SYCL 2020  ---- 4.17.9 Relational functions. -----------------------------*/

// marray relational functions

#define __SYCL_MARRAY_RELATIONAL_FUNCTION_BINOP_OVERLOAD(NAME)                 \
  template <typename T,                                                        \
            typename = std::enable_if_t<detail::is_mgenfloat_v<T>>>            \
  sycl::marray<bool, T::size()> NAME(T x, T y) {                               \
    sycl::marray<bool, T::size()> res;                                         \
    for (int i = 0; i < x.size(); i++) {                                       \
      res[i] = NAME(x[i], y[i]);                                               \
    }                                                                          \
    return res;                                                                \
  }

#define __SYCL_MARRAY_RELATIONAL_FUNCTION_UNOP_OVERLOAD(NAME)                  \
  template <typename T,                                                        \
            typename = std::enable_if_t<detail::is_mgenfloat_v<T>>>            \
  sycl::marray<bool, T::size()> NAME(T x) {                                    \
    sycl::marray<bool, T::size()> res;                                         \
    for (int i = 0; i < x.size(); i++) {                                       \
      res[i] = NAME(x[i]);                                                     \
    }                                                                          \
    return res;                                                                \
  }

__SYCL_MARRAY_RELATIONAL_FUNCTION_BINOP_OVERLOAD(isequal)
__SYCL_MARRAY_RELATIONAL_FUNCTION_BINOP_OVERLOAD(isnotequal)
__SYCL_MARRAY_RELATIONAL_FUNCTION_BINOP_OVERLOAD(isgreater)
__SYCL_MARRAY_RELATIONAL_FUNCTION_BINOP_OVERLOAD(isgreaterequal)
__SYCL_MARRAY_RELATIONAL_FUNCTION_BINOP_OVERLOAD(isless)
__SYCL_MARRAY_RELATIONAL_FUNCTION_BINOP_OVERLOAD(islessequal)
__SYCL_MARRAY_RELATIONAL_FUNCTION_BINOP_OVERLOAD(islessgreater)
__SYCL_MARRAY_RELATIONAL_FUNCTION_UNOP_OVERLOAD(isfinite)
__SYCL_MARRAY_RELATIONAL_FUNCTION_UNOP_OVERLOAD(isinf)
__SYCL_MARRAY_RELATIONAL_FUNCTION_UNOP_OVERLOAD(isnan)
__SYCL_MARRAY_RELATIONAL_FUNCTION_UNOP_OVERLOAD(isnormal)
__SYCL_MARRAY_RELATIONAL_FUNCTION_BINOP_OVERLOAD(isordered)
__SYCL_MARRAY_RELATIONAL_FUNCTION_BINOP_OVERLOAD(isunordered)
__SYCL_MARRAY_RELATIONAL_FUNCTION_UNOP_OVERLOAD(signbit)

// int any (vigeninteger x)
template <typename T>
std::enable_if_t<detail::is_vigeninteger_v<T>, int> any(T x) {
  return detail::rel_sign_bit_test_ret_t<T>(
      __sycl_std::__invoke_Any<detail::rel_sign_bit_test_ret_t<T>>(
          detail::rel_sign_bit_test_arg_t<T>(x)));
}

// int all (vigeninteger x)
template <typename T>
std::enable_if_t<detail::is_vigeninteger_v<T>, int> all(T x) {
  return detail::rel_sign_bit_test_ret_t<T>(
      __sycl_std::__invoke_All<detail::rel_sign_bit_test_ret_t<T>>(
          detail::rel_sign_bit_test_arg_t<T>(x)));
}

// other marray relational functions

template <typename T, size_t N>
std::enable_if_t<detail::is_sigeninteger_v<T>, bool> any(marray<T, N> x) {
  return std::any_of(x.begin(), x.end(), [](T i) { return any(i); });
}

template <typename T, size_t N>
std::enable_if_t<detail::is_sigeninteger_v<T>, bool> all(marray<T, N> x) {
  return std::all_of(x.begin(), x.end(), [](T i) { return all(i); });
}

template <typename T, size_t N>
std::enable_if_t<detail::is_gentype_v<T>, marray<T, N>>
bitselect(marray<T, N> a, marray<T, N> b, marray<T, N> c) {
  marray<T, N> res;
  for (int i = 0; i < N; i++) {
    res[i] = bitselect(a[i], b[i], c[i]);
  }
  return res;
}

template <typename T, size_t N>
std::enable_if_t<detail::is_gentype_v<T>, marray<T, N>>
select(marray<T, N> a, marray<T, N> b, marray<bool, N> c) {
  marray<T, N> res;
  for (int i = 0; i < N; i++) {
    res[i] = select(a[i], b[i], c[i]);
  }
  return res;
}

namespace native {
/* ----------------- 4.13.3 Math functions. ---------------------------------*/

#define __SYCL_NATIVE_MATH_FUNCTION_OVERLOAD(NAME)                             \
  template <size_t N>                                                          \
  inline __SYCL_ALWAYS_INLINE marray<float, N> NAME(marray<float, N> x) {      \
    marray<float, N> res;                                                      \
    for (size_t i = 0; i < N / 2; i++) {                                       \
      auto partial_res = __sycl_std::__invoke_native_##NAME<vec<float, 2>>(    \
          detail::to_vec2(x, i * 2));                                          \
      std::memcpy(&res[i * 2], &partial_res, sizeof(vec<float, 2>));           \
    }                                                                          \
    if (N % 2) {                                                               \
      res[N - 1] = __sycl_std::__invoke_native_##NAME<float>(x[N - 1]);        \
    }                                                                          \
    return res;                                                                \
  }

__SYCL_NATIVE_MATH_FUNCTION_OVERLOAD(sin)
__SYCL_NATIVE_MATH_FUNCTION_OVERLOAD(cos)
__SYCL_NATIVE_MATH_FUNCTION_OVERLOAD(tan)
__SYCL_NATIVE_MATH_FUNCTION_OVERLOAD(exp)
__SYCL_NATIVE_MATH_FUNCTION_OVERLOAD(exp2)
__SYCL_NATIVE_MATH_FUNCTION_OVERLOAD(exp10)
__SYCL_NATIVE_MATH_FUNCTION_OVERLOAD(log)
__SYCL_NATIVE_MATH_FUNCTION_OVERLOAD(log2)
__SYCL_NATIVE_MATH_FUNCTION_OVERLOAD(log10)
__SYCL_NATIVE_MATH_FUNCTION_OVERLOAD(sqrt)
__SYCL_NATIVE_MATH_FUNCTION_OVERLOAD(rsqrt)
__SYCL_NATIVE_MATH_FUNCTION_OVERLOAD(recip)

#undef __SYCL_NATIVE_MATH_FUNCTION_OVERLOAD

#define __SYCL_NATIVE_MATH_FUNCTION_2_OVERLOAD(NAME)                           \
  template <size_t N>                                                          \
  inline __SYCL_ALWAYS_INLINE marray<float, N> NAME(marray<float, N> x,        \
                                                    marray<float, N> y) {      \
    marray<float, N> res;                                                      \
    for (size_t i = 0; i < N / 2; i++) {                                       \
      auto partial_res = __sycl_std::__invoke_native_##NAME<vec<float, 2>>(    \
          detail::to_vec2(x, i * 2), detail::to_vec2(y, i * 2));               \
      std::memcpy(&res[i * 2], &partial_res, sizeof(vec<float, 2>));           \
    }                                                                          \
    if (N % 2) {                                                               \
      res[N - 1] =                                                             \
          __sycl_std::__invoke_native_##NAME<float>(x[N - 1], y[N - 1]);       \
    }                                                                          \
    return res;                                                                \
  }

__SYCL_NATIVE_MATH_FUNCTION_2_OVERLOAD(divide)
__SYCL_NATIVE_MATH_FUNCTION_2_OVERLOAD(powr)

#undef __SYCL_NATIVE_MATH_FUNCTION_2_OVERLOAD

} // namespace native
namespace half_precision {
/* ----------------- 4.13.3 Math functions. ---------------------------------*/
#define __SYCL_HALF_PRECISION_MATH_FUNCTION_OVERLOAD(NAME)                     \
  template <size_t N>                                                          \
  inline __SYCL_ALWAYS_INLINE marray<float, N> NAME(marray<float, N> x) {      \
    marray<float, N> res;                                                      \
    for (size_t i = 0; i < N / 2; i++) {                                       \
      auto partial_res = __sycl_std::__invoke_half_##NAME<vec<float, 2>>(      \
          detail::to_vec2(x, i * 2));                                          \
      std::memcpy(&res[i * 2], &partial_res, sizeof(vec<float, 2>));           \
    }                                                                          \
    if (N % 2) {                                                               \
      res[N - 1] = __sycl_std::__invoke_half_##NAME<float>(x[N - 1]);          \
    }                                                                          \
    return res;                                                                \
  }

__SYCL_HALF_PRECISION_MATH_FUNCTION_OVERLOAD(sin)
__SYCL_HALF_PRECISION_MATH_FUNCTION_OVERLOAD(cos)
__SYCL_HALF_PRECISION_MATH_FUNCTION_OVERLOAD(tan)
__SYCL_HALF_PRECISION_MATH_FUNCTION_OVERLOAD(exp)
__SYCL_HALF_PRECISION_MATH_FUNCTION_OVERLOAD(exp2)
__SYCL_HALF_PRECISION_MATH_FUNCTION_OVERLOAD(exp10)
__SYCL_HALF_PRECISION_MATH_FUNCTION_OVERLOAD(log)
__SYCL_HALF_PRECISION_MATH_FUNCTION_OVERLOAD(log2)
__SYCL_HALF_PRECISION_MATH_FUNCTION_OVERLOAD(log10)
__SYCL_HALF_PRECISION_MATH_FUNCTION_OVERLOAD(sqrt)
__SYCL_HALF_PRECISION_MATH_FUNCTION_OVERLOAD(rsqrt)
__SYCL_HALF_PRECISION_MATH_FUNCTION_OVERLOAD(recip)

#undef __SYCL_HALF_PRECISION_MATH_FUNCTION_OVERLOAD

#define __SYCL_HALF_PRECISION_MATH_FUNCTION_2_OVERLOAD(NAME)                   \
  template <size_t N>                                                          \
  inline __SYCL_ALWAYS_INLINE marray<float, N> NAME(marray<float, N> x,        \
                                                    marray<float, N> y) {      \
    marray<float, N> res;                                                      \
    for (size_t i = 0; i < N / 2; i++) {                                       \
      auto partial_res = __sycl_std::__invoke_half_##NAME<vec<float, 2>>(      \
          detail::to_vec2(x, i * 2), detail::to_vec2(y, i * 2));               \
      std::memcpy(&res[i * 2], &partial_res, sizeof(vec<float, 2>));           \
    }                                                                          \
    if (N % 2) {                                                               \
      res[N - 1] =                                                             \
          __sycl_std::__invoke_half_##NAME<float>(x[N - 1], y[N - 1]);         \
    }                                                                          \
    return res;                                                                \
  }

__SYCL_HALF_PRECISION_MATH_FUNCTION_2_OVERLOAD(divide)
__SYCL_HALF_PRECISION_MATH_FUNCTION_2_OVERLOAD(powr)

#undef __SYCL_HALF_PRECISION_MATH_FUNCTION_2_OVERLOAD

} // namespace half_precision

#ifdef __FAST_MATH__
/* ----------------- -ffast-math functions. ---------------------------------*/

#define __SYCL_MATH_FUNCTION_OVERLOAD_FM(NAME)                                 \
  template <typename T, size_t N>                                              \
  inline __SYCL_ALWAYS_INLINE                                                  \
      std::enable_if_t<std::is_same_v<T, float>, marray<T, N>>                 \
      NAME(marray<T, N> x) {                                                   \
    return native::NAME(x);                                                    \
  }

__SYCL_MATH_FUNCTION_OVERLOAD_FM(sin)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(cos)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(tan)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(exp)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(exp2)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(exp10)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(log)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(log2)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(log10)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(sqrt)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(rsqrt)
#undef __SYCL_MATH_FUNCTION_OVERLOAD_FM

template <typename T, size_t N>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<std::is_same_v<T, float>, marray<T, N>>
    powr(marray<T, N> x, marray<T, N> y) {
  return native::powr(x, y);
}

#endif // __FAST_MATH__
} // namespace _V1
} // namespace sycl
