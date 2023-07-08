//==----------- builtins.hpp - SYCL built-in functions ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/boolean.hpp>
#include <sycl/detail/builtins.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/generic_type_traits.hpp>
#include <sycl/pointers.hpp>
#include <sycl/types.hpp>

#include <algorithm>

// TODO Decide whether to mark functions with this attribute.
#define __NOEXC /*noexcept*/

namespace sycl {

__SYCL_INLINE_VER_NAMESPACE(_V1) {

namespace detail {
template <class T, size_t N> vec<T, 2> to_vec2(marray<T, N> x, size_t start) {
  return {x[start], x[start + 1]};
}
template <class T, size_t N> vec<T, N> to_vec(marray<T, N> x) {
  vec<T, N> vec;
  for (size_t i = 0; i < N; i++)
    vec[i] = x[i];
  return vec;
}
template <class T, int N> marray<T, N> to_marray(vec<T, N> x) {
  marray<T, N> marray;
  for (size_t i = 0; i < N; i++)
    marray[i] = x[i];
  return marray;
}
} // namespace detail

#ifdef __SYCL_DEVICE_ONLY__
#define __sycl_std
#else
namespace __sycl_std = __host_std;
#endif

#ifdef __FAST_MATH__
#define __FAST_MATH_GENFLOAT(T)                                                \
  (detail::is_svgenfloatd<T>::value || detail::is_svgenfloath<T>::value)
#define __FAST_MATH_SGENFLOAT(T)                                               \
  (std::is_same_v<T, double> || std::is_same_v<T, half>)
#else
#define __FAST_MATH_GENFLOAT(T) (detail::is_svgenfloat<T>::value)
#define __FAST_MATH_SGENFLOAT(T) (detail::is_sgenfloat<T>::value)
#endif

/* ----------------- 4.13.3 Math functions. ---------------------------------*/

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
      std::enable_if_t<detail::is_sgenfloat<T>::value, marray<T, N>>           \
      NAME(marray<T, N> x) __NOEXC {                                           \
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
      NAME(marray<T, N> x) __NOEXC {                                           \
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
    std::enable_if_t<detail::is_sgenfloat<T>::value, marray<int, N>>
    ilogb(marray<T, N> x) __NOEXC {
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
      std::enable_if_t<detail::is_sgenfloat<T>::value, marray<T, N>>           \
      NAME(marray<T, N> x, marray<T, N> y) __NOEXC {                           \
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
    powr(marray<T, N> x,
         marray<T, N> y) __NOEXC{__SYCL_MATH_FUNCTION_2_OVERLOAD_IMPL(powr)}

#undef __SYCL_MATH_FUNCTION_2_OVERLOAD_IMPL

#define __SYCL_MATH_FUNCTION_2_SGENFLOAT_Y_OVERLOAD(NAME)                      \
  template <typename T, size_t N>                                              \
  inline __SYCL_ALWAYS_INLINE                                                  \
      std::enable_if_t<detail::is_sgenfloat<T>::value, marray<T, N>>           \
      NAME(marray<T, N> x, T y) __NOEXC {                                      \
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
    std::enable_if_t<detail::is_sgenfloat<T>::value, marray<T, N>>
    ldexp(marray<T, N> x, marray<int, N> k) __NOEXC {
  // clang-format on
  marray<T, N> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = __sycl_std::__invoke_ldexp<T>(x[i], k[i]);
  }
  return res;
}

template <typename T, size_t N>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<detail::is_sgenfloat<T>::value, marray<T, N>>
    ldexp(marray<T, N> x, int k) __NOEXC {
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
    std::enable_if_t<detail::is_sgenfloat<T>::value, marray<T, N>>
    pown(marray<T, N> x, marray<int, N> y) __NOEXC {
  __SYCL_MATH_FUNCTION_2_GENINT_Y_OVERLOAD_IMPL(pown)
}

template <typename T, size_t N>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<detail::is_sgenfloat<T>::value, marray<T, N>>
    rootn(marray<T, N> x, marray<int, N> y) __NOEXC {
  __SYCL_MATH_FUNCTION_2_GENINT_Y_OVERLOAD_IMPL(rootn)
}

#undef __SYCL_MATH_FUNCTION_2_GENINT_Y_OVERLOAD_IMPL

#define __SYCL_MATH_FUNCTION_2_INT_Y_OVERLOAD_IMPL(NAME)                       \
  marray<T, N> res;                                                            \
  for (size_t i = 0; i < N; i++) {                                             \
    res[i] = __sycl_std::__invoke_##NAME<T>(x[i], y);                          \
  }                                                                            \
  return res;

template <typename T, size_t N>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<detail::is_sgenfloat<T>::value, marray<T, N>>
    pown(marray<T, N> x, int y) __NOEXC {
  __SYCL_MATH_FUNCTION_2_INT_Y_OVERLOAD_IMPL(pown)
}

template <typename T, size_t N>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<detail::is_sgenfloat<T>::value, marray<T, N>>
    rootn(marray<T, N> x,
          int y) __NOEXC{__SYCL_MATH_FUNCTION_2_INT_Y_OVERLOAD_IMPL(rootn)}

#undef __SYCL_MATH_FUNCTION_2_INT_Y_OVERLOAD_IMPL

#define __SYCL_MATH_FUNCTION_3_OVERLOAD(NAME)                                  \
  template <typename T, size_t N>                                              \
  inline __SYCL_ALWAYS_INLINE                                                  \
      std::enable_if_t<detail::is_sgenfloat<T>::value, marray<T, N>>           \
      NAME(marray<T, N> x, marray<T, N> y, marray<T, N> z) __NOEXC {           \
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

__SYCL_MATH_FUNCTION_3_OVERLOAD(mad) __SYCL_MATH_FUNCTION_3_OVERLOAD(mix)
    __SYCL_MATH_FUNCTION_3_OVERLOAD(fma)

#undef __SYCL_MATH_FUNCTION_3_OVERLOAD

    // svgenfloat acos (svgenfloat x)
    template <typename T>
    std::enable_if_t<detail::is_svgenfloat<T>::value, T> acos(T x) __NOEXC {
  return __sycl_std::__invoke_acos<T>(x);
}

// svgenfloat acosh (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> acosh(T x) __NOEXC {
  return __sycl_std::__invoke_acosh<T>(x);
}

// svgenfloat acospi (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> acospi(T x) __NOEXC {
  return __sycl_std::__invoke_acospi<T>(x);
}

// svgenfloat asin (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> asin(T x) __NOEXC {
  return __sycl_std::__invoke_asin<T>(x);
}

// svgenfloat asinh (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> asinh(T x) __NOEXC {
  return __sycl_std::__invoke_asinh<T>(x);
}

// svgenfloat asinpi (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> asinpi(T x) __NOEXC {
  return __sycl_std::__invoke_asinpi<T>(x);
}

// svgenfloat atan (svgenfloat y_over_x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> atan(T y_over_x) __NOEXC {
  return __sycl_std::__invoke_atan<T>(y_over_x);
}

// svgenfloat atan2 (svgenfloat y, svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> atan2(T y, T x) __NOEXC {
  return __sycl_std::__invoke_atan2<T>(y, x);
}

// svgenfloat atanh (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> atanh(T x) __NOEXC {
  return __sycl_std::__invoke_atanh<T>(x);
}

// svgenfloat atanpi (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> atanpi(T x) __NOEXC {
  return __sycl_std::__invoke_atanpi<T>(x);
}

// svgenfloat atan2pi (svgenfloat y, svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> atan2pi(T y, T x) __NOEXC {
  return __sycl_std::__invoke_atan2pi<T>(y, x);
}

// svgenfloat cbrt (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> cbrt(T x) __NOEXC {
  return __sycl_std::__invoke_cbrt<T>(x);
}

// svgenfloat ceil (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> ceil(T x) __NOEXC {
  return __sycl_std::__invoke_ceil<T>(x);
}

// svgenfloat copysign (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> copysign(T x,
                                                              T y) __NOEXC {
  return __sycl_std::__invoke_copysign<T>(x, y);
}

// svgenfloat cos (svgenfloat x)
template <typename T>
std::enable_if_t<__FAST_MATH_GENFLOAT(T), T> cos(T x) __NOEXC {
  return __sycl_std::__invoke_cos<T>(x);
}

// svgenfloat cosh (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> cosh(T x) __NOEXC {
  return __sycl_std::__invoke_cosh<T>(x);
}

// svgenfloat cospi (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> cospi(T x) __NOEXC {
  return __sycl_std::__invoke_cospi<T>(x);
}

// svgenfloat erfc (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> erfc(T x) __NOEXC {
  return __sycl_std::__invoke_erfc<T>(x);
}

// svgenfloat erf (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> erf(T x) __NOEXC {
  return __sycl_std::__invoke_erf<T>(x);
}

// svgenfloat exp (svgenfloat x )
template <typename T>
std::enable_if_t<__FAST_MATH_GENFLOAT(T), T> exp(T x) __NOEXC {
  return __sycl_std::__invoke_exp<T>(x);
}

// svgenfloat exp2 (svgenfloat x)
template <typename T>
std::enable_if_t<__FAST_MATH_GENFLOAT(T), T> exp2(T x) __NOEXC {
  return __sycl_std::__invoke_exp2<T>(x);
}

// svgenfloat exp10 (svgenfloat x)
template <typename T>
std::enable_if_t<__FAST_MATH_GENFLOAT(T), T> exp10(T x) __NOEXC {
  return __sycl_std::__invoke_exp10<T>(x);
}

// svgenfloat expm1 (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> expm1(T x) __NOEXC {
  return __sycl_std::__invoke_expm1<T>(x);
}

// svgenfloat fabs (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> fabs(T x) __NOEXC {
  return __sycl_std::__invoke_fabs<T>(x);
}

// svgenfloat fdim (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> fdim(T x, T y) __NOEXC {
  return __sycl_std::__invoke_fdim<T>(x, y);
}

// svgenfloat floor (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> floor(T x) __NOEXC {
  return __sycl_std::__invoke_floor<T>(x);
}

// svgenfloat fma (svgenfloat a, svgenfloat b, svgenfloat c)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> fma(T a, T b,
                                                         T c) __NOEXC {
  return __sycl_std::__invoke_fma<T>(a, b, c);
}

// svgenfloat fmax (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> fmax(T x, T y) __NOEXC {
  return __sycl_std::__invoke_fmax<T>(x, y);
}

// svgenfloat fmax (svgenfloat x, sgenfloat y)
template <typename T>
std::enable_if_t<detail::is_vgenfloat<T>::value, T>
fmax(T x, typename T::element_type y) __NOEXC {
  return __sycl_std::__invoke_fmax<T>(x, T(y));
}

// svgenfloat fmin (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> fmin(T x, T y) __NOEXC {
  return __sycl_std::__invoke_fmin<T>(x, y);
}

// svgenfloat fmin (svgenfloat x, sgenfloat y)
template <typename T>
std::enable_if_t<detail::is_vgenfloat<T>::value, T>
fmin(T x, typename T::element_type y) __NOEXC {
  return __sycl_std::__invoke_fmin<T>(x, T(y));
}

// svgenfloat fmod (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> fmod(T x, T y) __NOEXC {
  return __sycl_std::__invoke_fmod<T>(x, y);
}

// svgenfloat fract (svgenfloat x, genfloatptr iptr)
template <typename T, typename T2>
std::enable_if_t<
    detail::is_svgenfloat<T>::value && detail::is_genfloatptr<T2>::value, T>
fract(T x, T2 iptr) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_fract<T>(x, iptr);
}

// svgenfloat frexp (svgenfloat x, genintptr exp)
template <typename T, typename T2>
std::enable_if_t<
    detail::is_svgenfloat<T>::value && detail::is_genintptr<T2>::value, T>
frexp(T x, T2 exp) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_frexp<T>(x, exp);
}

// svgenfloat hypot (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> hypot(T x, T y) __NOEXC {
  return __sycl_std::__invoke_hypot<T>(x, y);
}

// genint ilogb (svgenfloat x)
template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat<T>::value, T>>
detail::change_base_type_t<T, int> ilogb(T x) __NOEXC {
  return __sycl_std::__invoke_ilogb<detail::change_base_type_t<T, int>>(x);
}

// float ldexp (float x, int k)
// double ldexp (double x, int k)
// half ldexp (half x, int k)
template <typename T>
std::enable_if_t<detail::is_sgenfloat<T>::value, T> ldexp(T x, int k) __NOEXC {
  return __sycl_std::__invoke_ldexp<T>(x, k);
}

// vgenfloat ldexp (vgenfloat x, int k)
template <typename T>
std::enable_if_t<detail::is_vgenfloat<T>::value, T> ldexp(T x, int k) __NOEXC {
  return __sycl_std::__invoke_ldexp<T>(x, vec<int, T::size()>(k));
}

// vgenfloat ldexp (vgenfloat x, genint k)
template <typename T, typename T2>
std::enable_if_t<detail::is_vgenfloat<T>::value && detail::is_intn<T2>::value,
                 T>
ldexp(T x, T2 k) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_ldexp<T>(x, k);
}

// svgenfloat lgamma (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> lgamma(T x) __NOEXC {
  return __sycl_std::__invoke_lgamma<T>(x);
}

// svgenfloat lgamma_r (svgenfloat x, genintptr signp)
template <typename T, typename T2>
std::enable_if_t<
    detail::is_svgenfloat<T>::value && detail::is_genintptr<T2>::value, T>
lgamma_r(T x, T2 signp) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_lgamma_r<T>(x, signp);
}

// svgenfloat log (svgenfloat x)
template <typename T>
std::enable_if_t<__FAST_MATH_GENFLOAT(T), T> log(T x) __NOEXC {
  return __sycl_std::__invoke_log<T>(x);
}

// svgenfloat log2 (svgenfloat x)
template <typename T>
std::enable_if_t<__FAST_MATH_GENFLOAT(T), T> log2(T x) __NOEXC {
  return __sycl_std::__invoke_log2<T>(x);
}

// svgenfloat log10 (svgenfloat x)
template <typename T>
std::enable_if_t<__FAST_MATH_GENFLOAT(T), T> log10(T x) __NOEXC {
  return __sycl_std::__invoke_log10<T>(x);
}

// svgenfloat log1p (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> log1p(T x) __NOEXC {
  return __sycl_std::__invoke_log1p<T>(x);
}

// svgenfloat logb (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> logb(T x) __NOEXC {
  return __sycl_std::__invoke_logb<T>(x);
}

// svgenfloat mad (svgenfloat a, svgenfloat b, svgenfloat c)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> mad(T a, T b,
                                                         T c) __NOEXC {
  return __sycl_std::__invoke_mad<T>(a, b, c);
}

// svgenfloat maxmag (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> maxmag(T x, T y) __NOEXC {
  return __sycl_std::__invoke_maxmag<T>(x, y);
}

// svgenfloat minmag (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> minmag(T x, T y) __NOEXC {
  return __sycl_std::__invoke_minmag<T>(x, y);
}

// svgenfloat modf (svgenfloat x, genfloatptr iptr)
template <typename T, typename T2>
std::enable_if_t<
    detail::is_svgenfloat<T>::value && detail::is_genfloatptr<T2>::value, T>
modf(T x, T2 iptr) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_modf<T>(x, iptr);
}

template <typename T,
          typename = std::enable_if_t<detail::is_nan_type<T>::value, T>>
detail::nan_return_t<T> nan(T nancode) __NOEXC {
  return __sycl_std::__invoke_nan<detail::nan_return_t<T>>(
      detail::convert_data_type<T, detail::nan_argument_base_t<T>>()(nancode));
}

// svgenfloat nextafter (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> nextafter(T x,
                                                               T y) __NOEXC {
  return __sycl_std::__invoke_nextafter<T>(x, y);
}

// svgenfloat pow (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> pow(T x, T y) __NOEXC {
  return __sycl_std::__invoke_pow<T>(x, y);
}

// svgenfloat pown (svgenfloat x, genint y)
template <typename T, typename T2>
std::enable_if_t<
    detail::is_svgenfloat<T>::value && detail::is_genint<T2>::value, T>
pown(T x, T2 y) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_pown<T>(x, y);
}

// svgenfloat powr (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<__FAST_MATH_GENFLOAT(T), T> powr(T x, T y) __NOEXC {
  return __sycl_std::__invoke_powr<T>(x, y);
}

// svgenfloat remainder (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> remainder(T x,
                                                               T y) __NOEXC {
  return __sycl_std::__invoke_remainder<T>(x, y);
}

// svgenfloat remquo (svgenfloat x, svgenfloat y, genintptr quo)
template <typename T, typename T2>
std::enable_if_t<
    detail::is_svgenfloat<T>::value && detail::is_genintptr<T2>::value, T>
remquo(T x, T y, T2 quo) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_remquo<T>(x, y, quo);
}

// svgenfloat rint (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> rint(T x) __NOEXC {
  return __sycl_std::__invoke_rint<T>(x);
}

// svgenfloat rootn (svgenfloat x, genint y)
template <typename T, typename T2>
std::enable_if_t<
    detail::is_svgenfloat<T>::value && detail::is_genint<T2>::value, T>
rootn(T x, T2 y) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_rootn<T>(x, y);
}

// svgenfloat round (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> round(T x) __NOEXC {
  return __sycl_std::__invoke_round<T>(x);
}

// svgenfloat rsqrt (svgenfloat x)
template <typename T>
std::enable_if_t<__FAST_MATH_GENFLOAT(T), T> rsqrt(T x) __NOEXC {
  return __sycl_std::__invoke_rsqrt<T>(x);
}

// svgenfloat sin (svgenfloat x)
template <typename T>
std::enable_if_t<__FAST_MATH_GENFLOAT(T), T> sin(T x) __NOEXC {
  return __sycl_std::__invoke_sin<T>(x);
}

// svgenfloat sincos (svgenfloat x, genfloatptr cosval)
template <typename T, typename T2>
std::enable_if_t<
    detail::is_svgenfloat<T>::value && detail::is_genfloatptr<T2>::value, T>
sincos(T x, T2 cosval) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_sincos<T>(x, cosval);
}

// svgenfloat sinh (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> sinh(T x) __NOEXC {
  return __sycl_std::__invoke_sinh<T>(x);
}

// svgenfloat sinpi (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> sinpi(T x) __NOEXC {
  return __sycl_std::__invoke_sinpi<T>(x);
}

// svgenfloat sqrt (svgenfloat x)
template <typename T>
std::enable_if_t<__FAST_MATH_GENFLOAT(T), T> sqrt(T x) __NOEXC {
  return __sycl_std::__invoke_sqrt<T>(x);
}

// svgenfloat tan (svgenfloat x)
template <typename T>
std::enable_if_t<__FAST_MATH_GENFLOAT(T), T> tan(T x) __NOEXC {
  return __sycl_std::__invoke_tan<T>(x);
}

// svgenfloat tanh (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> tanh(T x) __NOEXC {
  return __sycl_std::__invoke_tanh<T>(x);
}

// svgenfloat tanpi (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> tanpi(T x) __NOEXC {
  return __sycl_std::__invoke_tanpi<T>(x);
}

// svgenfloat tgamma (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> tgamma(T x) __NOEXC {
  return __sycl_std::__invoke_tgamma<T>(x);
}

// svgenfloat trunc (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> trunc(T x) __NOEXC {
  return __sycl_std::__invoke_trunc<T>(x);
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
      detail::is_svgenfloat<T>::value &&                                       \
          detail::is_genfloatptr_marray<T2, AddressSpace, IsDecorated>::value, \
      marray<T, N>>                                                            \
  NAME(marray<T, N> ARG1, multi_ptr<T2, AddressSpace, IsDecorated> ARG2)       \
      __NOEXC {                                                                \
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
      detail::is_svgenfloat<T>::value &&                                       \
          detail::is_genintptr_marray<T2, AddressSpace, IsDecorated>::value,   \
      marray<T, N>>                                                            \
  NAME(marray<T, N> ARG1, multi_ptr<T2, AddressSpace, IsDecorated> ARG2)       \
      __NOEXC {                                                                \
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
      detail::is_svgenfloat<T>::value &&                                       \
          detail::is_genintptr_marray<T2, AddressSpace, IsDecorated>::value,   \
      marray<T, N>>                                                            \
  NAME(marray<T, N> x, marray<T, N> y,                                         \
       multi_ptr<T2, AddressSpace, IsDecorated> quo) __NOEXC {                 \
    __SYCL_MARRAY_MATH_FUNCTION_W_GENPTR_ARG_OVERLOAD_IMPL(NAME, quo,          \
                                                           __VA_ARGS__)        \
  }

__SYCL_MARRAY_MATH_FUNCTION_REMQUO_OVERLOAD(remquo, x[j], y[j])

#undef __SYCL_MARRAY_MATH_FUNCTION_REMQUO_OVERLOAD

#undef __SYCL_MARRAY_MATH_FUNCTION_W_GENPTR_ARG_OVERLOAD_IMPL

template <typename T, size_t N>
std::enable_if_t<detail::is_nan_type<T>::value,
                 marray<detail::nan_return_t<T>, N>>
nan(marray<T, N> nancode) __NOEXC {
  marray<detail::nan_return_t<T>, N> res;
  for (int j = 0; j < N; j++) {
    res[j] = nan(nancode[j]);
  }
  return res;
}

/* --------------- 4.13.5 Common functions. ---------------------------------*/
// svgenfloat clamp (svgenfloat x, svgenfloat minval, svgenfloat maxval)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> clamp(T x, T minval,
                                                           T maxval) __NOEXC {
  return __sycl_std::__invoke_fclamp<T>(x, minval, maxval);
}

// vgenfloath clamp (vgenfloath x, half minval, half maxval)
// vgenfloatf clamp (vgenfloatf x, float minval, float maxval)
// vgenfloatd clamp (vgenfloatd x, double minval, double maxval)
template <typename T>
std::enable_if_t<detail::is_vgenfloat<T>::value, T>
clamp(T x, typename T::element_type minval,
      typename T::element_type maxval) __NOEXC {
  return __sycl_std::__invoke_fclamp<T>(x, T(minval), T(maxval));
}

// svgenfloat degrees (svgenfloat radians)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T>
degrees(T radians) __NOEXC {
  return __sycl_std::__invoke_degrees<T>(radians);
}

// svgenfloat abs (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> abs(T x) __NOEXC {
  return __sycl_std::__invoke_fabs<T>(x);
}

// svgenfloat max (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T>(max)(T x, T y) __NOEXC {
  return __sycl_std::__invoke_fmax_common<T>(x, y);
}

// vgenfloatf max (vgenfloatf x, float y)
// vgenfloatd max (vgenfloatd x, double y)
// vgenfloath max (vgenfloath x, half y)
template <typename T>
std::enable_if_t<detail::is_vgenfloat<T>::value, T>(max)(
    T x, typename T::element_type y) __NOEXC {
  return __sycl_std::__invoke_fmax_common<T>(x, T(y));
}

// svgenfloat min (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T>(min)(T x, T y) __NOEXC {
  return __sycl_std::__invoke_fmin_common<T>(x, y);
}

// vgenfloatf min (vgenfloatf x, float y)
// vgenfloatd min (vgenfloatd x, double y)
// vgenfloath min (vgenfloath x, half y)
template <typename T>
std::enable_if_t<detail::is_vgenfloat<T>::value, T>(min)(
    T x, typename T::element_type y) __NOEXC {
  return __sycl_std::__invoke_fmin_common<T>(x, T(y));
}

// svgenfloat mix (svgenfloat x, svgenfloat y, svgenfloat a)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> mix(T x, T y,
                                                         T a) __NOEXC {
  return __sycl_std::__invoke_mix<T>(x, y, a);
}

// vgenfloatf mix (vgenfloatf x, vgenfloatf y, float a)
// vgenfloatd mix (vgenfloatd x, vgenfloatd y, double a)
// vgenfloatd mix (vgenfloath x, vgenfloath y, half a)
template <typename T>
std::enable_if_t<detail::is_vgenfloat<T>::value, T>
mix(T x, T y, typename T::element_type a) __NOEXC {
  return __sycl_std::__invoke_mix<T>(x, y, T(a));
}

// svgenfloat radians (svgenfloat degrees)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T>
radians(T degrees) __NOEXC {
  return __sycl_std::__invoke_radians<T>(degrees);
}

// svgenfloat step (svgenfloat edge, svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> step(T edge, T x) __NOEXC {
  return __sycl_std::__invoke_step<T>(edge, x);
}

// vgenfloatf step (float edge, vgenfloatf x)
// vgenfloatd step (double edge, vgenfloatd x)
// vgenfloatd step (half edge, vgenfloath x)
template <typename T>
std::enable_if_t<detail::is_vgenfloat<T>::value, T>
step(typename T::element_type edge, T x) __NOEXC {
  return __sycl_std::__invoke_step<T>(T(edge), x);
}

// svgenfloat smoothstep (svgenfloat edge0, svgenfloat edge1, svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T>
smoothstep(T edge0, T edge1, T x) __NOEXC {
  return __sycl_std::__invoke_smoothstep<T>(edge0, edge1, x);
}

// vgenfloatf smoothstep (float edge0, float edge1, vgenfloatf x)
// vgenfloatd smoothstep (double edge0, double edge1, vgenfloatd x)
// vgenfloath smoothstep (half edge0, half edge1, vgenfloath x)
template <typename T>
std::enable_if_t<detail::is_vgenfloat<T>::value, T>
smoothstep(typename T::element_type edge0, typename T::element_type edge1,
           T x) __NOEXC {
  return __sycl_std::__invoke_smoothstep<T>(T(edge0), T(edge1), x);
}

// svgenfloat sign (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> sign(T x) __NOEXC {
  return __sycl_std::__invoke_sign<T>(x);
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
            typename = std::enable_if_t<detail::is_mgenfloat<T>::value>>       \
  T NAME(ARG) __NOEXC {                                                        \
    __SYCL_MARRAY_COMMON_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)             \
  }

__SYCL_MARRAY_COMMON_FUNCTION_UNOP_OVERLOAD(degrees, T radians, radians[i])
__SYCL_MARRAY_COMMON_FUNCTION_UNOP_OVERLOAD(radians, T degrees, degrees[i])
__SYCL_MARRAY_COMMON_FUNCTION_UNOP_OVERLOAD(sign, T x, x[i])

#undef __SYCL_MARRAY_COMMON_FUNCTION_UNOP_OVERLOAD

#define __SYCL_MARRAY_COMMON_FUNCTION_BINOP_OVERLOAD(NAME, ARG1, ARG2, ...)    \
  template <typename T,                                                        \
            typename = std::enable_if_t<detail::is_mgenfloat<T>::value>>       \
  T NAME(ARG1, ARG2) __NOEXC {                                                 \
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
            typename = std::enable_if_t<detail::is_mgenfloat<T>::value>>       \
  T NAME(ARG1, ARG2, ARG3) __NOEXC {                                           \
    __SYCL_MARRAY_COMMON_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)             \
  }

__SYCL_MARRAY_COMMON_FUNCTION_TEROP_OVERLOAD(clamp, T x, T minval, T maxval,
                                             x[i], minval[i], maxval[i])
__SYCL_MARRAY_COMMON_FUNCTION_TEROP_OVERLOAD(
    clamp, T x, detail::marray_element_t<T> minval,
    detail::marray_element_t<T> maxval, x[i], minval, maxval)
__SYCL_MARRAY_COMMON_FUNCTION_TEROP_OVERLOAD(mix, T x, T y, T a, x[i], y[i],
                                             a[i])
__SYCL_MARRAY_COMMON_FUNCTION_TEROP_OVERLOAD(mix, T x, T y,
                                             detail::marray_element_t<T> a,
                                             x[i], y[i], a)
__SYCL_MARRAY_COMMON_FUNCTION_TEROP_OVERLOAD(smoothstep, T edge0, T edge1, T x,
                                             edge0[i], edge1[i], x[i])
__SYCL_MARRAY_COMMON_FUNCTION_TEROP_OVERLOAD(
    smoothstep, detail::marray_element_t<T> edge0,
    detail::marray_element_t<T> edge1, T x, edge0, edge1, x[i])

#undef __SYCL_MARRAY_COMMON_FUNCTION_TEROP_OVERLOAD
#undef __SYCL_MARRAY_COMMON_FUNCTION_OVERLOAD_IMPL

/* --------------- 4.13.4 Integer functions. --------------------------------*/
// ugeninteger abs (geninteger x)
template <typename T>
std::enable_if_t<detail::is_ugeninteger<T>::value, T> abs(T x) __NOEXC {
  return __sycl_std::__invoke_u_abs<T>(x);
}

// igeninteger abs (geninteger x)
template <typename T>
std::enable_if_t<detail::is_igeninteger<T>::value, T> abs(T x) __NOEXC {
  auto res = __sycl_std::__invoke_s_abs<detail::make_unsigned_t<T>>(x);
  if constexpr (detail::is_vigeninteger<T>::value) {
    return res.template convert<detail::vector_element_t<T>>();
  } else
    return detail::make_signed_t<decltype(res)>(res);
}

// ugeninteger abs_diff (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_ugeninteger<T>::value, T> abs_diff(T x,
                                                               T y) __NOEXC {
  return __sycl_std::__invoke_u_abs_diff<T>(x, y);
}

// ugeninteger abs_diff (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_igeninteger<T>::value, detail::make_unsigned_t<T>>
abs_diff(T x, T y) __NOEXC {
  return __sycl_std::__invoke_s_abs_diff<detail::make_unsigned_t<T>>(x, y);
}

// geninteger add_sat (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_igeninteger<T>::value, T> add_sat(T x,
                                                              T y) __NOEXC {
  return __sycl_std::__invoke_s_add_sat<T>(x, y);
}

// geninteger add_sat (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_ugeninteger<T>::value, T> add_sat(T x,
                                                              T y) __NOEXC {
  return __sycl_std::__invoke_u_add_sat<T>(x, y);
}

// geninteger hadd (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_igeninteger<T>::value, T> hadd(T x, T y) __NOEXC {
  return __sycl_std::__invoke_s_hadd<T>(x, y);
}

// geninteger hadd (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_ugeninteger<T>::value, T> hadd(T x, T y) __NOEXC {
  return __sycl_std::__invoke_u_hadd<T>(x, y);
}

// geninteger rhadd (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_igeninteger<T>::value, T> rhadd(T x, T y) __NOEXC {
  return __sycl_std::__invoke_s_rhadd<T>(x, y);
}

// geninteger rhadd (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_ugeninteger<T>::value, T> rhadd(T x, T y) __NOEXC {
  return __sycl_std::__invoke_u_rhadd<T>(x, y);
}

// geninteger clamp (geninteger x, geninteger minval, geninteger maxval)
template <typename T>
std::enable_if_t<detail::is_igeninteger<T>::value, T> clamp(T x, T minval,
                                                            T maxval) __NOEXC {
  return __sycl_std::__invoke_s_clamp<T>(x, minval, maxval);
}

// geninteger clamp (geninteger x, geninteger minval, geninteger maxval)
template <typename T>
std::enable_if_t<detail::is_ugeninteger<T>::value, T> clamp(T x, T minval,
                                                            T maxval) __NOEXC {
  return __sycl_std::__invoke_u_clamp<T>(x, minval, maxval);
}

// geninteger clamp (geninteger x, sgeninteger minval, sgeninteger maxval)
template <typename T>
std::enable_if_t<detail::is_vigeninteger<T>::value, T>
clamp(T x, typename T::element_type minval,
      typename T::element_type maxval) __NOEXC {
  return __sycl_std::__invoke_s_clamp<T>(x, T(minval), T(maxval));
}

// geninteger clamp (geninteger x, sgeninteger minval, sgeninteger maxval)
template <typename T>
std::enable_if_t<detail::is_vugeninteger<T>::value, T>
clamp(T x, typename T::element_type minval,
      typename T::element_type maxval) __NOEXC {
  return __sycl_std::__invoke_u_clamp<T>(x, T(minval), T(maxval));
}

// geninteger clz (geninteger x)
template <typename T>
std::enable_if_t<detail::is_geninteger<T>::value, T> clz(T x) __NOEXC {
  return __sycl_std::__invoke_clz<T>(x);
}

// geninteger ctz (geninteger x)
template <typename T>
std::enable_if_t<detail::is_geninteger<T>::value, T> ctz(T x) __NOEXC {
  return __sycl_std::__invoke_ctz<T>(x);
}

// geninteger ctz (geninteger x) for calls with deprecated namespace
namespace ext::intel {
template <typename T>
__SYCL_DEPRECATED(
    "'sycl::ext::intel::ctz' is deprecated, use 'sycl::ctz' instead")
std::enable_if_t<sycl::detail::is_geninteger<T>::value, T> ctz(T x) __NOEXC {
  return sycl::ctz(x);
}
} // namespace ext::intel

namespace __SYCL2020_DEPRECATED("use 'ext::intel' instead") intel {
using namespace ext::intel;
}

// geninteger mad_hi (geninteger a, geninteger b, geninteger c)
template <typename T>
std::enable_if_t<detail::is_igeninteger<T>::value, T> mad_hi(T x, T y,
                                                             T z) __NOEXC {
  return __sycl_std::__invoke_s_mad_hi<T>(x, y, z);
}

// geninteger mad_hi (geninteger a, geninteger b, geninteger c)
template <typename T>
std::enable_if_t<detail::is_ugeninteger<T>::value, T> mad_hi(T x, T y,
                                                             T z) __NOEXC {
  return __sycl_std::__invoke_u_mad_hi<T>(x, y, z);
}

// geninteger mad_sat (geninteger a, geninteger b, geninteger c)
template <typename T>
std::enable_if_t<detail::is_igeninteger<T>::value, T> mad_sat(T a, T b,
                                                              T c) __NOEXC {
  return __sycl_std::__invoke_s_mad_sat<T>(a, b, c);
}

// geninteger mad_sat (geninteger a, geninteger b, geninteger c)
template <typename T>
std::enable_if_t<detail::is_ugeninteger<T>::value, T> mad_sat(T a, T b,
                                                              T c) __NOEXC {
  return __sycl_std::__invoke_u_mad_sat<T>(a, b, c);
}

// igeninteger max (igeninteger x, igeninteger y)
template <typename T>
std::enable_if_t<detail::is_igeninteger<T>::value, T>(max)(T x, T y) __NOEXC {
  return __sycl_std::__invoke_s_max<T>(x, y);
}

// ugeninteger max (ugeninteger x, ugeninteger y)
template <typename T>
std::enable_if_t<detail::is_ugeninteger<T>::value, T>(max)(T x, T y) __NOEXC {
  return __sycl_std::__invoke_u_max<T>(x, y);
}

// igeninteger max (vigeninteger x, sigeninteger y)
template <typename T>
std::enable_if_t<detail::is_vigeninteger<T>::value, T>(max)(
    T x, typename T::element_type y) __NOEXC {
  return __sycl_std::__invoke_s_max<T>(x, T(y));
}

// vugeninteger max (vugeninteger x, sugeninteger y)
template <typename T>
std::enable_if_t<detail::is_vugeninteger<T>::value, T>(max)(
    T x, typename T::element_type y) __NOEXC {
  return __sycl_std::__invoke_u_max<T>(x, T(y));
}

// igeninteger min (igeninteger x, igeninteger y)
template <typename T>
std::enable_if_t<detail::is_igeninteger<T>::value, T>(min)(T x, T y) __NOEXC {
  return __sycl_std::__invoke_s_min<T>(x, y);
}

// ugeninteger min (ugeninteger x, ugeninteger y)
template <typename T>
std::enable_if_t<detail::is_ugeninteger<T>::value, T>(min)(T x, T y) __NOEXC {
  return __sycl_std::__invoke_u_min<T>(x, y);
}

// vigeninteger min (vigeninteger x, sigeninteger y)
template <typename T>
std::enable_if_t<detail::is_vigeninteger<T>::value, T>(min)(
    T x, typename T::element_type y) __NOEXC {
  return __sycl_std::__invoke_s_min<T>(x, T(y));
}

// vugeninteger min (vugeninteger x, sugeninteger y)
template <typename T>
std::enable_if_t<detail::is_vugeninteger<T>::value, T>(min)(
    T x, typename T::element_type y) __NOEXC {
  return __sycl_std::__invoke_u_min<T>(x, T(y));
}

// geninteger mul_hi (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_igeninteger<T>::value, T> mul_hi(T x, T y) __NOEXC {
  return __sycl_std::__invoke_s_mul_hi<T>(x, y);
}

// geninteger mul_hi (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_ugeninteger<T>::value, T> mul_hi(T x, T y) __NOEXC {
  return __sycl_std::__invoke_u_mul_hi<T>(x, y);
}

// geninteger rotate (geninteger v, geninteger i)
template <typename T>
std::enable_if_t<detail::is_geninteger<T>::value, T> rotate(T v, T i) __NOEXC {
  return __sycl_std::__invoke_rotate<T>(v, i);
}

// geninteger sub_sat (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_igeninteger<T>::value, T> sub_sat(T x,
                                                              T y) __NOEXC {
  return __sycl_std::__invoke_s_sub_sat<T>(x, y);
}

// geninteger sub_sat (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_ugeninteger<T>::value, T> sub_sat(T x,
                                                              T y) __NOEXC {
  return __sycl_std::__invoke_u_sub_sat<T>(x, y);
}

// ugeninteger16bit upsample (ugeninteger8bit hi, ugeninteger8bit lo)
template <typename T>
std::enable_if_t<detail::is_ugeninteger8bit<T>::value, detail::make_larger_t<T>>
upsample(T hi, T lo) __NOEXC {
  return __sycl_std::__invoke_u_upsample<detail::make_larger_t<T>>(hi, lo);
}

// igeninteger16bit upsample (igeninteger8bit hi, ugeninteger8bit lo)
template <typename T, typename T2>
std::enable_if_t<detail::is_igeninteger8bit<T>::value &&
                     detail::is_ugeninteger8bit<T2>::value,
                 detail::make_larger_t<T>>
upsample(T hi, T2 lo) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_s_upsample<detail::make_larger_t<T>>(hi, lo);
}

// ugeninteger32bit upsample (ugeninteger16bit hi, ugeninteger16bit lo)
template <typename T>
std::enable_if_t<detail::is_ugeninteger16bit<T>::value,
                 detail::make_larger_t<T>>
upsample(T hi, T lo) __NOEXC {
  return __sycl_std::__invoke_u_upsample<detail::make_larger_t<T>>(hi, lo);
}

// igeninteger32bit upsample (igeninteger16bit hi, ugeninteger16bit lo)
template <typename T, typename T2>
std::enable_if_t<detail::is_igeninteger16bit<T>::value &&
                     detail::is_ugeninteger16bit<T2>::value,
                 detail::make_larger_t<T>>
upsample(T hi, T2 lo) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_s_upsample<detail::make_larger_t<T>>(hi, lo);
}

// ugeninteger64bit upsample (ugeninteger32bit hi, ugeninteger32bit lo)
template <typename T>
std::enable_if_t<detail::is_ugeninteger32bit<T>::value,
                 detail::make_larger_t<T>>
upsample(T hi, T lo) __NOEXC {
  return __sycl_std::__invoke_u_upsample<detail::make_larger_t<T>>(hi, lo);
}

// igeninteger64bit upsample (igeninteger32bit hi, ugeninteger32bit lo)
template <typename T, typename T2>
std::enable_if_t<detail::is_igeninteger32bit<T>::value &&
                     detail::is_ugeninteger32bit<T2>::value,
                 detail::make_larger_t<T>>
upsample(T hi, T2 lo) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_s_upsample<detail::make_larger_t<T>>(hi, lo);
}

// geninteger popcount (geninteger x)
template <typename T>
std::enable_if_t<detail::is_geninteger<T>::value, T> popcount(T x) __NOEXC {
  return __sycl_std::__invoke_popcount<T>(x);
}

// geninteger32bit mad24 (geninteger32bit x, geninteger32bit y,
// geninteger32bit z)
template <typename T>
std::enable_if_t<detail::is_igeninteger32bit<T>::value, T> mad24(T x, T y,
                                                                 T z) __NOEXC {
  return __sycl_std::__invoke_s_mad24<T>(x, y, z);
}

// geninteger32bit mad24 (geninteger32bit x, geninteger32bit y,
// geninteger32bit z)
template <typename T>
std::enable_if_t<detail::is_ugeninteger32bit<T>::value, T> mad24(T x, T y,
                                                                 T z) __NOEXC {
  return __sycl_std::__invoke_u_mad24<T>(x, y, z);
}

// geninteger32bit mul24 (geninteger32bit x, geninteger32bit y)
template <typename T>
std::enable_if_t<detail::is_igeninteger32bit<T>::value, T> mul24(T x,
                                                                 T y) __NOEXC {
  return __sycl_std::__invoke_s_mul24<T>(x, y);
}

// geninteger32bit mul24 (geninteger32bit x, geninteger32bit y)
template <typename T>
std::enable_if_t<detail::is_ugeninteger32bit<T>::value, T> mul24(T x,
                                                                 T y) __NOEXC {
  return __sycl_std::__invoke_u_mul24<T>(x, y);
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
  std::enable_if_t<detail::is_ugeninteger<T>::value, marray<T, N>> NAME(       \
      marray<T, N> ARG) __NOEXC {                                              \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_ABS_I_OVERLOAD(NAME, ARG, ...)          \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_igeninteger<T>::value, marray<T, N>> NAME(       \
      marray<T, N> ARG) __NOEXC {                                              \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

__SYCL_MARRAY_INTEGER_FUNCTION_ABS_U_OVERLOAD(abs, x, x[j])
__SYCL_MARRAY_INTEGER_FUNCTION_ABS_I_OVERLOAD(abs, x, x[j])

#undef __SYCL_MARRAY_INTEGER_FUNCTION_ABS_I_OVERLOAD
#undef __SYCL_MARRAY_INTEGER_FUNCTION_ABS_U_OVERLOAD

#define __SYCL_MARRAY_INTEGER_FUNCTION_UNOP_OVERLOAD(NAME, ARG, ...)           \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_geninteger<T>::value, marray<T, N>> NAME(        \
      marray<T, N> ARG) __NOEXC {                                              \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

__SYCL_MARRAY_INTEGER_FUNCTION_UNOP_OVERLOAD(clz, x, x[j])
__SYCL_MARRAY_INTEGER_FUNCTION_UNOP_OVERLOAD(ctz, x, x[j])
__SYCL_MARRAY_INTEGER_FUNCTION_UNOP_OVERLOAD(popcount, x, x[j])

#undef __SYCL_MARRAY_INTEGER_FUNCTION_UNOP_OVERLOAD

#define __SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_OVERLOAD(NAME, ARG1, ARG2, ...) \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_ugeninteger<T>::value, marray<T, N>> NAME(       \
      marray<T, N> ARG1, marray<T, N> ARG2) __NOEXC {                          \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_RET_U_OVERLOAD(NAME, ARG1,      \
                                                              ARG2, ...)       \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_igeninteger<T>::value,                           \
                   marray<detail::make_unsigned_t<T>, N>>                      \
  NAME(marray<T, N> ARG1, marray<T, N> ARG2) __NOEXC {                         \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_OVERLOAD(NAME, ARG1, ARG2, ...) \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_igeninteger<T>::value, marray<T, N>> NAME(       \
      marray<T, N> ARG1, marray<T, N> ARG2) __NOEXC {                          \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_2ND_ARG_SCALAR_OVERLOAD(        \
    NAME, ARG1, ARG2, ...)                                                     \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_ugeninteger<T>::value, marray<T, N>> NAME(       \
      marray<T, N> ARG1, T ARG2) __NOEXC {                                     \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_2ND_ARG_SCALAR_OVERLOAD(        \
    NAME, ARG1, ARG2, ...)                                                     \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_igeninteger<T>::value, marray<T, N>> NAME(       \
      marray<T, N> ARG1, T ARG2) __NOEXC {                                     \
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
  std::enable_if_t<detail::is_ugeninteger<T>::value, marray<T, N>> NAME(       \
      marray<T, N> ARG1, marray<T, N> ARG2, marray<T, N> ARG3) __NOEXC {       \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_TEROP_I_OVERLOAD(NAME, ARG1, ARG2,      \
                                                        ARG3, ...)             \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_igeninteger<T>::value, marray<T, N>> NAME(       \
      marray<T, N> ARG1, marray<T, N> ARG2, marray<T, N> ARG3) __NOEXC {       \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_TEROP_U_2ND_3RD_ARGS_SCALAR_OVERLOAD(   \
    NAME, ARG1, ARG2, ARG3, ...)                                               \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_ugeninteger<T>::value, marray<T, N>> NAME(       \
      marray<T, N> ARG1, T ARG2, T ARG3) __NOEXC {                             \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_TEROP_I_2ND_3RD_ARGS_SCALAR_OVERLOAD(   \
    NAME, ARG1, ARG2, ARG3, ...)                                               \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_igeninteger<T>::value, marray<T, N>> NAME(       \
      marray<T, N> ARG1, T ARG2, T ARG3) __NOEXC {                             \
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
  std::enable_if_t<detail::is_ugeninteger32bit<T>::value, marray<T, N>> NAME(  \
      marray<T, N> ARG1, marray<T, N> ARG2, marray<T, N> ARG3) __NOEXC {       \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_MAD24_I_OVERLOAD(NAME, ARG1, ARG2,      \
                                                        ARG3, ...)             \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_igeninteger32bit<T>::value, marray<T, N>> NAME(  \
      marray<T, N> ARG1, marray<T, N> ARG2, marray<T, N> ARG3) __NOEXC {       \
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
  std::enable_if_t<detail::is_ugeninteger32bit<T>::value, marray<T, N>> NAME(  \
      marray<T, N> ARG1, marray<T, N> ARG2) __NOEXC {                          \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_MUL24_I_OVERLOAD(NAME, ARG1, ARG2, ...) \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_igeninteger32bit<T>::value, marray<T, N>> NAME(  \
      marray<T, N> ARG1, marray<T, N> ARG2) __NOEXC {                          \
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
  std::enable_if_t<detail::is_ugeninteger##KBIT<T>::value,                     \
                   detail::make_larger_t<marray<T, N>>>                        \
  NAME(marray<T, N> hi, marray<T, N> lo) __NOEXC {                             \
    __SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_OVERLOAD_IMPL(NAME)                \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_IU_OVERLOAD(NAME, KBIT)        \
  template <typename T, typename T2, size_t N>                                 \
  std::enable_if_t<detail::is_igeninteger##KBIT<T>::value &&                   \
                       detail::is_ugeninteger##KBIT<T2>::value,                \
                   detail::make_larger_t<marray<T, N>>>                        \
  NAME(marray<T, N> hi, marray<T2, N> lo) __NOEXC {                            \
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
std::enable_if_t<detail::is_gencross<T>::value, T> cross(T p0, T p1) __NOEXC {
  return __sycl_std::__invoke_cross<T>(p0, p1);
}

// float dot (float p0, float p1)
// double dot (double p0, double p1)
// half dot (half p0, half p1)
template <typename T>
std::enable_if_t<detail::is_sgenfloat<T>::value, T> dot(T p0, T p1) __NOEXC {
  return p0 * p1;
}

// float dot (vgengeofloat p0, vgengeofloat p1)
template <typename T>
std::enable_if_t<detail::is_vgengeofloat<T>::value, float> dot(T p0,
                                                               T p1) __NOEXC {
  return __sycl_std::__invoke_Dot<float>(p0, p1);
}

// double dot (vgengeodouble p0, vgengeodouble p1)
template <typename T>
std::enable_if_t<detail::is_vgengeodouble<T>::value, double> dot(T p0,
                                                                 T p1) __NOEXC {
  return __sycl_std::__invoke_Dot<double>(p0, p1);
}

// half dot (vgengeohalf p0, vgengeohalf p1)
template <typename T>
std::enable_if_t<detail::is_vgengeohalf<T>::value, half> dot(T p0,
                                                             T p1) __NOEXC {
  return __sycl_std::__invoke_Dot<half>(p0, p1);
}

// float distance (gengeofloat p0, gengeofloat p1)
template <typename T,
          typename = std::enable_if_t<detail::is_gengeofloat<T>::value, T>>
float distance(T p0, T p1) __NOEXC {
  return __sycl_std::__invoke_distance<float>(p0, p1);
}

// double distance (gengeodouble p0, gengeodouble p1)
template <typename T,
          typename = std::enable_if_t<detail::is_gengeodouble<T>::value, T>>
double distance(T p0, T p1) __NOEXC {
  return __sycl_std::__invoke_distance<double>(p0, p1);
}

// half distance (gengeohalf p0, gengeohalf p1)
template <typename T,
          typename = std::enable_if_t<detail::is_gengeohalf<T>::value, T>>
half distance(T p0, T p1) __NOEXC {
  return __sycl_std::__invoke_distance<half>(p0, p1);
}

// float length (gengeofloat p)
template <typename T,
          typename = std::enable_if_t<detail::is_gengeofloat<T>::value, T>>
float length(T p) __NOEXC {
  return __sycl_std::__invoke_length<float>(p);
}

// double length (gengeodouble p)
template <typename T,
          typename = std::enable_if_t<detail::is_gengeodouble<T>::value, T>>
double length(T p) __NOEXC {
  return __sycl_std::__invoke_length<double>(p);
}

// half length (gengeohalf p)
template <typename T,
          typename = std::enable_if_t<detail::is_gengeohalf<T>::value, T>>
half length(T p) __NOEXC {
  return __sycl_std::__invoke_length<half>(p);
}

// gengeofloat normalize (gengeofloat p)
template <typename T>
std::enable_if_t<detail::is_gengeofloat<T>::value, T> normalize(T p) __NOEXC {
  return __sycl_std::__invoke_normalize<T>(p);
}

// gengeodouble normalize (gengeodouble p)
template <typename T>
std::enable_if_t<detail::is_gengeodouble<T>::value, T> normalize(T p) __NOEXC {
  return __sycl_std::__invoke_normalize<T>(p);
}

// gengeohalf normalize (gengeohalf p)
template <typename T>
std::enable_if_t<detail::is_gengeohalf<T>::value, T> normalize(T p) __NOEXC {
  return __sycl_std::__invoke_normalize<T>(p);
}

// float fast_distance (gengeofloat p0, gengeofloat p1)
template <typename T,
          typename = std::enable_if_t<detail::is_gengeofloat<T>::value, T>>
float fast_distance(T p0, T p1) __NOEXC {
  return __sycl_std::__invoke_fast_distance<float>(p0, p1);
}

// double fast_distance (gengeodouble p0, gengeodouble p1)
template <typename T,
          typename = std::enable_if_t<detail::is_gengeodouble<T>::value, T>>
double fast_distance(T p0, T p1) __NOEXC {
  return __sycl_std::__invoke_fast_distance<double>(p0, p1);
}

// float fast_length (gengeofloat p)
template <typename T,
          typename = std::enable_if_t<detail::is_gengeofloat<T>::value, T>>
float fast_length(T p) __NOEXC {
  return __sycl_std::__invoke_fast_length<float>(p);
}

// double fast_length (gengeodouble p)
template <typename T,
          typename = std::enable_if_t<detail::is_gengeodouble<T>::value, T>>
double fast_length(T p) __NOEXC {
  return __sycl_std::__invoke_fast_length<double>(p);
}

// gengeofloat fast_normalize (gengeofloat p)
template <typename T>
std::enable_if_t<detail::is_gengeofloat<T>::value, T>
fast_normalize(T p) __NOEXC {
  return __sycl_std::__invoke_fast_normalize<T>(p);
}

// gengeodouble fast_normalize (gengeodouble p)
template <typename T>
std::enable_if_t<detail::is_gengeodouble<T>::value, T>
fast_normalize(T p) __NOEXC {
  return __sycl_std::__invoke_fast_normalize<T>(p);
}

// marray geometric functions

#define __SYCL_MARRAY_GEOMETRIC_FUNCTION_OVERLOAD_IMPL(NAME, ...)              \
  vec<detail::marray_element_t<T>, T::size()> result_v;                        \
  result_v = NAME(__VA_ARGS__);                                                \
  return detail::to_marray(result_v);

template <typename T>
std::enable_if_t<detail::is_gencrossmarray<T>::value, T> cross(T p0,
                                                               T p1) __NOEXC {
  __SYCL_MARRAY_GEOMETRIC_FUNCTION_OVERLOAD_IMPL(cross, detail::to_vec(p0),
                                                 detail::to_vec(p1))
}

template <typename T>
std::enable_if_t<detail::is_gengeomarray<T>::value, T> normalize(T p) __NOEXC {
  __SYCL_MARRAY_GEOMETRIC_FUNCTION_OVERLOAD_IMPL(normalize, detail::to_vec(p))
}

template <typename T>
std::enable_if_t<detail::is_gengeomarrayfloat<T>::value, T>
fast_normalize(T p) __NOEXC {
  __SYCL_MARRAY_GEOMETRIC_FUNCTION_OVERLOAD_IMPL(fast_normalize,
                                                 detail::to_vec(p))
}

#undef __SYCL_MARRAY_GEOMETRIC_FUNCTION_OVERLOAD_IMPL

#define __SYCL_MARRAY_GEOMETRIC_FUNCTION_IS_GENGEOMARRAY_BINOP_OVERLOAD(NAME)  \
  template <typename T>                                                        \
  std::enable_if_t<detail::is_gengeomarray<T>::value,                          \
                   detail::marray_element_t<T>>                                \
  NAME(T p0, T p1) __NOEXC {                                                   \
    return NAME(detail::to_vec(p0), detail::to_vec(p1));                       \
  }

// clang-format off
__SYCL_MARRAY_GEOMETRIC_FUNCTION_IS_GENGEOMARRAY_BINOP_OVERLOAD(dot)
__SYCL_MARRAY_GEOMETRIC_FUNCTION_IS_GENGEOMARRAY_BINOP_OVERLOAD(distance)
// clang-format on

#undef __SYCL_MARRAY_GEOMETRIC_FUNCTION_IS_GENGEOMARRAY_BINOP_OVERLOAD

template <typename T>
std::enable_if_t<detail::is_gengeomarray<T>::value, detail::marray_element_t<T>>
length(T p) __NOEXC {
  return __sycl_std::__invoke_length<detail::marray_element_t<T>>(
      detail::to_vec(p));
}

template <typename T>
std::enable_if_t<detail::is_gengeomarrayfloat<T>::value,
                 detail::marray_element_t<T>>
fast_distance(T p0, T p1) __NOEXC {
  return fast_distance(detail::to_vec(p0), detail::to_vec(p1));
}

template <typename T>
std::enable_if_t<detail::is_gengeomarrayfloat<T>::value,
                 detail::marray_element_t<T>>
fast_length(T p) __NOEXC {
  return fast_length(detail::to_vec(p));
}

/* SYCL 1.2.1 ---- 4.13.7 Relational functions. -----------------------------*/
/* SYCL 2020  ---- 4.17.9 Relational functions. -----------------------------*/

template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat<T>::value, T>>
detail::common_rel_ret_t<T> isequal(T x, T y) __NOEXC {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_FOrdEqual<detail::internal_rel_ret_t<T>>(x, y));
}

template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat<T>::value, T>>
detail::common_rel_ret_t<T> isnotequal(T x, T y) __NOEXC {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_FUnordNotEqual<detail::internal_rel_ret_t<T>>(x, y));
}

template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat<T>::value, T>>
detail::common_rel_ret_t<T> isgreater(T x, T y) __NOEXC {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_FOrdGreaterThan<detail::internal_rel_ret_t<T>>(x,
                                                                          y));
}

template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat<T>::value, T>>
detail::common_rel_ret_t<T> isgreaterequal(T x, T y) __NOEXC {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_FOrdGreaterThanEqual<detail::internal_rel_ret_t<T>>(
          x, y));
}

template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat<T>::value, T>>
detail::common_rel_ret_t<T> isless(T x, T y) __NOEXC {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_FOrdLessThan<detail::internal_rel_ret_t<T>>(x, y));
}

template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat<T>::value, T>>
detail::common_rel_ret_t<T> islessequal(T x, T y) __NOEXC {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_FOrdLessThanEqual<detail::internal_rel_ret_t<T>>(x,
                                                                            y));
}

template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat<T>::value, T>>
detail::common_rel_ret_t<T> islessgreater(T x, T y) __NOEXC {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_FOrdNotEqual<detail::internal_rel_ret_t<T>>(x, y));
}

template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat<T>::value, T>>
detail::common_rel_ret_t<T> isfinite(T x) __NOEXC {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_IsFinite<detail::internal_rel_ret_t<T>>(x));
}

template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat<T>::value, T>>
detail::common_rel_ret_t<T> isinf(T x) __NOEXC {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_IsInf<detail::internal_rel_ret_t<T>>(x));
}

template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat<T>::value, T>>
detail::common_rel_ret_t<T> isnan(T x) __NOEXC {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_IsNan<detail::internal_rel_ret_t<T>>(x));
}

template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat<T>::value, T>>
detail::common_rel_ret_t<T> isnormal(T x) __NOEXC {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_IsNormal<detail::internal_rel_ret_t<T>>(x));
}

template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat<T>::value, T>>
detail::common_rel_ret_t<T> isordered(T x, T y) __NOEXC {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_Ordered<detail::internal_rel_ret_t<T>>(x, y));
}

template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat<T>::value, T>>
detail::common_rel_ret_t<T> isunordered(T x, T y) __NOEXC {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_Unordered<detail::internal_rel_ret_t<T>>(x, y));
}

template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat<T>::value, T>>
detail::common_rel_ret_t<T> signbit(T x) __NOEXC {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_SignBitSet<detail::internal_rel_ret_t<T>>(x));
}

// marray relational functions

#define __SYCL_MARRAY_RELATIONAL_FUNCTION_BINOP_OVERLOAD(NAME)                 \
  template <typename T,                                                        \
            typename = std::enable_if_t<detail::is_mgenfloat<T>::value>>       \
  sycl::marray<bool, T::size()> NAME(T x, T y) __NOEXC {                       \
    sycl::marray<bool, T::size()> res;                                         \
    for (int i = 0; i < x.size(); i++) {                                       \
      res[i] = NAME(x[i], y[i]);                                               \
    }                                                                          \
    return res;                                                                \
  }

#define __SYCL_MARRAY_RELATIONAL_FUNCTION_UNOP_OVERLOAD(NAME)                  \
  template <typename T,                                                        \
            typename = std::enable_if_t<detail::is_mgenfloat<T>::value>>       \
  sycl::marray<bool, T::size()> NAME(T x) __NOEXC {                            \
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

// bool any (sigeninteger x)
template <typename T>
std::enable_if_t<detail::is_sigeninteger<T>::value, bool> any(T x) __NOEXC {
  return detail::Boolean<1>(int(detail::msbIsSet(x)));
}

// int any (vigeninteger x)
template <typename T>
std::enable_if_t<detail::is_vigeninteger<T>::value, int> any(T x) __NOEXC {
  return detail::rel_sign_bit_test_ret_t<T>(
      __sycl_std::__invoke_Any<detail::rel_sign_bit_test_ret_t<T>>(
          detail::rel_sign_bit_test_arg_t<T>(x)));
}

// bool all (sigeninteger x)
template <typename T>
std::enable_if_t<detail::is_sigeninteger<T>::value, bool> all(T x) __NOEXC {
  return detail::Boolean<1>(int(detail::msbIsSet(x)));
}

// int all (vigeninteger x)
template <typename T>
std::enable_if_t<detail::is_vigeninteger<T>::value, int> all(T x) __NOEXC {
  return detail::rel_sign_bit_test_ret_t<T>(
      __sycl_std::__invoke_All<detail::rel_sign_bit_test_ret_t<T>>(
          detail::rel_sign_bit_test_arg_t<T>(x)));
}

// gentype bitselect (gentype a, gentype b, gentype c)
template <typename T>
std::enable_if_t<detail::is_gentype<T>::value, T> bitselect(T a, T b,
                                                            T c) __NOEXC {
  return __sycl_std::__invoke_bitselect<T>(a, b, c);
}

// sgentype select (sgentype a, sgentype b, bool c)
template <typename T>
std::enable_if_t<detail::is_sgentype<T>::value, T> select(T a, T b,
                                                          bool c) __NOEXC {
  constexpr size_t SizeT = sizeof(T);

  // sycl::select(sgentype a, sgentype b, bool c) calls OpenCL built-in
  // select(sgentype a, sgentype b, igentype c). This type trait makes the
  // proper conversion for argument c from bool to igentype, based on sgentype
  // == T.
  using get_select_opencl_builtin_c_arg_type = typename std::conditional_t<
      SizeT == 1, char,
      std::conditional_t<
          SizeT == 2, short,
          std::conditional_t<
              (detail::is_contained<
                   T, detail::type_list<long, unsigned long>>::value &&
               (SizeT == 4 || SizeT == 8)),
              long, // long and ulong are 32-bit on
                    // Windows and 64-bit on Linux
              std::conditional_t<
                  SizeT == 4, int,
                  std::conditional_t<SizeT == 8, long long, void>>>>>;

  return __sycl_std::__invoke_select<T>(
      a, b, static_cast<get_select_opencl_builtin_c_arg_type>(c));
}

// geninteger select (geninteger a, geninteger b, igeninteger c)
template <typename T, typename T2>
std::enable_if_t<
    detail::is_geninteger<T>::value && detail::is_igeninteger<T2>::value, T>
select(T a, T b, T2 c) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_select<T>(a, b, c);
}

// geninteger select (geninteger a, geninteger b, ugeninteger c)
template <typename T, typename T2>
std::enable_if_t<
    detail::is_geninteger<T>::value && detail::is_ugeninteger<T2>::value, T>
select(T a, T b, T2 c) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_select<T>(a, b, c);
}

// svgenfloatf select (svgenfloatf a, svgenfloatf b, genint c)
template <typename T, typename T2>
std::enable_if_t<
    detail::is_svgenfloatf<T>::value && detail::is_genint<T2>::value, T>
select(T a, T b, T2 c) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_select<T>(a, b, c);
}

// svgenfloatf select (svgenfloatf a, svgenfloatf b, ugenint c)
template <typename T, typename T2>
std::enable_if_t<
    detail::is_svgenfloatf<T>::value && detail::is_ugenint<T2>::value, T>
select(T a, T b, T2 c) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_select<T>(a, b, c);
}

// svgenfloatd select (svgenfloatd a, svgenfloatd b, igeninteger64 c)
template <typename T, typename T2>
std::enable_if_t<detail::is_svgenfloatd<T>::value &&
                     detail::is_igeninteger64bit<T2>::value,
                 T>
select(T a, T b, T2 c) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_select<T>(a, b, c);
}

// svgenfloatd select (svgenfloatd a, svgenfloatd b, ugeninteger64 c)
template <typename T, typename T2>
std::enable_if_t<detail::is_svgenfloatd<T>::value &&
                     detail::is_ugeninteger64bit<T2>::value,
                 T>
select(T a, T b, T2 c) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_select<T>(a, b, c);
}

// svgenfloath select (svgenfloath a, svgenfloath b, igeninteger16 c)
template <typename T, typename T2>
std::enable_if_t<detail::is_svgenfloath<T>::value &&
                     detail::is_igeninteger16bit<T2>::value,
                 T>
select(T a, T b, T2 c) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_select<T>(a, b, c);
}

// svgenfloath select (svgenfloath a, svgenfloath b, ugeninteger16 c)
template <typename T, typename T2>
std::enable_if_t<detail::is_svgenfloath<T>::value &&
                     detail::is_ugeninteger16bit<T2>::value,
                 T>
select(T a, T b, T2 c) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_select<T>(a, b, c);
}

// other marray relational functions

template <typename T, size_t N>
std::enable_if_t<detail::is_sigeninteger<T>::value, bool>
any(marray<T, N> x) __NOEXC {
  return std::any_of(x.begin(), x.end(), [](T i) { return any(i); });
}

template <typename T, size_t N>
std::enable_if_t<detail::is_sigeninteger<T>::value, bool>
all(marray<T, N> x) __NOEXC {
  return std::all_of(x.begin(), x.end(), [](T i) { return all(i); });
}

template <typename T, size_t N>
std::enable_if_t<detail::is_gentype<T>::value, marray<T, N>>
bitselect(marray<T, N> a, marray<T, N> b, marray<T, N> c) __NOEXC {
  marray<T, N> res;
  for (int i = 0; i < N; i++) {
    res[i] = bitselect(a[i], b[i], c[i]);
  }
  return res;
}

template <typename T, size_t N>
std::enable_if_t<detail::is_gentype<T>::value, marray<T, N>>
select(marray<T, N> a, marray<T, N> b, marray<bool, N> c) __NOEXC {
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
  inline __SYCL_ALWAYS_INLINE marray<float, N> NAME(marray<float, N> x)        \
      __NOEXC {                                                                \
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
  inline __SYCL_ALWAYS_INLINE marray<float, N> NAME(                           \
      marray<float, N> x, marray<float, N> y) __NOEXC {                        \
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

// svgenfloatf cos (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> cos(T x) __NOEXC {
  return __sycl_std::__invoke_native_cos<T>(x);
}

// svgenfloatf divide (svgenfloatf x, svgenfloatf y)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> divide(T x, T y) __NOEXC {
  return __sycl_std::__invoke_native_divide<T>(x, y);
}

// svgenfloatf exp (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> exp(T x) __NOEXC {
  return __sycl_std::__invoke_native_exp<T>(x);
}

// svgenfloatf exp2 (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> exp2(T x) __NOEXC {
  return __sycl_std::__invoke_native_exp2<T>(x);
}

// svgenfloatf exp10 (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> exp10(T x) __NOEXC {
  return __sycl_std::__invoke_native_exp10<T>(x);
}

// svgenfloatf log (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> log(T x) __NOEXC {
  return __sycl_std::__invoke_native_log<T>(x);
}

// svgenfloatf log2 (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> log2(T x) __NOEXC {
  return __sycl_std::__invoke_native_log2<T>(x);
}

// svgenfloatf log10 (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> log10(T x) __NOEXC {
  return __sycl_std::__invoke_native_log10<T>(x);
}

// svgenfloatf powr (svgenfloatf x, svgenfloatf y)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> powr(T x, T y) __NOEXC {
  return __sycl_std::__invoke_native_powr<T>(x, y);
}

// svgenfloatf recip (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> recip(T x) __NOEXC {
  return __sycl_std::__invoke_native_recip<T>(x);
}

// svgenfloatf rsqrt (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> rsqrt(T x) __NOEXC {
  return __sycl_std::__invoke_native_rsqrt<T>(x);
}

// svgenfloatf sin (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> sin(T x) __NOEXC {
  return __sycl_std::__invoke_native_sin<T>(x);
}

// svgenfloatf sqrt (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> sqrt(T x) __NOEXC {
  return __sycl_std::__invoke_native_sqrt<T>(x);
}

// svgenfloatf tan (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> tan(T x) __NOEXC {
  return __sycl_std::__invoke_native_tan<T>(x);
}

} // namespace native
namespace half_precision {
/* ----------------- 4.13.3 Math functions. ---------------------------------*/
#define __SYCL_HALF_PRECISION_MATH_FUNCTION_OVERLOAD(NAME)                     \
  template <size_t N>                                                          \
  inline __SYCL_ALWAYS_INLINE marray<float, N> NAME(marray<float, N> x)        \
      __NOEXC {                                                                \
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
  inline __SYCL_ALWAYS_INLINE marray<float, N> NAME(                           \
      marray<float, N> x, marray<float, N> y) __NOEXC {                        \
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

// svgenfloatf cos (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> cos(T x) __NOEXC {
  return __sycl_std::__invoke_half_cos<T>(x);
}

// svgenfloatf divide (svgenfloatf x, svgenfloatf y)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> divide(T x, T y) __NOEXC {
  return __sycl_std::__invoke_half_divide<T>(x, y);
}

// svgenfloatf exp (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> exp(T x) __NOEXC {
  return __sycl_std::__invoke_half_exp<T>(x);
}

// svgenfloatf exp2 (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> exp2(T x) __NOEXC {
  return __sycl_std::__invoke_half_exp2<T>(x);
}

// svgenfloatf exp10 (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> exp10(T x) __NOEXC {
  return __sycl_std::__invoke_half_exp10<T>(x);
}

// svgenfloatf log (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> log(T x) __NOEXC {
  return __sycl_std::__invoke_half_log<T>(x);
}

// svgenfloatf log2 (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> log2(T x) __NOEXC {
  return __sycl_std::__invoke_half_log2<T>(x);
}

// svgenfloatf log10 (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> log10(T x) __NOEXC {
  return __sycl_std::__invoke_half_log10<T>(x);
}

// svgenfloatf powr (svgenfloatf x, svgenfloatf y)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> powr(T x, T y) __NOEXC {
  return __sycl_std::__invoke_half_powr<T>(x, y);
}

// svgenfloatf recip (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> recip(T x) __NOEXC {
  return __sycl_std::__invoke_half_recip<T>(x);
}

// svgenfloatf rsqrt (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> rsqrt(T x) __NOEXC {
  return __sycl_std::__invoke_half_rsqrt<T>(x);
}

// svgenfloatf sin (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> sin(T x) __NOEXC {
  return __sycl_std::__invoke_half_sin<T>(x);
}

// svgenfloatf sqrt (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> sqrt(T x) __NOEXC {
  return __sycl_std::__invoke_half_sqrt<T>(x);
}

// svgenfloatf tan (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> tan(T x) __NOEXC {
  return __sycl_std::__invoke_half_tan<T>(x);
}

} // namespace half_precision

#ifdef __FAST_MATH__
/* ----------------- -ffast-math functions. ---------------------------------*/

#define __SYCL_MATH_FUNCTION_OVERLOAD_FM(NAME)                                 \
  template <typename T, size_t N>                                              \
  inline __SYCL_ALWAYS_INLINE                                                  \
      std::enable_if_t<std::is_same_v<T, float>, marray<T, N>>                 \
      NAME(marray<T, N> x) __NOEXC {                                           \
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
    powr(marray<T, N> x, marray<T, N> y) __NOEXC {
  return native::powr(x, y);
}

// svgenfloatf cos (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> cos(T x) __NOEXC {
  return native::cos(x);
}

// svgenfloatf exp (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> exp(T x) __NOEXC {
  return native::exp(x);
}

// svgenfloatf exp2 (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> exp2(T x) __NOEXC {
  return native::exp2(x);
}

// svgenfloatf exp10 (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> exp10(T x) __NOEXC {
  return native::exp10(x);
}

// svgenfloatf log(svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> log(T x) __NOEXC {
  return native::log(x);
}

// svgenfloatf log2 (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> log2(T x) __NOEXC {
  return native::log2(x);
}

// svgenfloatf log10 (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> log10(T x) __NOEXC {
  return native::log10(x);
}

// svgenfloatf powr (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> powr(T x, T y) __NOEXC {
  return native::powr(x, y);
}

// svgenfloatf rsqrt (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> rsqrt(T x) __NOEXC {
  return native::rsqrt(x);
}

// svgenfloatf sin (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> sin(T x) __NOEXC {
  return native::sin(x);
}

// svgenfloatf sqrt (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> sqrt(T x) __NOEXC {
  return native::sqrt(x);
}

// svgenfloatf tan (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf<T>::value, T> tan(T x) __NOEXC {
  return native::tan(x);
}

#endif // __FAST_MATH__
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

#ifdef __SYCL_DEVICE_ONLY__
extern "C" {
extern __DPCPP_SYCL_EXTERNAL int abs(int x);
extern __DPCPP_SYCL_EXTERNAL long int labs(long int x);
extern __DPCPP_SYCL_EXTERNAL long long int llabs(long long int x);

extern __DPCPP_SYCL_EXTERNAL div_t div(int x, int y);
extern __DPCPP_SYCL_EXTERNAL ldiv_t ldiv(long int x, long int y);
extern __DPCPP_SYCL_EXTERNAL lldiv_t lldiv(long long int x, long long int y);
extern __DPCPP_SYCL_EXTERNAL float scalbnf(float x, int n);
extern __DPCPP_SYCL_EXTERNAL double scalbn(double x, int n);
extern __DPCPP_SYCL_EXTERNAL float logf(float x);
extern __DPCPP_SYCL_EXTERNAL double log(double x);
extern __DPCPP_SYCL_EXTERNAL float expf(float x);
extern __DPCPP_SYCL_EXTERNAL double exp(double x);
extern __DPCPP_SYCL_EXTERNAL float log10f(float x);
extern __DPCPP_SYCL_EXTERNAL double log10(double x);
extern __DPCPP_SYCL_EXTERNAL float modff(float x, float *intpart);
extern __DPCPP_SYCL_EXTERNAL double modf(double x, double *intpart);
extern __DPCPP_SYCL_EXTERNAL float exp2f(float x);
extern __DPCPP_SYCL_EXTERNAL double exp2(double x);
extern __DPCPP_SYCL_EXTERNAL float expm1f(float x);
extern __DPCPP_SYCL_EXTERNAL double expm1(double x);
extern __DPCPP_SYCL_EXTERNAL int ilogbf(float x);
extern __DPCPP_SYCL_EXTERNAL int ilogb(double x);
extern __DPCPP_SYCL_EXTERNAL float log1pf(float x);
extern __DPCPP_SYCL_EXTERNAL double log1p(double x);
extern __DPCPP_SYCL_EXTERNAL float log2f(float x);
extern __DPCPP_SYCL_EXTERNAL double log2(double x);
extern __DPCPP_SYCL_EXTERNAL float logbf(float x);
extern __DPCPP_SYCL_EXTERNAL double logb(double x);
extern __DPCPP_SYCL_EXTERNAL float sqrtf(float x);
extern __DPCPP_SYCL_EXTERNAL double sqrt(double x);
extern __DPCPP_SYCL_EXTERNAL float cbrtf(float x);
extern __DPCPP_SYCL_EXTERNAL double cbrt(double x);
extern __DPCPP_SYCL_EXTERNAL float erff(float x);
extern __DPCPP_SYCL_EXTERNAL double erf(double x);
extern __DPCPP_SYCL_EXTERNAL float erfcf(float x);
extern __DPCPP_SYCL_EXTERNAL double erfc(double x);
extern __DPCPP_SYCL_EXTERNAL float tgammaf(float x);
extern __DPCPP_SYCL_EXTERNAL double tgamma(double x);
extern __DPCPP_SYCL_EXTERNAL float lgammaf(float x);
extern __DPCPP_SYCL_EXTERNAL double lgamma(double x);
extern __DPCPP_SYCL_EXTERNAL float fmodf(float x, float y);
extern __DPCPP_SYCL_EXTERNAL double fmod(double x, double y);
extern __DPCPP_SYCL_EXTERNAL float remainderf(float x, float y);
extern __DPCPP_SYCL_EXTERNAL double remainder(double x, double y);
extern __DPCPP_SYCL_EXTERNAL float remquof(float x, float y, int *q);
extern __DPCPP_SYCL_EXTERNAL double remquo(double x, double y, int *q);
extern __DPCPP_SYCL_EXTERNAL float nextafterf(float x, float y);
extern __DPCPP_SYCL_EXTERNAL double nextafter(double x, double y);
extern __DPCPP_SYCL_EXTERNAL float fdimf(float x, float y);
extern __DPCPP_SYCL_EXTERNAL double fdim(double x, double y);
extern __DPCPP_SYCL_EXTERNAL float fmaf(float x, float y, float z);
extern __DPCPP_SYCL_EXTERNAL double fma(double x, double y, double z);
extern __DPCPP_SYCL_EXTERNAL float sinf(float x);
extern __DPCPP_SYCL_EXTERNAL double sin(double x);
extern __DPCPP_SYCL_EXTERNAL float cosf(float x);
extern __DPCPP_SYCL_EXTERNAL double cos(double x);
extern __DPCPP_SYCL_EXTERNAL float tanf(float x);
extern __DPCPP_SYCL_EXTERNAL double tan(double x);
extern __DPCPP_SYCL_EXTERNAL float asinf(float x);
extern __DPCPP_SYCL_EXTERNAL double asin(double x);
extern __DPCPP_SYCL_EXTERNAL float acosf(float x);
extern __DPCPP_SYCL_EXTERNAL double acos(double x);
extern __DPCPP_SYCL_EXTERNAL float atanf(float x);
extern __DPCPP_SYCL_EXTERNAL double atan(double x);
extern __DPCPP_SYCL_EXTERNAL float powf(float x, float y);
extern __DPCPP_SYCL_EXTERNAL double pow(double x, double y);
extern __DPCPP_SYCL_EXTERNAL float atan2f(float x, float y);
extern __DPCPP_SYCL_EXTERNAL double atan2(double x, double y);

extern __DPCPP_SYCL_EXTERNAL float sinhf(float x);
extern __DPCPP_SYCL_EXTERNAL double sinh(double x);
extern __DPCPP_SYCL_EXTERNAL float coshf(float x);
extern __DPCPP_SYCL_EXTERNAL double cosh(double x);
extern __DPCPP_SYCL_EXTERNAL float tanhf(float x);
extern __DPCPP_SYCL_EXTERNAL double tanh(double x);
extern __DPCPP_SYCL_EXTERNAL float asinhf(float x);
extern __DPCPP_SYCL_EXTERNAL double asinh(double x);
extern __DPCPP_SYCL_EXTERNAL float acoshf(float x);
extern __DPCPP_SYCL_EXTERNAL double acosh(double x);
extern __DPCPP_SYCL_EXTERNAL float atanhf(float x);
extern __DPCPP_SYCL_EXTERNAL double atanh(double x);
extern __DPCPP_SYCL_EXTERNAL double frexp(double x, int *exp);
extern __DPCPP_SYCL_EXTERNAL double ldexp(double x, int exp);
extern __DPCPP_SYCL_EXTERNAL double hypot(double x, double y);

extern __DPCPP_SYCL_EXTERNAL void *memcpy(void *dest, const void *src,
                                          size_t n);
extern __DPCPP_SYCL_EXTERNAL void *memset(void *dest, int c, size_t n);
extern __DPCPP_SYCL_EXTERNAL int memcmp(const void *s1, const void *s2,
                                        size_t n);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_llmax(long long int x,
                                                       long long int y);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_llmin(long long int x,
                                                       long long int y);
extern __DPCPP_SYCL_EXTERNAL int __imf_max(int x, int y);
extern __DPCPP_SYCL_EXTERNAL int __imf_min(int x, int y);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int
__imf_ullmax(unsigned long long int x, unsigned long long int y);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int
__imf_ullmin(unsigned long long int x, unsigned long long int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_umax(unsigned int x,
                                                     unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_umin(unsigned int x,
                                                     unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_brev(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int
__imf_brevll(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_byte_perm(unsigned int x, unsigned int y, unsigned int s);
extern __DPCPP_SYCL_EXTERNAL int __imf_ffs(int x);
extern __DPCPP_SYCL_EXTERNAL int __imf_ffsll(long long int x);
extern __DPCPP_SYCL_EXTERNAL int __imf_clz(int x);
extern __DPCPP_SYCL_EXTERNAL int __imf_clzll(long long int x);
extern __DPCPP_SYCL_EXTERNAL int __imf_popc(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL int __imf_popcll(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_sad(int x, int y,
                                                    unsigned int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_usad(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL int __imf_rhadd(int x, int y);
extern __DPCPP_SYCL_EXTERNAL int __imf_hadd(int x, int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_urhadd(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_uhadd(unsigned int x,
                                                      unsigned int y);
extern __DPCPP_SYCL_EXTERNAL int __imf_mul24(int x, int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_umul24(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL int __imf_mulhi(int x, int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_umulhi(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_mul64hi(long long int x,
                                                         long long int y);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int
__imf_umul64hi(unsigned long long int x, unsigned long long int y);
extern __DPCPP_SYCL_EXTERNAL int __imf_abs(int x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_llabs(long long int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_saturatef(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_fmaf(float x, float y, float z);
extern __DPCPP_SYCL_EXTERNAL float __imf_fabsf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_floorf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_ceilf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_truncf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_rintf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_nearbyintf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_sqrtf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_rsqrtf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_invf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_fmaxf(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fminf(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_copysignf(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fast_exp10f(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_fast_expf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_fast_logf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_fast_log2f(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_fast_log10f(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_fast_powf(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fast_fdividef(float x, float y);
extern __DPCPP_SYCL_EXTERNAL int __imf_float2int_rd(float x);
extern __DPCPP_SYCL_EXTERNAL int __imf_float2int_rn(float x);
extern __DPCPP_SYCL_EXTERNAL int __imf_float2int_ru(float x);
extern __DPCPP_SYCL_EXTERNAL int __imf_float2int_rz(float x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_float2uint_rd(float x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_float2uint_rn(float x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_float2uint_ru(float x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_float2uint_rz(float x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_float2ll_rd(float x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_float2ll_rn(float x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_float2ll_ru(float x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_float2ll_rz(float x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int __imf_float2ull_rd(float x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int __imf_float2ull_rn(float x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int __imf_float2ull_ru(float x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int __imf_float2ull_rz(float x);
extern __DPCPP_SYCL_EXTERNAL int __imf_float_as_int(float x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_float_as_uint(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_int2float_rd(int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_int2float_rn(int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_int2float_ru(int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_int2float_rz(int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_int_as_float(int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_ll2float_rd(long long int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_ll2float_rn(long long int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_ll2float_ru(long long int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_ll2float_rz(long long int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_uint2float_rd(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_uint2float_rn(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_uint2float_ru(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_uint2float_rz(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_uint_as_float(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_ull2float_rd(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_ull2float_rn(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_ull2float_ru(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_ull2float_rz(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_half2float(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_float2half_rd(float x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_float2half_rn(float x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_float2half_ru(float x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_float2half_rz(float x);
extern __DPCPP_SYCL_EXTERNAL int __imf_half2int_rd(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL int __imf_half2int_rn(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL int __imf_half2int_ru(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL int __imf_half2int_rz(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL long long __imf_half2ll_rd(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL long long __imf_half2ll_rn(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL long long __imf_half2ll_ru(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL long long __imf_half2ll_rz(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL short __imf_half2short_rd(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL short __imf_half2short_rn(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL short __imf_half2short_ru(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL short __imf_half2short_rz(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_half2uint_rd(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_half2uint_rn(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_half2uint_ru(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_half2uint_rz(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long __imf_half2ull_rd(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long __imf_half2ull_rn(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long __imf_half2ull_ru(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long __imf_half2ull_rz(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned short __imf_half2ushort_rd(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned short __imf_half2ushort_rn(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned short __imf_half2ushort_ru(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned short __imf_half2ushort_rz(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL short __imf_half_as_short(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned short __imf_half_as_ushort(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_int2half_rd(int x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_int2half_rn(int x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_int2half_ru(int x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_int2half_rz(int x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ll2half_rd(long long x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ll2half_rn(long long x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ll2half_ru(long long x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ll2half_rz(long long x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_short2half_rd(short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_short2half_rn(short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_short2half_ru(short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_short2half_rz(short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_short_as_half(short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_uint2half_rd(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_uint2half_rn(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_uint2half_ru(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_uint2half_rz(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ull2half_rd(unsigned long long x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ull2half_rn(unsigned long long x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ull2half_ru(unsigned long long x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ull2half_rz(unsigned long long x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ushort2half_rd(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ushort2half_rn(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ushort2half_ru(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ushort2half_rz(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ushort_as_half(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_double2half(double x);

extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_fmaf16(_Float16 x, _Float16 y,
                                                   _Float16 z);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_fabsf16(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_floorf16(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ceilf16(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_truncf16(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_rintf16(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_nearbyintf16(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_sqrtf16(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_rsqrtf16(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_invf16(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_fmaxf16(_Float16 x, _Float16 y);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_fminf16(_Float16 x, _Float16 y);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_copysignf16(_Float16 x, _Float16 y);
extern __DPCPP_SYCL_EXTERNAL float __imf_half2float(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL float __imf_bfloat162float(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_bfloat162uint_rd(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_bfloat162uint_rn(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_bfloat162uint_ru(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_bfloat162uint_rz(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned short
__imf_bfloat162ushort_rd(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned short
__imf_bfloat162ushort_rn(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned short
__imf_bfloat162ushort_ru(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned short
__imf_bfloat162ushort_rz(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long
__imf_bfloat162ull_rd(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long
__imf_bfloat162ull_rn(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long
__imf_bfloat162ull_ru(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long
__imf_bfloat162ull_rz(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL int __imf_bfloat162int_rd(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL int __imf_bfloat162int_rn(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL int __imf_bfloat162int_ru(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL int __imf_bfloat162int_rz(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL short __imf_bfloat162short_rd(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL short __imf_bfloat162short_rn(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL short __imf_bfloat162short_ru(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL short __imf_bfloat162short_rz(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL long long __imf_bfloat162ll_rd(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL long long __imf_bfloat162ll_rn(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL long long __imf_bfloat162ll_ru(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL long long __imf_bfloat162ll_rz(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_float2bfloat16(float x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_float2bfloat16_rd(float x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_float2bfloat16_rn(float x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_float2bfloat16_ru(float x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_float2bfloat16_rz(float x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ushort2bfloat16_rd(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ushort2bfloat16_rn(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ushort2bfloat16_ru(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ushort2bfloat16_rz(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_uint2bfloat16_rd(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_uint2bfloat16_rn(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_uint2bfloat16_ru(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_uint2bfloat16_rz(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ull2bfloat16_rd(unsigned long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ull2bfloat16_rn(unsigned long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ull2bfloat16_ru(unsigned long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ull2bfloat16_rz(unsigned long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_short2bfloat16_rd(short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_short2bfloat16_rn(short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_short2bfloat16_ru(short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_short2bfloat16_rz(short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_int2bfloat16_rd(int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_int2bfloat16_rn(int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_int2bfloat16_ru(int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_int2bfloat16_rz(int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_ll2bfloat16_rd(long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_ll2bfloat16_rn(long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_ll2bfloat16_ru(long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_ll2bfloat16_rz(long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_double2bfloat16(double x);
extern __DPCPP_SYCL_EXTERNAL short __imf_bfloat16_as_short(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned short
__imf_bfloat16_as_ushort(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_short_as_bfloat16(short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ushort_as_bfloat16(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_fmabf16(uint16_t x, uint16_t y,
                                                    uint16_t z);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_fmaxbf16(uint16_t x, uint16_t y);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_fminbf16(uint16_t x, uint16_t y);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_fabsbf16(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_rintbf16(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_floorbf16(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_ceilbf16(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_truncbf16(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_copysignbf16(uint16_t x,
                                                         uint16_t y);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_sqrtbf16(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_rsqrtbf16(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL double __imf_fma(double x, double y, double z);
extern __DPCPP_SYCL_EXTERNAL double __imf_fabs(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_floor(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_ceil(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_trunc(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_rint(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_nearbyint(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_sqrt(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_rsqrt(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_inv(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_fmax(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_fmin(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_copysign(double x, double y);
extern __DPCPP_SYCL_EXTERNAL float __imf_double2float_rd(double x);
extern __DPCPP_SYCL_EXTERNAL float __imf_double2float_rn(double x);
extern __DPCPP_SYCL_EXTERNAL float __imf_double2float_ru(double x);
extern __DPCPP_SYCL_EXTERNAL float __imf_double2float_rz(double x);
extern __DPCPP_SYCL_EXTERNAL int __imf_double2hiint(double x);
extern __DPCPP_SYCL_EXTERNAL int __imf_double2loint(double x);
extern __DPCPP_SYCL_EXTERNAL int __imf_double2int_rd(double x);
extern __DPCPP_SYCL_EXTERNAL int __imf_double2int_rn(double x);
extern __DPCPP_SYCL_EXTERNAL int __imf_double2int_ru(double x);
extern __DPCPP_SYCL_EXTERNAL int __imf_double2int_rz(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_int2double_rn(int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_double2uint_rd(double x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_double2uint_rn(double x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_double2uint_ru(double x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_double2uint_rz(double x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_double2ll_rd(double x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_double2ll_rn(double x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_double2ll_ru(double x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_double2ll_rz(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_ll2double_rd(long long int x);
extern __DPCPP_SYCL_EXTERNAL double __imf_ll2double_rn(long long int x);
extern __DPCPP_SYCL_EXTERNAL double __imf_ll2double_ru(long long int x);
extern __DPCPP_SYCL_EXTERNAL double __imf_ll2double_rz(long long int x);
extern __DPCPP_SYCL_EXTERNAL double
__imf_ull2double_rd(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL double
__imf_ull2double_rn(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL double
__imf_ull2double_ru(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL double
__imf_ull2double_rz(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int
__imf_double2ull_rd(double x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int
__imf_double2ull_rn(double x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int
__imf_double2ull_ru(double x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int
__imf_double2ull_rz(double x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_double_as_longlong(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_longlong_as_double(long long int x);
extern __DPCPP_SYCL_EXTERNAL double __imf_uint2double_rd(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL double __imf_uint2double_rn(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL double __imf_uint2double_ru(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL double __imf_uint2double_rz(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL double __imf_hiloint2double(int hi, int lo);

extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabs2(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabs4(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabsss2(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabsss4(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vneg2(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vneg4(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vnegss2(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vnegss4(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabsdiffs2(unsigned int x,
                                                           unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabsdiffs4(unsigned int x,
                                                           unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabsdiffu2(unsigned int x,
                                                           unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabsdiffu4(unsigned int x,
                                                           unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vadd2(unsigned int x,
                                                      unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vadd4(unsigned int x,
                                                      unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vaddss2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vaddss4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vaddus2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vaddus4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsub2(unsigned int x,
                                                      unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsub4(unsigned int x,
                                                      unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsubss2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsubss4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsubus2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsubus4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vavgs2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vavgs4(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vavgu2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vavgu4(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vhaddu2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vhaddu4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpeq2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpeq4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpne2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpne4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpges2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpges4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpgeu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpgeu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpgts2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpgts4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpgtu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpgtu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmples2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmples4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpleu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpleu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmplts2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmplts4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpltu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpltu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vmaxs2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vmaxs4(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vmaxu2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vmaxu4(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vmins2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vmins4(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vminu2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vminu4(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vseteq2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vseteq4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetne2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetne4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetges2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetges4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetgeu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetgeu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetgts2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetgts4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetgtu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetgtu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetles2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetles4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetleu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetleu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetlts2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetlts4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetltu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetltu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsads2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsads4(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsadu2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsadu4(unsigned int x,
                                                       unsigned int y);
}
#ifdef __GLIBC__
extern "C" {
extern __DPCPP_SYCL_EXTERNAL void __assert_fail(const char *expr,
                                                const char *file,
                                                unsigned int line,
                                                const char *func);
extern __DPCPP_SYCL_EXTERNAL float frexpf(float x, int *exp);
extern __DPCPP_SYCL_EXTERNAL float ldexpf(float x, int exp);
extern __DPCPP_SYCL_EXTERNAL float hypotf(float x, float y);

// MS UCRT supports most of the C standard library but <complex.h> is
// an exception.
extern __DPCPP_SYCL_EXTERNAL float cimagf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double cimag(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float crealf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double creal(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float cargf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double carg(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float cabsf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double cabs(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ cprojf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ cproj(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ cexpf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ cexp(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ clogf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ clog(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ cpowf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ cpow(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ csqrtf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ csqrt(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ csinhf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ csinh(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ ccoshf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ ccosh(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ ctanhf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ ctanh(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ csinf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ csin(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ ccosf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ ccos(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ ctanf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ ctan(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ cacosf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ cacos(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ cacoshf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ cacosh(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ casinf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ casin(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ casinhf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ casinh(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ catanf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ catan(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ catanhf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ catanh(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ cpolarf(float rho, float theta);
extern __DPCPP_SYCL_EXTERNAL double __complex__ cpolar(double rho,
                                                       double theta);
extern __DPCPP_SYCL_EXTERNAL float __complex__ __mulsc3(float a, float b,
                                                        float c, float d);
extern __DPCPP_SYCL_EXTERNAL double __complex__ __muldc3(double a, double b,
                                                         double c, double d);
extern __DPCPP_SYCL_EXTERNAL float __complex__ __divsc3(float a, float b,
                                                        float c, float d);
extern __DPCPP_SYCL_EXTERNAL double __complex__ __divdc3(float a, float b,
                                                         float c, float d);
}
#elif defined(_WIN32)
extern "C" {
// TODO: documented C runtime library APIs must be recognized as
//       builtins by FE. This includes _dpcomp, _dsign, _dtest,
//       _fdpcomp, _fdsign, _fdtest, _hypotf, _wassert.
//       APIs used by STL, such as _Cosh, are undocumented, even though
//       they are open-sourced. Recognizing them as builtins is not
//       straightforward currently.
extern __DPCPP_SYCL_EXTERNAL double _Cosh(double x, double y);
extern __DPCPP_SYCL_EXTERNAL int _dpcomp(double x, double y);
extern __DPCPP_SYCL_EXTERNAL int _dsign(double x);
extern __DPCPP_SYCL_EXTERNAL short _Dtest(double *px);
extern __DPCPP_SYCL_EXTERNAL short _dtest(double *px);
extern __DPCPP_SYCL_EXTERNAL short _Exp(double *px, double y, short eoff);
extern __DPCPP_SYCL_EXTERNAL float _FCosh(float x, float y);
extern __DPCPP_SYCL_EXTERNAL int _fdpcomp(float x, float y);
extern __DPCPP_SYCL_EXTERNAL int _fdsign(float x);
extern __DPCPP_SYCL_EXTERNAL short _FDtest(float *px);
extern __DPCPP_SYCL_EXTERNAL short _fdtest(float *px);
extern __DPCPP_SYCL_EXTERNAL short _FExp(float *px, float y, short eoff);
extern __DPCPP_SYCL_EXTERNAL float _FSinh(float x, float y);
extern __DPCPP_SYCL_EXTERNAL double _Sinh(double x, double y);
extern __DPCPP_SYCL_EXTERNAL float _hypotf(float x, float y);
extern __DPCPP_SYCL_EXTERNAL void _wassert(const wchar_t *wexpr,
                                           const wchar_t *wfile, unsigned line);
}
#endif
#endif // __SYCL_DEVICE_ONLY__

#undef __NOEXC
