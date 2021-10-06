//==----------- builtins.hpp - SYCL built-in functions ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/boolean.hpp>
#include <CL/sycl/detail/builtins.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/generic_type_traits.hpp>
#include <CL/sycl/types.hpp>

// TODO Decide whether to mark functions with this attribute.
#define __NOEXC /*noexcept*/

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
#ifdef __SYCL_DEVICE_ONLY__
#define __sycl_std
#else
namespace __sycl_std = __host_std;
#endif

/* ----------------- 4.13.3 Math functions. ---------------------------------*/
// genfloat acos (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> acos(T x) __NOEXC {
  return __sycl_std::__invoke_acos<T>(x);
}

// genfloat acosh (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> acosh(T x) __NOEXC {
  return __sycl_std::__invoke_acosh<T>(x);
}

// genfloat acospi (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> acospi(T x) __NOEXC {
  return __sycl_std::__invoke_acospi<T>(x);
}

// genfloat asin (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> asin(T x) __NOEXC {
  return __sycl_std::__invoke_asin<T>(x);
}

// genfloat asinh (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> asinh(T x) __NOEXC {
  return __sycl_std::__invoke_asinh<T>(x);
}

// genfloat asinpi (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> asinpi(T x) __NOEXC {
  return __sycl_std::__invoke_asinpi<T>(x);
}

// genfloat atan (genfloat y_over_x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> atan(T y_over_x) __NOEXC {
  return __sycl_std::__invoke_atan<T>(y_over_x);
}

// genfloat atan2 (genfloat y, genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> atan2(T y, T x) __NOEXC {
  return __sycl_std::__invoke_atan2<T>(y, x);
}

// genfloat atanh (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> atanh(T x) __NOEXC {
  return __sycl_std::__invoke_atanh<T>(x);
}

// genfloat atanpi (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> atanpi(T x) __NOEXC {
  return __sycl_std::__invoke_atanpi<T>(x);
}

// genfloat atan2pi (genfloat y, genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> atan2pi(T y,
                                                              T x) __NOEXC {
  return __sycl_std::__invoke_atan2pi<T>(y, x);
}

// genfloat cbrt (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> cbrt(T x) __NOEXC {
  return __sycl_std::__invoke_cbrt<T>(x);
}

// genfloat ceil (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> ceil(T x) __NOEXC {
  return __sycl_std::__invoke_ceil<T>(x);
}

// genfloat copysign (genfloat x, genfloat y)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> copysign(T x,
                                                               T y) __NOEXC {
  return __sycl_std::__invoke_copysign<T>(x, y);
}

// genfloat cos (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> cos(T x) __NOEXC {
  return __sycl_std::__invoke_cos<T>(x);
}

// genfloat cosh (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> cosh(T x) __NOEXC {
  return __sycl_std::__invoke_cosh<T>(x);
}

// genfloat cospi (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> cospi(T x) __NOEXC {
  return __sycl_std::__invoke_cospi<T>(x);
}

// genfloat erfc (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> erfc(T x) __NOEXC {
  return __sycl_std::__invoke_erfc<T>(x);
}

// genfloat erf (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> erf(T x) __NOEXC {
  return __sycl_std::__invoke_erf<T>(x);
}

// genfloat exp (genfloat x )
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> exp(T x) __NOEXC {
  return __sycl_std::__invoke_exp<T>(x);
}

// genfloat exp2 (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> exp2(T x) __NOEXC {
  return __sycl_std::__invoke_exp2<T>(x);
}

// genfloat exp10 (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> exp10(T x) __NOEXC {
  return __sycl_std::__invoke_exp10<T>(x);
}

// genfloat expm1 (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> expm1(T x) __NOEXC {
  return __sycl_std::__invoke_expm1<T>(x);
}

// genfloat fabs (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> fabs(T x) __NOEXC {
  return __sycl_std::__invoke_fabs<T>(x);
}

// genfloat fdim (genfloat x, genfloat y)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> fdim(T x, T y) __NOEXC {
  return __sycl_std::__invoke_fdim<T>(x, y);
}

// genfloat floor (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> floor(T x) __NOEXC {
  return __sycl_std::__invoke_floor<T>(x);
}

// genfloat fma (genfloat a, genfloat b, genfloat c)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> fma(T a, T b,
                                                          T c) __NOEXC {
  return __sycl_std::__invoke_fma<T>(a, b, c);
}

// genfloat fmax (genfloat x, genfloat y)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> fmax(T x, T y) __NOEXC {
  return __sycl_std::__invoke_fmax<T>(x, y);
}

// genfloat fmax (genfloat x, sgenfloat y)
template <typename T>
detail::enable_if_t<detail::is_vgenfloat<T>::value, T>
fmax(T x, typename T::element_type y) __NOEXC {
  return __sycl_std::__invoke_fmax<T>(x, T(y));
}

// genfloat fmin (genfloat x, genfloat y)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> fmin(T x, T y) __NOEXC {
  return __sycl_std::__invoke_fmin<T>(x, y);
}

// genfloat fmin (genfloat x, sgenfloat y)
template <typename T>
detail::enable_if_t<detail::is_vgenfloat<T>::value, T>
fmin(T x, typename T::element_type y) __NOEXC {
  return __sycl_std::__invoke_fmin<T>(x, T(y));
}

// genfloat fmod (genfloat x, genfloat y)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> fmod(T x, T y) __NOEXC {
  return __sycl_std::__invoke_fmod<T>(x, y);
}

// genfloat fract (genfloat x, genfloatptr iptr)
template <typename T, typename T2>
detail::enable_if_t<
    detail::is_genfloat<T>::value && detail::is_genfloatptr<T2>::value, T>
fract(T x, T2 iptr) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_fract<T>(x, iptr);
}

// genfloat frexp (genfloat x, genintptr exp)
template <typename T, typename T2>
detail::enable_if_t<
    detail::is_genfloat<T>::value && detail::is_genintptr<T2>::value, T>
frexp(T x, T2 exp) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_frexp<T>(x, exp);
}

// genfloat hypot (genfloat x, genfloat y)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> hypot(T x, T y) __NOEXC {
  return __sycl_std::__invoke_hypot<T>(x, y);
}

// genint ilogb (genfloat x)
template <typename T,
          typename = detail::enable_if_t<detail::is_genfloat<T>::value, T>>
detail::change_base_type_t<T, int> ilogb(T x) __NOEXC {
  return __sycl_std::__invoke_ilogb<detail::change_base_type_t<T, int>>(x);
}

// float ldexp (float x, int k)
// double ldexp (double x, int k)
// half ldexp (half x, int k)
template <typename T>
detail::enable_if_t<detail::is_sgenfloat<T>::value, T> ldexp(T x,
                                                             int k) __NOEXC {
  return __sycl_std::__invoke_ldexp<T>(x, k);
}

// vgenfloat ldexp (vgenfloat x, int k)
template <typename T>
detail::enable_if_t<detail::is_vgenfloat<T>::value, T> ldexp(T x,
                                                             int k) __NOEXC {
  return __sycl_std::__invoke_ldexp<T>(x, vec<int, T::size()>(k));
}

// vgenfloat ldexp (vgenfloat x, genint k)
template <typename T, typename T2>
detail::enable_if_t<
    detail::is_vgenfloat<T>::value && detail::is_intn<T2>::value, T>
ldexp(T x, T2 k) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_ldexp<T>(x, k);
}

// genfloat lgamma (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> lgamma(T x) __NOEXC {
  return __sycl_std::__invoke_lgamma<T>(x);
}

// genfloat lgamma_r (genfloat x, genintptr signp)
template <typename T, typename T2>
detail::enable_if_t<
    detail::is_genfloat<T>::value && detail::is_genintptr<T2>::value, T>
lgamma_r(T x, T2 signp) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_lgamma_r<T>(x, signp);
}

// genfloat log (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> log(T x) __NOEXC {
  return __sycl_std::__invoke_log<T>(x);
}

// genfloat log2 (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> log2(T x) __NOEXC {
  return __sycl_std::__invoke_log2<T>(x);
}

// genfloat log10 (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> log10(T x) __NOEXC {
  return __sycl_std::__invoke_log10<T>(x);
}

// genfloat log1p (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> log1p(T x) __NOEXC {
  return __sycl_std::__invoke_log1p<T>(x);
}

// genfloat logb (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> logb(T x) __NOEXC {
  return __sycl_std::__invoke_logb<T>(x);
}

// genfloat mad (genfloat a, genfloat b, genfloat c)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> mad(T a, T b,
                                                          T c) __NOEXC {
  return __sycl_std::__invoke_mad<T>(a, b, c);
}

// genfloat maxmag (genfloat x, genfloat y)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> maxmag(T x, T y) __NOEXC {
  return __sycl_std::__invoke_maxmag<T>(x, y);
}

// genfloat minmag (genfloat x, genfloat y)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> minmag(T x, T y) __NOEXC {
  return __sycl_std::__invoke_minmag<T>(x, y);
}

// genfloat modf (genfloat x, genfloatptr iptr)
template <typename T, typename T2>
detail::enable_if_t<
    detail::is_genfloat<T>::value && detail::is_genfloatptr<T2>::value, T>
modf(T x, T2 iptr) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_modf<T>(x, iptr);
}

template <typename T,
          typename = detail::enable_if_t<detail::is_nan_type<T>::value, T>>
detail::nan_return_t<T> nan(T nancode) __NOEXC {
  return __sycl_std::__invoke_nan<detail::nan_return_t<T>>(
      detail::convert_data_type<T, detail::nan_argument_base_t<T>>()(nancode));
}

// genfloat nextafter (genfloat x, genfloat y)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> nextafter(T x,
                                                                T y) __NOEXC {
  return __sycl_std::__invoke_nextafter<T>(x, y);
}

// genfloat pow (genfloat x, genfloat y)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> pow(T x, T y) __NOEXC {
  return __sycl_std::__invoke_pow<T>(x, y);
}

// genfloat pown (genfloat x, genint y)
template <typename T, typename T2>
detail::enable_if_t<
    detail::is_genfloat<T>::value && detail::is_genint<T2>::value, T>
pown(T x, T2 y) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_pown<T>(x, y);
}

// genfloat powr (genfloat x, genfloat y)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> powr(T x, T y) __NOEXC {
  return __sycl_std::__invoke_powr<T>(x, y);
}

// genfloat remainder (genfloat x, genfloat y)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> remainder(T x,
                                                                T y) __NOEXC {
  return __sycl_std::__invoke_remainder<T>(x, y);
}

// genfloat remquo (genfloat x, genfloat y, genintptr quo)
template <typename T, typename T2>
detail::enable_if_t<
    detail::is_genfloat<T>::value && detail::is_genintptr<T2>::value, T>
remquo(T x, T y, T2 quo) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_remquo<T>(x, y, quo);
}

// genfloat rint (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> rint(T x) __NOEXC {
  return __sycl_std::__invoke_rint<T>(x);
}

// genfloat rootn (genfloat x, genint y)
template <typename T, typename T2>
detail::enable_if_t<
    detail::is_genfloat<T>::value && detail::is_genint<T2>::value, T>
rootn(T x, T2 y) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_rootn<T>(x, y);
}

// genfloat round (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> round(T x) __NOEXC {
  return __sycl_std::__invoke_round<T>(x);
}

// genfloat rsqrt (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> rsqrt(T x) __NOEXC {
  return __sycl_std::__invoke_rsqrt<T>(x);
}

// genfloat sin (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> sin(T x) __NOEXC {
  return __sycl_std::__invoke_sin<T>(x);
}

// genfloat sincos (genfloat x, genfloatptr cosval)
template <typename T, typename T2>
detail::enable_if_t<
    detail::is_genfloat<T>::value && detail::is_genfloatptr<T2>::value, T>
sincos(T x, T2 cosval) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_sincos<T>(x, cosval);
}

// genfloat sinh (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> sinh(T x) __NOEXC {
  return __sycl_std::__invoke_sinh<T>(x);
}

// genfloat sinpi (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> sinpi(T x) __NOEXC {
  return __sycl_std::__invoke_sinpi<T>(x);
}

// genfloat sqrt (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> sqrt(T x) __NOEXC {
  return __sycl_std::__invoke_sqrt<T>(x);
}

// genfloat tan (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> tan(T x) __NOEXC {
  return __sycl_std::__invoke_tan<T>(x);
}

// genfloat tanh (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> tanh(T x) __NOEXC {
  return __sycl_std::__invoke_tanh<T>(x);
}

// genfloat tanpi (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> tanpi(T x) __NOEXC {
  return __sycl_std::__invoke_tanpi<T>(x);
}

// genfloat tgamma (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> tgamma(T x) __NOEXC {
  return __sycl_std::__invoke_tgamma<T>(x);
}

// genfloat trunc (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> trunc(T x) __NOEXC {
  return __sycl_std::__invoke_trunc<T>(x);
}

/* --------------- 4.13.5 Common functions. ---------------------------------*/
// genfloat clamp (genfloat x, genfloat minval, genfloat maxval)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> clamp(T x, T minval,
                                                            T maxval) __NOEXC {
  return __sycl_std::__invoke_fclamp<T>(x, minval, maxval);
}

// genfloath clamp (genfloath x, half minval, half maxval)
// genfloatf clamp (genfloatf x, float minval, float maxval)
// genfloatd clamp (genfloatd x, double minval, double maxval)
template <typename T>
detail::enable_if_t<detail::is_vgenfloat<T>::value, T>
clamp(T x, typename T::element_type minval,
      typename T::element_type maxval) __NOEXC {
  return __sycl_std::__invoke_fclamp<T>(x, T(minval), T(maxval));
}

// genfloat degrees (genfloat radians)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T>
degrees(T radians) __NOEXC {
  return __sycl_std::__invoke_degrees<T>(radians);
}

// genfloat abs (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> abs(T x) __NOEXC {
  return __sycl_std::__invoke_fabs<T>(x);
}

// genfloat max (genfloat x, genfloat y)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T>(max)(T x, T y) __NOEXC {
  return __sycl_std::__invoke_fmax_common<T>(x, y);
}

// genfloatf max (genfloatf x, float y)
// genfloatd max (genfloatd x, double y)
// genfloath max (genfloath x, half y)
template <typename T>
detail::enable_if_t<detail::is_vgenfloat<T>::value, T>(max)(
    T x, typename T::element_type y) __NOEXC {
  return __sycl_std::__invoke_fmax_common<T>(x, T(y));
}

// genfloat min (genfloat x, genfloat y)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T>(min)(T x, T y) __NOEXC {
  return __sycl_std::__invoke_fmin_common<T>(x, y);
}

// genfloatf min (genfloatf x, float y)
// genfloatd min (genfloatd x, double y)
// genfloath min (genfloath x, half y)
template <typename T>
detail::enable_if_t<detail::is_vgenfloat<T>::value, T>(min)(
    T x, typename T::element_type y) __NOEXC {
  return __sycl_std::__invoke_fmin_common<T>(x, T(y));
}

// genfloat mix (genfloat x, genfloat y, genfloat a)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> mix(T x, T y,
                                                          T a) __NOEXC {
  return __sycl_std::__invoke_mix<T>(x, y, a);
}

// genfloatf mix (genfloatf x, genfloatf y, float a)
// genfloatd mix (genfloatd x, genfloatd y, double a)
// genfloatd mix (genfloath x, genfloath y, half a)
template <typename T>
detail::enable_if_t<detail::is_vgenfloat<T>::value, T>
mix(T x, T y, typename T::element_type a) __NOEXC {
  return __sycl_std::__invoke_mix<T>(x, y, T(a));
}

// genfloat radians (genfloat degrees)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T>
radians(T degrees) __NOEXC {
  return __sycl_std::__invoke_radians<T>(degrees);
}

// genfloat step (genfloat edge, genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> step(T edge,
                                                           T x) __NOEXC {
  return __sycl_std::__invoke_step<T>(edge, x);
}

// genfloatf step (float edge, genfloatf x)
// genfloatd step (double edge, genfloatd x)
// genfloatd step (half edge, genfloath x)
template <typename T>
detail::enable_if_t<detail::is_vgenfloat<T>::value, T>
step(typename T::element_type edge, T x) __NOEXC {
  return __sycl_std::__invoke_step<T>(T(edge), x);
}

// genfloat smoothstep (genfloat edge0, genfloat edge1, genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T>
smoothstep(T edge0, T edge1, T x) __NOEXC {
  return __sycl_std::__invoke_smoothstep<T>(edge0, edge1, x);
}

// genfloatf smoothstep (float edge0, float edge1, genfloatf x)
// genfloatd smoothstep (double edge0, double edge1, genfloatd x)
// genfloath smoothstep (half edge0, half edge1, genfloath x)
template <typename T>
detail::enable_if_t<detail::is_vgenfloat<T>::value, T>
smoothstep(typename T::element_type edge0, typename T::element_type edge1,
           T x) __NOEXC {
  return __sycl_std::__invoke_smoothstep<T>(T(edge0), T(edge1), x);
}

// genfloat sign (genfloat x)
template <typename T>
detail::enable_if_t<detail::is_genfloat<T>::value, T> sign(T x) __NOEXC {
  return __sycl_std::__invoke_sign<T>(x);
}

/* --------------- 4.13.4 Integer functions. --------------------------------*/
// ugeninteger abs (geninteger x)
template <typename T>
detail::enable_if_t<detail::is_ugeninteger<T>::value, T> abs(T x) __NOEXC {
  return __sycl_std::__invoke_u_abs<T>(x);
}

// ugeninteger abs (geninteger x)
template <typename T>
detail::enable_if_t<detail::is_igeninteger<T>::value,
                    detail::make_unsigned_t<T>>
abs(T x) __NOEXC {
  return __sycl_std::__invoke_s_abs<detail::make_unsigned_t<T>>(x);
}

// ugeninteger abs_diff (geninteger x, geninteger y)
template <typename T>
detail::enable_if_t<detail::is_ugeninteger<T>::value, T> abs_diff(T x,
                                                                  T y) __NOEXC {
  return __sycl_std::__invoke_u_abs_diff<T>(x, y);
}

// ugeninteger abs_diff (geninteger x, geninteger y)
template <typename T>
detail::enable_if_t<detail::is_igeninteger<T>::value,
                    detail::make_unsigned_t<T>>
abs_diff(T x, T y) __NOEXC {
  return __sycl_std::__invoke_s_abs_diff<detail::make_unsigned_t<T>>(x, y);
}

// geninteger add_sat (geninteger x, geninteger y)
template <typename T>
detail::enable_if_t<detail::is_igeninteger<T>::value, T> add_sat(T x,
                                                                 T y) __NOEXC {
  return __sycl_std::__invoke_s_add_sat<T>(x, y);
}

// geninteger add_sat (geninteger x, geninteger y)
template <typename T>
detail::enable_if_t<detail::is_ugeninteger<T>::value, T> add_sat(T x,
                                                                 T y) __NOEXC {
  return __sycl_std::__invoke_u_add_sat<T>(x, y);
}

// geninteger hadd (geninteger x, geninteger y)
template <typename T>
detail::enable_if_t<detail::is_igeninteger<T>::value, T> hadd(T x,
                                                              T y) __NOEXC {
  return __sycl_std::__invoke_s_hadd<T>(x, y);
}

// geninteger hadd (geninteger x, geninteger y)
template <typename T>
detail::enable_if_t<detail::is_ugeninteger<T>::value, T> hadd(T x,
                                                              T y) __NOEXC {
  return __sycl_std::__invoke_u_hadd<T>(x, y);
}

// geninteger rhadd (geninteger x, geninteger y)
template <typename T>
detail::enable_if_t<detail::is_igeninteger<T>::value, T> rhadd(T x,
                                                               T y) __NOEXC {
  return __sycl_std::__invoke_s_rhadd<T>(x, y);
}

// geninteger rhadd (geninteger x, geninteger y)
template <typename T>
detail::enable_if_t<detail::is_ugeninteger<T>::value, T> rhadd(T x,
                                                               T y) __NOEXC {
  return __sycl_std::__invoke_u_rhadd<T>(x, y);
}

// geninteger clamp (geninteger x, geninteger minval, geninteger maxval)
template <typename T>
detail::enable_if_t<detail::is_igeninteger<T>::value, T>
clamp(T x, T minval, T maxval) __NOEXC {
  return __sycl_std::__invoke_s_clamp<T>(x, minval, maxval);
}

// geninteger clamp (geninteger x, geninteger minval, geninteger maxval)
template <typename T>
detail::enable_if_t<detail::is_ugeninteger<T>::value, T>
clamp(T x, T minval, T maxval) __NOEXC {
  return __sycl_std::__invoke_u_clamp<T>(x, minval, maxval);
}

// geninteger clamp (geninteger x, sgeninteger minval, sgeninteger maxval)
template <typename T>
detail::enable_if_t<detail::is_vigeninteger<T>::value, T>
clamp(T x, typename T::element_type minval,
      typename T::element_type maxval) __NOEXC {
  return __sycl_std::__invoke_s_clamp<T>(x, T(minval), T(maxval));
}

// geninteger clamp (geninteger x, sgeninteger minval, sgeninteger maxval)
template <typename T>
detail::enable_if_t<detail::is_vugeninteger<T>::value, T>
clamp(T x, typename T::element_type minval,
      typename T::element_type maxval) __NOEXC {
  return __sycl_std::__invoke_u_clamp<T>(x, T(minval), T(maxval));
}

// geninteger clz (geninteger x)
template <typename T>
detail::enable_if_t<detail::is_geninteger<T>::value, T> clz(T x) __NOEXC {
  return __sycl_std::__invoke_clz<T>(x);
}

namespace ext {
namespace intel {
// geninteger ctz (geninteger x)
template <typename T>
sycl::detail::enable_if_t<sycl::detail::is_geninteger<T>::value, T>
ctz(T x) __NOEXC {
  return __sycl_std::__invoke_ctz<T>(x);
}
} // namespace intel
} // namespace ext

namespace __SYCL2020_DEPRECATED("use 'ext::intel' instead") intel {
  using namespace ext::intel;
}

// geninteger mad_hi (geninteger a, geninteger b, geninteger c)
template <typename T>
detail::enable_if_t<detail::is_igeninteger<T>::value, T> mad_hi(T x, T y,
                                                                T z) __NOEXC {
  return __sycl_std::__invoke_s_mad_hi<T>(x, y, z);
}

// geninteger mad_hi (geninteger a, geninteger b, geninteger c)
template <typename T>
detail::enable_if_t<detail::is_ugeninteger<T>::value, T> mad_hi(T x, T y,
                                                                T z) __NOEXC {
  return __sycl_std::__invoke_u_mad_hi<T>(x, y, z);
}

// geninteger mad_sat (geninteger a, geninteger b, geninteger c)
template <typename T>
detail::enable_if_t<detail::is_igeninteger<T>::value, T> mad_sat(T a, T b,
                                                                 T c) __NOEXC {
  return __sycl_std::__invoke_s_mad_sat<T>(a, b, c);
}

// geninteger mad_sat (geninteger a, geninteger b, geninteger c)
template <typename T>
detail::enable_if_t<detail::is_ugeninteger<T>::value, T> mad_sat(T a, T b,
                                                                 T c) __NOEXC {
  return __sycl_std::__invoke_u_mad_sat<T>(a, b, c);
}

// igeninteger max (igeninteger x, igeninteger y)
template <typename T>
detail::enable_if_t<detail::is_igeninteger<T>::value, T>(max)(T x,
                                                              T y) __NOEXC {
  return __sycl_std::__invoke_s_max<T>(x, y);
}

// ugeninteger max (ugeninteger x, ugeninteger y)
template <typename T>
detail::enable_if_t<detail::is_ugeninteger<T>::value, T>(max)(T x,
                                                              T y) __NOEXC {
  return __sycl_std::__invoke_u_max<T>(x, y);
}

// igeninteger max (vigeninteger x, sigeninteger y)
template <typename T>
detail::enable_if_t<detail::is_vigeninteger<T>::value, T>(max)(
    T x, typename T::element_type y) __NOEXC {
  return __sycl_std::__invoke_s_max<T>(x, T(y));
}

// vugeninteger max (vugeninteger x, sugeninteger y)
template <typename T>
detail::enable_if_t<detail::is_vugeninteger<T>::value, T>(max)(
    T x, typename T::element_type y) __NOEXC {
  return __sycl_std::__invoke_u_max<T>(x, T(y));
}

// igeninteger min (igeninteger x, igeninteger y)
template <typename T>
detail::enable_if_t<detail::is_igeninteger<T>::value, T>(min)(T x,
                                                              T y) __NOEXC {
  return __sycl_std::__invoke_s_min<T>(x, y);
}

// ugeninteger min (ugeninteger x, ugeninteger y)
template <typename T>
detail::enable_if_t<detail::is_ugeninteger<T>::value, T>(min)(T x,
                                                              T y) __NOEXC {
  return __sycl_std::__invoke_u_min<T>(x, y);
}

// vigeninteger min (vigeninteger x, sigeninteger y)
template <typename T>
detail::enable_if_t<detail::is_vigeninteger<T>::value, T>(min)(
    T x, typename T::element_type y) __NOEXC {
  return __sycl_std::__invoke_s_min<T>(x, T(y));
}

// vugeninteger min (vugeninteger x, sugeninteger y)
template <typename T>
detail::enable_if_t<detail::is_vugeninteger<T>::value, T>(min)(
    T x, typename T::element_type y) __NOEXC {
  return __sycl_std::__invoke_u_min<T>(x, T(y));
}

// geninteger mul_hi (geninteger x, geninteger y)
template <typename T>
detail::enable_if_t<detail::is_igeninteger<T>::value, T> mul_hi(T x,
                                                                T y) __NOEXC {
  return __sycl_std::__invoke_s_mul_hi<T>(x, y);
}

// geninteger mul_hi (geninteger x, geninteger y)
template <typename T>
detail::enable_if_t<detail::is_ugeninteger<T>::value, T> mul_hi(T x,
                                                                T y) __NOEXC {
  return __sycl_std::__invoke_u_mul_hi<T>(x, y);
}

// geninteger rotate (geninteger v, geninteger i)
template <typename T>
detail::enable_if_t<detail::is_geninteger<T>::value, T> rotate(T v,
                                                               T i) __NOEXC {
  return __sycl_std::__invoke_rotate<T>(v, i);
}

// geninteger sub_sat (geninteger x, geninteger y)
template <typename T>
detail::enable_if_t<detail::is_igeninteger<T>::value, T> sub_sat(T x,
                                                                 T y) __NOEXC {
  return __sycl_std::__invoke_s_sub_sat<T>(x, y);
}

// geninteger sub_sat (geninteger x, geninteger y)
template <typename T>
detail::enable_if_t<detail::is_ugeninteger<T>::value, T> sub_sat(T x,
                                                                 T y) __NOEXC {
  return __sycl_std::__invoke_u_sub_sat<T>(x, y);
}

// ugeninteger16bit upsample (ugeninteger8bit hi, ugeninteger8bit lo)
template <typename T>
detail::enable_if_t<detail::is_ugeninteger8bit<T>::value,
                    detail::make_larger_t<T>>
upsample(T hi, T lo) __NOEXC {
  return __sycl_std::__invoke_u_upsample<detail::make_larger_t<T>>(hi, lo);
}

// igeninteger16bit upsample (igeninteger8bit hi, ugeninteger8bit lo)
template <typename T, typename T2>
detail::enable_if_t<detail::is_igeninteger8bit<T>::value &&
                        detail::is_ugeninteger8bit<T2>::value,
                    detail::make_larger_t<T>>
upsample(T hi, T2 lo) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_s_upsample<detail::make_larger_t<T>>(hi, lo);
}

// ugeninteger32bit upsample (ugeninteger16bit hi, ugeninteger16bit lo)
template <typename T>
detail::enable_if_t<detail::is_ugeninteger16bit<T>::value,
                    detail::make_larger_t<T>>
upsample(T hi, T lo) __NOEXC {
  return __sycl_std::__invoke_u_upsample<detail::make_larger_t<T>>(hi, lo);
}

// igeninteger32bit upsample (igeninteger16bit hi, ugeninteger16bit lo)
template <typename T, typename T2>
detail::enable_if_t<detail::is_igeninteger16bit<T>::value &&
                        detail::is_ugeninteger16bit<T2>::value,
                    detail::make_larger_t<T>>
upsample(T hi, T2 lo) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_s_upsample<detail::make_larger_t<T>>(hi, lo);
}

// ugeninteger64bit upsample (ugeninteger32bit hi, ugeninteger32bit lo)
template <typename T>
detail::enable_if_t<detail::is_ugeninteger32bit<T>::value,
                    detail::make_larger_t<T>>
upsample(T hi, T lo) __NOEXC {
  return __sycl_std::__invoke_u_upsample<detail::make_larger_t<T>>(hi, lo);
}

// igeninteger64bit upsample (igeninteger32bit hi, ugeninteger32bit lo)
template <typename T, typename T2>
detail::enable_if_t<detail::is_igeninteger32bit<T>::value &&
                        detail::is_ugeninteger32bit<T2>::value,
                    detail::make_larger_t<T>>
upsample(T hi, T2 lo) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_s_upsample<detail::make_larger_t<T>>(hi, lo);
}

// geninteger popcount (geninteger x)
template <typename T>
detail::enable_if_t<detail::is_geninteger<T>::value, T> popcount(T x) __NOEXC {
  return __sycl_std::__invoke_popcount<T>(x);
}

// geninteger32bit mad24 (geninteger32bit x, geninteger32bit y,
// geninteger32bit z)
template <typename T>
detail::enable_if_t<detail::is_igeninteger32bit<T>::value, T>
mad24(T x, T y, T z) __NOEXC {
  return __sycl_std::__invoke_s_mad24<T>(x, y, z);
}

// geninteger32bit mad24 (geninteger32bit x, geninteger32bit y,
// geninteger32bit z)
template <typename T>
detail::enable_if_t<detail::is_ugeninteger32bit<T>::value, T>
mad24(T x, T y, T z) __NOEXC {
  return __sycl_std::__invoke_u_mad24<T>(x, y, z);
}

// geninteger32bit mul24 (geninteger32bit x, geninteger32bit y)
template <typename T>
detail::enable_if_t<detail::is_igeninteger32bit<T>::value, T>
mul24(T x, T y) __NOEXC {
  return __sycl_std::__invoke_s_mul24<T>(x, y);
}

// geninteger32bit mul24 (geninteger32bit x, geninteger32bit y)
template <typename T>
detail::enable_if_t<detail::is_ugeninteger32bit<T>::value, T>
mul24(T x, T y) __NOEXC {
  return __sycl_std::__invoke_u_mul24<T>(x, y);
}

/* --------------- 4.13.6 Geometric Functions. ------------------------------*/
// float3 cross (float3 p0, float3 p1)
// float4 cross (float4 p0, float4 p1)
// double3 cross (double3 p0, double3 p1)
// double4 cross (double4 p0, double4 p1)
// half3 cross (half3 p0, half3 p1)
// half4 cross (half4 p0, half4 p1)
template <typename T>
detail::enable_if_t<detail::is_gencross<T>::value, T> cross(T p0,
                                                            T p1) __NOEXC {
  return __sycl_std::__invoke_cross<T>(p0, p1);
}

// float dot (float p0, float p1)
// double dot (double p0, double p1)
// half dot (half p0, half p1)
template <typename T>
detail::enable_if_t<detail::is_sgenfloat<T>::value, T> dot(T p0, T p1) __NOEXC {
  return p0 * p1;
}

// float dot (vgengeofloat p0, vgengeofloat p1)
template <typename T>
detail::enable_if_t<detail::is_vgengeofloat<T>::value, float>
dot(T p0, T p1) __NOEXC {
  return __sycl_std::__invoke_Dot<float>(p0, p1);
}

// double dot (vgengeodouble p0, vgengeodouble p1)
template <typename T>
detail::enable_if_t<detail::is_vgengeodouble<T>::value, double>
dot(T p0, T p1) __NOEXC {
  return __sycl_std::__invoke_Dot<double>(p0, p1);
}

// half dot (vgengeohalf p0, vgengeohalf p1)
template <typename T>
detail::enable_if_t<detail::is_vgengeohalf<T>::value, half> dot(T p0,
                                                                T p1) __NOEXC {
  return __sycl_std::__invoke_Dot<half>(p0, p1);
}

// float distance (gengeofloat p0, gengeofloat p1)
template <typename T,
          typename = detail::enable_if_t<detail::is_gengeofloat<T>::value, T>>
float distance(T p0, T p1) __NOEXC {
  return __sycl_std::__invoke_distance<float>(p0, p1);
}

// double distance (gengeodouble p0, gengeodouble p1)
template <typename T,
          typename = detail::enable_if_t<detail::is_gengeodouble<T>::value, T>>
double distance(T p0, T p1) __NOEXC {
  return __sycl_std::__invoke_distance<double>(p0, p1);
}

// half distance (gengeohalf p0, gengeohalf p1)
template <typename T,
          typename = detail::enable_if_t<detail::is_gengeohalf<T>::value, T>>
half distance(T p0, T p1) __NOEXC {
  return __sycl_std::__invoke_distance<half>(p0, p1);
}

// float length (gengeofloat p)
template <typename T,
          typename = detail::enable_if_t<detail::is_gengeofloat<T>::value, T>>
float length(T p) __NOEXC {
  return __sycl_std::__invoke_length<float>(p);
}

// double length (gengeodouble p)
template <typename T,
          typename = detail::enable_if_t<detail::is_gengeodouble<T>::value, T>>
double length(T p) __NOEXC {
  return __sycl_std::__invoke_length<double>(p);
}

// half length (gengeohalf p)
template <typename T,
          typename = detail::enable_if_t<detail::is_gengeohalf<T>::value, T>>
half length(T p) __NOEXC {
  return __sycl_std::__invoke_length<half>(p);
}

// gengeofloat normalize (gengeofloat p)
template <typename T>
detail::enable_if_t<detail::is_gengeofloat<T>::value, T>
normalize(T p) __NOEXC {
  return __sycl_std::__invoke_normalize<T>(p);
}

// gengeodouble normalize (gengeodouble p)
template <typename T>
detail::enable_if_t<detail::is_gengeodouble<T>::value, T>
normalize(T p) __NOEXC {
  return __sycl_std::__invoke_normalize<T>(p);
}

// gengeohalf normalize (gengeohalf p)
template <typename T>
detail::enable_if_t<detail::is_gengeohalf<T>::value, T> normalize(T p) __NOEXC {
  return __sycl_std::__invoke_normalize<T>(p);
}

// float fast_distance (gengeofloat p0, gengeofloat p1)
template <typename T,
          typename = detail::enable_if_t<detail::is_gengeofloat<T>::value, T>>
float fast_distance(T p0, T p1) __NOEXC {
  return __sycl_std::__invoke_fast_distance<float>(p0, p1);
}

// double fast_distance (gengeodouble p0, gengeodouble p1)
template <typename T,
          typename = detail::enable_if_t<detail::is_gengeodouble<T>::value, T>>
double fast_distance(T p0, T p1) __NOEXC {
  return __sycl_std::__invoke_fast_distance<double>(p0, p1);
}

// float fast_length (gengeofloat p)
template <typename T,
          typename = detail::enable_if_t<detail::is_gengeofloat<T>::value, T>>
float fast_length(T p) __NOEXC {
  return __sycl_std::__invoke_fast_length<float>(p);
}

// double fast_length (gengeodouble p)
template <typename T,
          typename = detail::enable_if_t<detail::is_gengeodouble<T>::value, T>>
double fast_length(T p) __NOEXC {
  return __sycl_std::__invoke_fast_length<double>(p);
}

// gengeofloat fast_normalize (gengeofloat p)
template <typename T>
detail::enable_if_t<detail::is_gengeofloat<T>::value, T>
fast_normalize(T p) __NOEXC {
  return __sycl_std::__invoke_fast_normalize<T>(p);
}

// gengeodouble fast_normalize (gengeodouble p)
template <typename T>
detail::enable_if_t<detail::is_gengeodouble<T>::value, T>
fast_normalize(T p) __NOEXC {
  return __sycl_std::__invoke_fast_normalize<T>(p);
}

/* --------------- 4.13.7 Relational functions. Device version --------------*/
// int isequal (half x, half y)
// shortn isequal (halfn x, halfn y)
// igeninteger32bit isequal (genfloatf x, genfloatf y)
// int isequal (double x,double y);
// longn isequal (doublen x, doublen y)
template <typename T,
          typename = detail::enable_if_t<detail::is_genfloat<T>::value, T>>
detail::common_rel_ret_t<T> isequal(T x, T y) __NOEXC {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_FOrdEqual<detail::rel_ret_t<T>>(x, y));
}

// int isnotequal (half x, half y)
// shortn isnotequal (halfn x, halfn y)
// igeninteger32bit isnotequal (genfloatf x, genfloatf y)
// int isnotequal (double x, double y)
// longn isnotequal (doublen x, doublen y)
template <typename T,
          typename = detail::enable_if_t<detail::is_genfloat<T>::value, T>>
detail::common_rel_ret_t<T> isnotequal(T x, T y) __NOEXC {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_FUnordNotEqual<detail::rel_ret_t<T>>(x, y));
}

// int isgreater (half x, half y)
// shortn isgreater (halfn x, halfn y)
// igeninteger32bit isgreater (genfloatf x, genfloatf y)
// int isgreater (double x, double y)
// longn isgreater (doublen x, doublen y)
template <typename T,
          typename = detail::enable_if_t<detail::is_genfloat<T>::value, T>>
detail::common_rel_ret_t<T> isgreater(T x, T y) __NOEXC {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_FOrdGreaterThan<detail::rel_ret_t<T>>(x, y));
}

// int isgreaterequal (half x, half y)
// shortn isgreaterequal (halfn x, halfn y)
// igeninteger32bit isgreaterequal (genfloatf x, genfloatf y)
// int isgreaterequal (double x, double y)
// longn isgreaterequal (doublen x, doublen y)
template <typename T,
          typename = detail::enable_if_t<detail::is_genfloat<T>::value, T>>
detail::common_rel_ret_t<T> isgreaterequal(T x, T y) __NOEXC {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_FOrdGreaterThanEqual<detail::rel_ret_t<T>>(x, y));
}

// int isless (half x, half y)
// shortn isless (halfn x, halfn y)
// igeninteger32bit isless (genfloatf x, genfloatf y)
// int isless (long x, long y)
// longn isless (doublen x, doublen y)
template <typename T,
          typename = detail::enable_if_t<detail::is_genfloat<T>::value, T>>
detail::common_rel_ret_t<T> isless(T x, T y) __NOEXC {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_FOrdLessThan<detail::rel_ret_t<T>>(x, y));
}

// int islessequal (half x, half y)
// shortn islessequal (halfn x, halfn y)
// igeninteger32bit islessequal (genfloatf x, genfloatf y)
// int islessequal (double x, double y)
// longn islessequal (doublen x, doublen y)
template <typename T,
          typename = detail::enable_if_t<detail::is_genfloat<T>::value, T>>
detail::common_rel_ret_t<T> islessequal(T x, T y) __NOEXC {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_FOrdLessThanEqual<detail::rel_ret_t<T>>(x, y));
}

// int islessgreater (half x, half y)
// shortn islessgreater (halfn x, halfn y)
// igeninteger32bit islessgreater (genfloatf x, genfloatf y)
// int islessgreater (double x, double y)
// longn islessgreater (doublen x, doublen y)
template <typename T,
          typename = detail::enable_if_t<detail::is_genfloat<T>::value, T>>
detail::common_rel_ret_t<T> islessgreater(T x, T y) __NOEXC {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_LessOrGreater<detail::rel_ret_t<T>>(x, y));
}

// int isfinite (half x)
// shortn isfinite (halfn x)
// igeninteger32bit isfinite (genfloatf x)
// int isfinite (double x)
// longn isfinite (doublen x)
template <typename T,
          typename = detail::enable_if_t<detail::is_genfloat<T>::value, T>>
detail::common_rel_ret_t<T> isfinite(T x) __NOEXC {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_IsFinite<detail::rel_ret_t<T>>(x));
}

// int isinf (half x)
// shortn isinf (halfn x)
// igeninteger32bit isinf (genfloatf x)
// int isinf (double x)
// longn isinf (doublen x)
template <typename T,
          typename = detail::enable_if_t<detail::is_genfloat<T>::value, T>>
detail::common_rel_ret_t<T> isinf(T x) __NOEXC {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_IsInf<detail::rel_ret_t<T>>(x));
}

// int isnan (half x)
// shortn isnan (halfn x)
// igeninteger32bit isnan (genfloatf x)
// int isnan (double x)
// longn isnan (doublen x)
template <typename T,
          typename = detail::enable_if_t<detail::is_genfloat<T>::value, T>>
detail::common_rel_ret_t<T> isnan(T x) __NOEXC {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_IsNan<detail::rel_ret_t<T>>(x));
}

// int isnormal (half x)
// shortn isnormal (halfn x)
// igeninteger32bit isnormal (genfloatf x)
// int isnormal (double x)
// longn isnormal (doublen x)
template <typename T,
          typename = detail::enable_if_t<detail::is_genfloat<T>::value, T>>
detail::common_rel_ret_t<T> isnormal(T x) __NOEXC {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_IsNormal<detail::rel_ret_t<T>>(x));
}

// int isordered (half x)
// shortn isordered (halfn x, halfn y)
// igeninteger32bit isordered (genfloatf x, genfloatf y)
// int isordered (double x, double y)
// longn isordered (doublen x, doublen y)
template <typename T,
          typename = detail::enable_if_t<detail::is_genfloat<T>::value, T>>
detail::common_rel_ret_t<T> isordered(T x, T y) __NOEXC {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_Ordered<detail::rel_ret_t<T>>(x, y));
}

// int isunordered (half x, half y)
// shortn isunordered (halfn x, halfn y)
// igeninteger32bit isunordered (genfloatf x, genfloatf y)
// int isunordered (double x, double y)
// longn isunordered (doublen x, doublen y)
template <typename T,
          typename = detail::enable_if_t<detail::is_genfloat<T>::value, T>>
detail::common_rel_ret_t<T> isunordered(T x, T y) __NOEXC {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_Unordered<detail::rel_ret_t<T>>(x, y));
}

// int signbit (half x)
// shortn signbit (halfn x)
// igeninteger32bit signbit (genfloatf x)
// int signbit (double)
// longn signbit (doublen x)
template <typename T,
          typename = detail::enable_if_t<detail::is_genfloat<T>::value, T>>
detail::common_rel_ret_t<T> signbit(T x) __NOEXC {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_SignBitSet<detail::rel_ret_t<T>>(x));
}

// int any (sigeninteger x)
template <typename T>
detail::enable_if_t<detail::is_sigeninteger<T>::value, int> any(T x) __NOEXC {
  return detail::Boolean<1>(int(detail::msbIsSet(x)));
}

// int any (vigeninteger x)
template <typename T>
detail::enable_if_t<detail::is_vigeninteger<T>::value, int> any(T x) __NOEXC {
  return detail::rel_sign_bit_test_ret_t<T>(
      __sycl_std::__invoke_Any<detail::rel_sign_bit_test_ret_t<T>>(
          detail::rel_sign_bit_test_arg_t<T>(x)));
}

// int all (sigeninteger x)
template <typename T>
detail::enable_if_t<detail::is_sigeninteger<T>::value, int> all(T x) __NOEXC {
  return detail::Boolean<1>(int(detail::msbIsSet(x)));
}

// int all (vigeninteger x)
template <typename T>
detail::enable_if_t<detail::is_vigeninteger<T>::value, int> all(T x) __NOEXC {
  return detail::rel_sign_bit_test_ret_t<T>(
      __sycl_std::__invoke_All<detail::rel_sign_bit_test_ret_t<T>>(
          detail::rel_sign_bit_test_arg_t<T>(x)));
}

// gentype bitselect (gentype a, gentype b, gentype c)
template <typename T>
detail::enable_if_t<detail::is_gentype<T>::value, T> bitselect(T a, T b,
                                                               T c) __NOEXC {
  return __sycl_std::__invoke_bitselect<T>(a, b, c);
}

// geninteger select (geninteger a, geninteger b, igeninteger c)
template <typename T, typename T2>
detail::enable_if_t<
    detail::is_geninteger<T>::value && detail::is_igeninteger<T2>::value, T>
select(T a, T b, T2 c) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_select<T>(a, b, c);
}

// geninteger select (geninteger a, geninteger b, ugeninteger c)
template <typename T, typename T2>
detail::enable_if_t<
    detail::is_geninteger<T>::value && detail::is_ugeninteger<T2>::value, T>
select(T a, T b, T2 c) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_select<T>(a, b, c);
}

// genfloatf select (genfloatf a, genfloatf b, genint c)
template <typename T, typename T2>
detail::enable_if_t<
    detail::is_genfloatf<T>::value && detail::is_genint<T2>::value, T>
select(T a, T b, T2 c) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_select<T>(a, b, c);
}

// genfloatf select (genfloatf a, genfloatf b, ugenint c)
template <typename T, typename T2>
detail::enable_if_t<
    detail::is_genfloatf<T>::value && detail::is_ugenint<T2>::value, T>
select(T a, T b, T2 c) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_select<T>(a, b, c);
}

// genfloatd select (genfloatd a, genfloatd b, igeninteger64 c)
template <typename T, typename T2>
detail::enable_if_t<
    detail::is_genfloatd<T>::value && detail::is_igeninteger64bit<T2>::value, T>
select(T a, T b, T2 c) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_select<T>(a, b, c);
}

// genfloatd select (genfloatd a, genfloatd b, ugeninteger64 c)
template <typename T, typename T2>
detail::enable_if_t<
    detail::is_genfloatd<T>::value && detail::is_ugeninteger64bit<T2>::value, T>
select(T a, T b, T2 c) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_select<T>(a, b, c);
}

// genfloath select (genfloath a, genfloath b, igeninteger16 c)
template <typename T, typename T2>
detail::enable_if_t<
    detail::is_genfloath<T>::value && detail::is_igeninteger16bit<T2>::value, T>
select(T a, T b, T2 c) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_select<T>(a, b, c);
}

// genfloath select (genfloath a, genfloath b, ugeninteger16 c)
template <typename T, typename T2>
detail::enable_if_t<
    detail::is_genfloath<T>::value && detail::is_ugeninteger16bit<T2>::value, T>
select(T a, T b, T2 c) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_select<T>(a, b, c);
}

namespace native {
/* ----------------- 4.13.3 Math functions. ---------------------------------*/
// genfloatf cos (genfloatf x)
template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> cos(T x) __NOEXC {
  return __sycl_std::__invoke_native_cos<T>(x);
}

// genfloatf divide (genfloatf x, genfloatf y)
template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> divide(T x,
                                                              T y) __NOEXC {
  return __sycl_std::__invoke_native_divide<T>(x, y);
}

// genfloatf exp (genfloatf x)
template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> exp(T x) __NOEXC {
  return __sycl_std::__invoke_native_exp<T>(x);
}

// genfloatf exp2 (genfloatf x)
template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> exp2(T x) __NOEXC {
  return __sycl_std::__invoke_native_exp2<T>(x);
}

// genfloatf exp10 (genfloatf x)
template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> exp10(T x) __NOEXC {
  return __sycl_std::__invoke_native_exp10<T>(x);
}

// genfloatf log (genfloatf x)
template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> log(T x) __NOEXC {
  return __sycl_std::__invoke_native_log<T>(x);
}

// genfloatf log2 (genfloatf x)
template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> log2(T x) __NOEXC {
  return __sycl_std::__invoke_native_log2<T>(x);
}

// genfloatf log10 (genfloatf x)
template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> log10(T x) __NOEXC {
  return __sycl_std::__invoke_native_log10<T>(x);
}

// genfloatf powr (genfloatf x, genfloatf y)
template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> powr(T x, T y) __NOEXC {
  return __sycl_std::__invoke_native_powr<T>(x, y);
}

// genfloatf recip (genfloatf x)
template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> recip(T x) __NOEXC {
  return __sycl_std::__invoke_native_recip<T>(x);
}

// genfloatf rsqrt (genfloatf x)
template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> rsqrt(T x) __NOEXC {
  return __sycl_std::__invoke_native_rsqrt<T>(x);
}

// genfloatf sin (genfloatf x)
template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> sin(T x) __NOEXC {
  return __sycl_std::__invoke_native_sin<T>(x);
}

// genfloatf sqrt (genfloatf x)
template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> sqrt(T x) __NOEXC {
  return __sycl_std::__invoke_native_sqrt<T>(x);
}

// genfloatf tan (genfloatf x)
template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> tan(T x) __NOEXC {
  return __sycl_std::__invoke_native_tan<T>(x);
}

} // namespace native
namespace half_precision {
/* ----------------- 4.13.3 Math functions. ---------------------------------*/
// genfloatf cos (genfloatf x)
template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> cos(T x) __NOEXC {
  return __sycl_std::__invoke_half_cos<T>(x);
}

// genfloatf divide (genfloatf x, genfloatf y)
template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> divide(T x,
                                                              T y) __NOEXC {
  return __sycl_std::__invoke_half_divide<T>(x, y);
}

// genfloatf exp (genfloatf x)
template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> exp(T x) __NOEXC {
  return __sycl_std::__invoke_half_exp<T>(x);
}

// genfloatf exp2 (genfloatf x)
template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> exp2(T x) __NOEXC {
  return __sycl_std::__invoke_half_exp2<T>(x);
}

// genfloatf exp10 (genfloatf x)
template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> exp10(T x) __NOEXC {
  return __sycl_std::__invoke_half_exp10<T>(x);
}

// genfloatf log (genfloatf x)
template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> log(T x) __NOEXC {
  return __sycl_std::__invoke_half_log<T>(x);
}

// genfloatf log2 (genfloatf x)
template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> log2(T x) __NOEXC {
  return __sycl_std::__invoke_half_log2<T>(x);
}

// genfloatf log10 (genfloatf x)
template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> log10(T x) __NOEXC {
  return __sycl_std::__invoke_half_log10<T>(x);
}

// genfloatf powr (genfloatf x, genfloatf y)
template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> powr(T x, T y) __NOEXC {
  return __sycl_std::__invoke_half_powr<T>(x, y);
}

// genfloatf recip (genfloatf x)
template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> recip(T x) __NOEXC {
  return __sycl_std::__invoke_half_recip<T>(x);
}

// genfloatf rsqrt (genfloatf x)
template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> rsqrt(T x) __NOEXC {
  return __sycl_std::__invoke_half_rsqrt<T>(x);
}

// genfloatf sin (genfloatf x)
template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> sin(T x) __NOEXC {
  return __sycl_std::__invoke_half_sin<T>(x);
}

// genfloatf sqrt (genfloatf x)
template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> sqrt(T x) __NOEXC {
  return __sycl_std::__invoke_half_sqrt<T>(x);
}

// genfloatf tan (genfloatf x)
template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> tan(T x) __NOEXC {
  return __sycl_std::__invoke_half_tan<T>(x);
}

} // namespace half_precision
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

#ifdef __SYCL_DEVICE_ONLY__
extern "C" {
extern SYCL_EXTERNAL int abs(int x);
extern SYCL_EXTERNAL long int labs(long int x);
extern SYCL_EXTERNAL long long int llabs(long long int x);

extern SYCL_EXTERNAL div_t div(int x, int y);
extern SYCL_EXTERNAL ldiv_t ldiv(long int x, long int y);
extern SYCL_EXTERNAL lldiv_t lldiv(long long int x, long long int y);
extern SYCL_EXTERNAL float scalbnf(float x, int n);
extern SYCL_EXTERNAL double scalbn(double x, int n);
extern SYCL_EXTERNAL float logf(float x);
extern SYCL_EXTERNAL double log(double x);
extern SYCL_EXTERNAL float expf(float x);
extern SYCL_EXTERNAL double exp(double x);
extern SYCL_EXTERNAL float log10f(float x);
extern SYCL_EXTERNAL double log10(double x);
extern SYCL_EXTERNAL float modff(float x, float *intpart);
extern SYCL_EXTERNAL double modf(double x, double *intpart);
extern SYCL_EXTERNAL float exp2f(float x);
extern SYCL_EXTERNAL double exp2(double x);
extern SYCL_EXTERNAL float expm1f(float x);
extern SYCL_EXTERNAL double expm1(double x);
extern SYCL_EXTERNAL int ilogbf(float x);
extern SYCL_EXTERNAL int ilogb(double x);
extern SYCL_EXTERNAL float log1pf(float x);
extern SYCL_EXTERNAL double log1p(double x);
extern SYCL_EXTERNAL float log2f(float x);
extern SYCL_EXTERNAL double log2(double x);
extern SYCL_EXTERNAL float logbf(float x);
extern SYCL_EXTERNAL double logb(double x);
extern SYCL_EXTERNAL float sqrtf(float x);
extern SYCL_EXTERNAL double sqrt(double x);
extern SYCL_EXTERNAL float cbrtf(float x);
extern SYCL_EXTERNAL double cbrt(double x);
extern SYCL_EXTERNAL float erff(float x);
extern SYCL_EXTERNAL double erf(double x);
extern SYCL_EXTERNAL float erfcf(float x);
extern SYCL_EXTERNAL double erfc(double x);
extern SYCL_EXTERNAL float tgammaf(float x);
extern SYCL_EXTERNAL double tgamma(double x);
extern SYCL_EXTERNAL float lgammaf(float x);
extern SYCL_EXTERNAL double lgamma(double x);
extern SYCL_EXTERNAL float fmodf(float x, float y);
extern SYCL_EXTERNAL double fmod(double x, double y);
extern SYCL_EXTERNAL float remainderf(float x, float y);
extern SYCL_EXTERNAL double remainder(double x, double y);
extern SYCL_EXTERNAL float remquof(float x, float y, int *q);
extern SYCL_EXTERNAL double remquo(double x, double y, int *q);
extern SYCL_EXTERNAL float nextafterf(float x, float y);
extern SYCL_EXTERNAL double nextafter(double x, double y);
extern SYCL_EXTERNAL float fdimf(float x, float y);
extern SYCL_EXTERNAL double fdim(double x, double y);
extern SYCL_EXTERNAL float fmaf(float x, float y, float z);
extern SYCL_EXTERNAL double fma(double x, double y, double z);
extern SYCL_EXTERNAL float sinf(float x);
extern SYCL_EXTERNAL double sin(double x);
extern SYCL_EXTERNAL float cosf(float x);
extern SYCL_EXTERNAL double cos(double x);
extern SYCL_EXTERNAL float tanf(float x);
extern SYCL_EXTERNAL double tan(double x);
extern SYCL_EXTERNAL float asinf(float x);
extern SYCL_EXTERNAL double asin(double x);
extern SYCL_EXTERNAL float acosf(float x);
extern SYCL_EXTERNAL double acos(double x);
extern SYCL_EXTERNAL float atanf(float x);
extern SYCL_EXTERNAL double atan(double x);
extern SYCL_EXTERNAL float powf(float x, float y);
extern SYCL_EXTERNAL double pow(double x, double y);
extern SYCL_EXTERNAL float atan2f(float x, float y);
extern SYCL_EXTERNAL double atan2(double x, double y);

extern SYCL_EXTERNAL float sinhf(float x);
extern SYCL_EXTERNAL double sinh(double x);
extern SYCL_EXTERNAL float coshf(float x);
extern SYCL_EXTERNAL double cosh(double x);
extern SYCL_EXTERNAL float tanhf(float x);
extern SYCL_EXTERNAL double tanh(double x);
extern SYCL_EXTERNAL float asinhf(float x);
extern SYCL_EXTERNAL double asinh(double x);
extern SYCL_EXTERNAL float acoshf(float x);
extern SYCL_EXTERNAL double acosh(double x);
extern SYCL_EXTERNAL float atanhf(float x);
extern SYCL_EXTERNAL double atanh(double x);
extern SYCL_EXTERNAL double frexp(double x, int *exp);
extern SYCL_EXTERNAL double ldexp(double x, int exp);
extern SYCL_EXTERNAL double hypot(double x, double y);

extern SYCL_EXTERNAL void *memcpy(void *dest, const void *src, size_t n);
extern SYCL_EXTERNAL void *memset(void *dest, int c, size_t n);
extern SYCL_EXTERNAL int memcmp(const void *s1, const void *s2, size_t n);
}
#ifdef __GLIBC__
extern "C" {
extern SYCL_EXTERNAL void __assert_fail(const char *expr, const char *file,
                                        unsigned int line, const char *func);
extern SYCL_EXTERNAL float frexpf(float x, int *exp);
extern SYCL_EXTERNAL float ldexpf(float x, int exp);
extern SYCL_EXTERNAL float hypotf(float x, float y);

// MS UCRT supports most of the C standard library but <complex.h> is
// an exception.
extern SYCL_EXTERNAL float cimagf(float __complex__ z);
extern SYCL_EXTERNAL double cimag(double __complex__ z);
extern SYCL_EXTERNAL float crealf(float __complex__ z);
extern SYCL_EXTERNAL double creal(double __complex__ z);
extern SYCL_EXTERNAL float cargf(float __complex__ z);
extern SYCL_EXTERNAL double carg(double __complex__ z);
extern SYCL_EXTERNAL float cabsf(float __complex__ z);
extern SYCL_EXTERNAL double cabs(double __complex__ z);
extern SYCL_EXTERNAL float __complex__ cprojf(float __complex__ z);
extern SYCL_EXTERNAL double __complex__ cproj(double __complex__ z);
extern SYCL_EXTERNAL float __complex__ cexpf(float __complex__ z);
extern SYCL_EXTERNAL double __complex__ cexp(double __complex__ z);
extern SYCL_EXTERNAL float __complex__ clogf(float __complex__ z);
extern SYCL_EXTERNAL double __complex__ clog(double __complex__ z);
extern SYCL_EXTERNAL float __complex__ cpowf(float __complex__ z);
extern SYCL_EXTERNAL double __complex__ cpow(double __complex__ z);
extern SYCL_EXTERNAL float __complex__ csqrtf(float __complex__ z);
extern SYCL_EXTERNAL double __complex__ csqrt(double __complex__ z);
extern SYCL_EXTERNAL float __complex__ csinhf(float __complex__ z);
extern SYCL_EXTERNAL double __complex__ csinh(double __complex__ z);
extern SYCL_EXTERNAL float __complex__ ccoshf(float __complex__ z);
extern SYCL_EXTERNAL double __complex__ ccosh(double __complex__ z);
extern SYCL_EXTERNAL float __complex__ ctanhf(float __complex__ z);
extern SYCL_EXTERNAL double __complex__ ctanh(double __complex__ z);
extern SYCL_EXTERNAL float __complex__ csinf(float __complex__ z);
extern SYCL_EXTERNAL double __complex__ csin(double __complex__ z);
extern SYCL_EXTERNAL float __complex__ ccosf(float __complex__ z);
extern SYCL_EXTERNAL double __complex__ ccos(double __complex__ z);
extern SYCL_EXTERNAL float __complex__ ctanf(float __complex__ z);
extern SYCL_EXTERNAL double __complex__ ctan(double __complex__ z);
extern SYCL_EXTERNAL float __complex__ cacosf(float __complex__ z);
extern SYCL_EXTERNAL double __complex__ cacos(double __complex__ z);
extern SYCL_EXTERNAL float __complex__ cacoshf(float __complex__ z);
extern SYCL_EXTERNAL double __complex__ cacosh(double __complex__ z);
extern SYCL_EXTERNAL float __complex__ casinf(float __complex__ z);
extern SYCL_EXTERNAL double __complex__ casin(double __complex__ z);
extern SYCL_EXTERNAL float __complex__ casinhf(float __complex__ z);
extern SYCL_EXTERNAL double __complex__ casinh(double __complex__ z);
extern SYCL_EXTERNAL float __complex__ catanf(float __complex__ z);
extern SYCL_EXTERNAL double __complex__ catan(double __complex__ z);
extern SYCL_EXTERNAL float __complex__ catanhf(float __complex__ z);
extern SYCL_EXTERNAL double __complex__ catanh(double __complex__ z);
extern SYCL_EXTERNAL float __complex__ cpolarf(float rho, float theta);
extern SYCL_EXTERNAL double __complex__ cpolar(double rho, double theta);
extern SYCL_EXTERNAL float __complex__ __mulsc3(float a, float b, float c,
                                                float d);
extern SYCL_EXTERNAL double __complex__ __muldc3(double a, double b, double c,
                                                 double d);
extern SYCL_EXTERNAL float __complex__ __divsc3(float a, float b, float c,
                                                float d);
extern SYCL_EXTERNAL double __complex__ __divdc3(float a, float b, float c,
                                                 float d);
}
#elif defined(_WIN32)
extern "C" {
// TODO: documented C runtime library APIs must be recognized as
//       builtins by FE. This includes _dpcomp, _dsign, _dtest,
//       _fdpcomp, _fdsign, _fdtest, _hypotf, _wassert.
//       APIs used by STL, such as _Cosh, are undocumented, even though
//       they are open-sourced. Recognizing them as builtins is not
//       straightforward currently.
extern SYCL_EXTERNAL double _Cosh(double x, double y);
extern SYCL_EXTERNAL int _dpcomp(double x, double y);
extern SYCL_EXTERNAL int _dsign(double x);
extern SYCL_EXTERNAL short _Dtest(double *px);
extern SYCL_EXTERNAL short _dtest(double *px);
extern SYCL_EXTERNAL short _Exp(double *px, double y, short eoff);
extern SYCL_EXTERNAL float _FCosh(float x, float y);
extern SYCL_EXTERNAL int _fdpcomp(float x, float y);
extern SYCL_EXTERNAL int _fdsign(float x);
extern SYCL_EXTERNAL short _FDtest(float *px);
extern SYCL_EXTERNAL short _fdtest(float *px);
extern SYCL_EXTERNAL short _FExp(float *px, float y, short eoff);
extern SYCL_EXTERNAL float _FSinh(float x, float y);
extern SYCL_EXTERNAL double _Sinh(double x, double y);
extern SYCL_EXTERNAL float _hypotf(float x, float y);
extern SYCL_EXTERNAL void _wassert(const wchar_t *wexpr, const wchar_t *wfile,
                                   unsigned line);
}
#endif
#endif // __SYCL_DEVICE_ONLY__

#undef __NOEXC
