//==--- builtins_legacy_scalar.hpp - Old SYCL built-in scalar definitions --==//
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
#include <sycl/detail/boolean.hpp>             // for Boolean
#include <sycl/detail/builtins.hpp>            // for __invoke_select, __in...
#include <sycl/detail/defines_elementary.hpp>  // for __SYCL_ALWAYS_INLINE
#include <sycl/detail/generic_type_traits.hpp> // for is_svgenfloat, is_sge...
#include <sycl/detail/type_list.hpp>           // for is_contained, type_list
#include <sycl/half_type.hpp>                  // for half, intel
#include <sycl/multi_ptr.hpp>                  // for address_space_cast

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

// svgenfloat acos (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> acos(T x) {
  return __sycl_std::__invoke_acos<T>(x);
}

// svgenfloat acosh (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> acosh(T x) {
  return __sycl_std::__invoke_acosh<T>(x);
}

// svgenfloat acospi (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> acospi(T x) {
  return __sycl_std::__invoke_acospi<T>(x);
}

// svgenfloat asin (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> asin(T x) {
  return __sycl_std::__invoke_asin<T>(x);
}

// svgenfloat asinh (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> asinh(T x) {
  return __sycl_std::__invoke_asinh<T>(x);
}

// svgenfloat asinpi (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> asinpi(T x) {
  return __sycl_std::__invoke_asinpi<T>(x);
}

// svgenfloat atan (svgenfloat y_over_x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> atan(T y_over_x) {
  return __sycl_std::__invoke_atan<T>(y_over_x);
}

// svgenfloat atan2 (svgenfloat y, svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> atan2(T y, T x) {
  return __sycl_std::__invoke_atan2<T>(y, x);
}

// svgenfloat atanh (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> atanh(T x) {
  return __sycl_std::__invoke_atanh<T>(x);
}

// svgenfloat atanpi (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> atanpi(T x) {
  return __sycl_std::__invoke_atanpi<T>(x);
}

// svgenfloat atan2pi (svgenfloat y, svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> atan2pi(T y, T x) {
  return __sycl_std::__invoke_atan2pi<T>(y, x);
}

// svgenfloat cbrt (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> cbrt(T x) {
  return __sycl_std::__invoke_cbrt<T>(x);
}

// svgenfloat ceil (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> ceil(T x) {
  return __sycl_std::__invoke_ceil<T>(x);
}

// svgenfloat copysign (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> copysign(T x, T y) {
  return __sycl_std::__invoke_copysign<T>(x, y);
}

// svgenfloat cos (svgenfloat x)
template <typename T> std::enable_if_t<__FAST_MATH_GENFLOAT(T), T> cos(T x) {
  return __sycl_std::__invoke_cos<T>(x);
}

// svgenfloat cosh (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> cosh(T x) {
  return __sycl_std::__invoke_cosh<T>(x);
}

// svgenfloat cospi (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> cospi(T x) {
  return __sycl_std::__invoke_cospi<T>(x);
}

// svgenfloat erfc (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> erfc(T x) {
  return __sycl_std::__invoke_erfc<T>(x);
}

// svgenfloat erf (svgenfloat x)
template <typename T> std::enable_if_t<detail::is_svgenfloat_v<T>, T> erf(T x) {
  return __sycl_std::__invoke_erf<T>(x);
}

// svgenfloat exp (svgenfloat x )
template <typename T> std::enable_if_t<__FAST_MATH_GENFLOAT(T), T> exp(T x) {
  return __sycl_std::__invoke_exp<T>(x);
}

// svgenfloat exp2 (svgenfloat x)
template <typename T> std::enable_if_t<__FAST_MATH_GENFLOAT(T), T> exp2(T x) {
  return __sycl_std::__invoke_exp2<T>(x);
}

// svgenfloat exp10 (svgenfloat x)
template <typename T> std::enable_if_t<__FAST_MATH_GENFLOAT(T), T> exp10(T x) {
  return __sycl_std::__invoke_exp10<T>(x);
}

// svgenfloat expm1 (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> expm1(T x) {
  return __sycl_std::__invoke_expm1<T>(x);
}

// svgenfloat fabs (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> fabs(T x) {
  return __sycl_std::__invoke_fabs<T>(x);
}

// svgenfloat fdim (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> fdim(T x, T y) {
  return __sycl_std::__invoke_fdim<T>(x, y);
}

// svgenfloat floor (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> floor(T x) {
  return __sycl_std::__invoke_floor<T>(x);
}

// svgenfloat fma (svgenfloat a, svgenfloat b, svgenfloat c)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> fma(T a, T b, T c) {
  return __sycl_std::__invoke_fma<T>(a, b, c);
}

// svgenfloat fmax (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> fmax(T x, T y) {
  return __sycl_std::__invoke_fmax<T>(x, y);
}

// svgenfloat fmin (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> fmin(T x, T y) {
  return __sycl_std::__invoke_fmin<T>(x, y);
}

// svgenfloat fmod (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> fmod(T x, T y) {
  return __sycl_std::__invoke_fmod<T>(x, y);
}

// svgenfloat fract (svgenfloat x, genfloatptr iptr)
template <typename T, typename T2>
std::enable_if_t<detail::is_svgenfloat_v<T> && detail::is_genfloatptr_v<T2>, T>
fract(T x, T2 iptr) {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_fract<T>(x, iptr);
}

// svgenfloat frexp (svgenfloat x, genintptr exp)
template <typename T, typename T2>
std::enable_if_t<detail::is_svgenfloat_v<T> && detail::is_genintptr_v<T2>, T>
frexp(T x, T2 exp) {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_frexp<T>(x, exp);
}

// svgenfloat hypot (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> hypot(T x, T y) {
  return __sycl_std::__invoke_hypot<T>(x, y);
}

// genint ilogb (svgenfloat x)
template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat_v<T>, T>>
detail::change_base_type_t<T, int> ilogb(T x) {
  return __sycl_std::__invoke_ilogb<detail::change_base_type_t<T, int>>(x);
}

// float ldexp (float x, int k)
// double ldexp (double x, int k)
// half ldexp (half x, int k)
template <typename T>
std::enable_if_t<detail::is_sgenfloat_v<T>, T> ldexp(T x, int k) {
  return __sycl_std::__invoke_ldexp<T>(x, k);
}

// svgenfloat lgamma (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> lgamma(T x) {
  return __sycl_std::__invoke_lgamma<T>(x);
}

// svgenfloat lgamma_r (svgenfloat x, genintptr signp)
template <typename T, typename T2>
std::enable_if_t<detail::is_svgenfloat_v<T> && detail::is_genintptr_v<T2>, T>
lgamma_r(T x, T2 signp) {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_lgamma_r<T>(x, signp);
}

// svgenfloat log (svgenfloat x)
template <typename T> std::enable_if_t<__FAST_MATH_GENFLOAT(T), T> log(T x) {
  return __sycl_std::__invoke_log<T>(x);
}

// svgenfloat log2 (svgenfloat x)
template <typename T> std::enable_if_t<__FAST_MATH_GENFLOAT(T), T> log2(T x) {
  return __sycl_std::__invoke_log2<T>(x);
}

// svgenfloat log10 (svgenfloat x)
template <typename T> std::enable_if_t<__FAST_MATH_GENFLOAT(T), T> log10(T x) {
  return __sycl_std::__invoke_log10<T>(x);
}

// svgenfloat log1p (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> log1p(T x) {
  return __sycl_std::__invoke_log1p<T>(x);
}

// svgenfloat logb (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> logb(T x) {
  return __sycl_std::__invoke_logb<T>(x);
}

// svgenfloat mad (svgenfloat a, svgenfloat b, svgenfloat c)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> mad(T a, T b, T c) {
  return __sycl_std::__invoke_mad<T>(a, b, c);
}

// svgenfloat maxmag (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> maxmag(T x, T y) {
  return __sycl_std::__invoke_maxmag<T>(x, y);
}

// svgenfloat minmag (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> minmag(T x, T y) {
  return __sycl_std::__invoke_minmag<T>(x, y);
}

// svgenfloat modf (svgenfloat x, genfloatptr iptr)
template <typename T, typename T2>
std::enable_if_t<detail::is_svgenfloat_v<T> && detail::is_genfloatptr_v<T2>, T>
modf(T x, T2 iptr) {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_modf<T>(x, iptr);
}

namespace detail {
// SYCL 2020, revision 9 modifies accepted overloads to scalar/vec/marray of
// uint16_t/uint32_t/uint64_t.
template <typename T>
inline constexpr bool is_non_deprecated_nan_type_v =
    std::is_same_v<get_elem_type_t<T>, uint16_t> ||
    std::is_same_v<get_elem_type_t<T>, uint32_t> ||
    std::is_same_v<get_elem_type_t<T>, uint64_t>;

template <typename T, typename B, typename Enable = void>
struct convert_data_type_impl;

template <typename T, typename B>
struct convert_data_type_impl<T, B, std::enable_if_t<is_sgentype_v<T>, T>> {
  B operator()(T t) { return static_cast<B>(t); }
};

template <typename T, typename B>
struct convert_data_type_impl<T, B, std::enable_if_t<is_vgentype_v<T>, T>> {
  vec<B, T::size()> operator()(T t) { return t.template convert<B>(); }
};

template <typename T, typename B>
using convert_data_type = convert_data_type_impl<T, B, T>;
} // namespace detail

template <typename T>
std::enable_if_t<detail::is_nan_type_v<T> &&
                     detail::is_non_deprecated_nan_type_v<T>,
                 detail::nan_return_t<T>>
nan(T nancode) {
  return __sycl_std::__invoke_nan<detail::nan_return_t<T>>(
      detail::convert_data_type<T, detail::nan_argument_base_t<T>>()(nancode));
}
template <typename T>
__SYCL_DEPRECATED(
    "This is a deprecated argument type for SYCL nan built-in function.")
std::enable_if_t<detail::is_nan_type_v<T> &&
                     !detail::is_non_deprecated_nan_type_v<T>,
                 detail::nan_return_t<T>> nan(T nancode) {
  return __sycl_std::__invoke_nan<detail::nan_return_t<T>>(
      detail::convert_data_type<T, detail::nan_argument_base_t<T>>()(nancode));
}

// svgenfloat nextafter (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> nextafter(T x, T y) {
  return __sycl_std::__invoke_nextafter<T>(x, y);
}

// svgenfloat pow (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> pow(T x, T y) {
  return __sycl_std::__invoke_pow<T>(x, y);
}

// svgenfloat pown (svgenfloat x, genint y)
template <typename T, typename T2>
std::enable_if_t<detail::is_svgenfloat_v<T> && detail::is_genint_v<T2>, T>
pown(T x, T2 y) {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_pown<T>(x, y);
}

// svgenfloat powr (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<__FAST_MATH_GENFLOAT(T), T> powr(T x, T y) {
  return __sycl_std::__invoke_powr<T>(x, y);
}

// svgenfloat remainder (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> remainder(T x, T y) {
  return __sycl_std::__invoke_remainder<T>(x, y);
}

// svgenfloat remquo (svgenfloat x, svgenfloat y, genintptr quo)
template <typename T, typename T2>
std::enable_if_t<detail::is_svgenfloat_v<T> && detail::is_genintptr_v<T2>, T>
remquo(T x, T y, T2 quo) {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_remquo<T>(x, y, quo);
}

// svgenfloat rint (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> rint(T x) {
  return __sycl_std::__invoke_rint<T>(x);
}

// svgenfloat rootn (svgenfloat x, genint y)
template <typename T, typename T2>
std::enable_if_t<detail::is_svgenfloat_v<T> && detail::is_genint_v<T2>, T>
rootn(T x, T2 y) {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_rootn<T>(x, y);
}

// svgenfloat round (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> round(T x) {
  return __sycl_std::__invoke_round<T>(x);
}

// svgenfloat rsqrt (svgenfloat x)
template <typename T> std::enable_if_t<__FAST_MATH_GENFLOAT(T), T> rsqrt(T x) {
  return __sycl_std::__invoke_rsqrt<T>(x);
}

// svgenfloat sin (svgenfloat x)
template <typename T> std::enable_if_t<__FAST_MATH_GENFLOAT(T), T> sin(T x) {
  return __sycl_std::__invoke_sin<T>(x);
}

// svgenfloat sincos (svgenfloat x, genfloatptr cosval)
template <typename T, typename T2>
std::enable_if_t<__FAST_MATH_GENFLOAT(T) && detail::is_genfloatptr_v<T2>, T>
sincos(T x, T2 cosval) {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_sincos<T>(x, cosval);
}

// svgenfloat sinh (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> sinh(T x) {
  return __sycl_std::__invoke_sinh<T>(x);
}

// svgenfloat sinpi (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> sinpi(T x) {
  return __sycl_std::__invoke_sinpi<T>(x);
}

// svgenfloat sqrt (svgenfloat x)
template <typename T> std::enable_if_t<__FAST_MATH_GENFLOAT(T), T> sqrt(T x) {
  return __sycl_std::__invoke_sqrt<T>(x);
}

// svgenfloat tan (svgenfloat x)
template <typename T> std::enable_if_t<__FAST_MATH_GENFLOAT(T), T> tan(T x) {
  return __sycl_std::__invoke_tan<T>(x);
}

// svgenfloat tanh (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> tanh(T x) {
  return __sycl_std::__invoke_tanh<T>(x);
}

// svgenfloat tanpi (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> tanpi(T x) {
  return __sycl_std::__invoke_tanpi<T>(x);
}

// svgenfloat tgamma (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> tgamma(T x) {
  return __sycl_std::__invoke_tgamma<T>(x);
}

// svgenfloat trunc (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> trunc(T x) {
  return __sycl_std::__invoke_trunc<T>(x);
}

/* --------------- 4.13.5 Common functions. ---------------------------------*/
// svgenfloat clamp (svgenfloat x, svgenfloat minval, svgenfloat maxval)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> clamp(T x, T minval, T maxval) {
  return __sycl_std::__invoke_fclamp<T>(x, minval, maxval);
}

// svgenfloat degrees (svgenfloat radians)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> degrees(T radians) {
  return __sycl_std::__invoke_degrees<T>(radians);
}

// svgenfloat abs (svgenfloat x)
template <typename T>
__SYCL_DEPRECATED("abs for floating point types is non-standard and has been "
                  "deprecated. Please use fabs instead.")
std::enable_if_t<detail::is_svgenfloat_v<T>, T> abs(T x) {
  return __sycl_std::__invoke_fabs<T>(x);
}

// svgenfloat max (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T>(max)(T x, T y) {
  return __sycl_std::__invoke_fmax_common<T>(x, y);
}

// svgenfloat min (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T>(min)(T x, T y) {
  return __sycl_std::__invoke_fmin_common<T>(x, y);
}

// svgenfloat mix (svgenfloat x, svgenfloat y, svgenfloat a)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> mix(T x, T y, T a) {
  return __sycl_std::__invoke_mix<T>(x, y, a);
}

// svgenfloat radians (svgenfloat degrees)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> radians(T degrees) {
  return __sycl_std::__invoke_radians<T>(degrees);
}

// svgenfloat step (svgenfloat edge, svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> step(T edge, T x) {
  return __sycl_std::__invoke_step<T>(edge, x);
}

// svgenfloat smoothstep (svgenfloat edge0, svgenfloat edge1, svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> smoothstep(T edge0, T edge1,
                                                           T x) {
  return __sycl_std::__invoke_smoothstep<T>(edge0, edge1, x);
}

// svgenfloat sign (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat_v<T>, T> sign(T x) {
  return __sycl_std::__invoke_sign<T>(x);
}

/* --------------- 4.13.4 Integer functions. --------------------------------*/
// ugeninteger abs (geninteger x)
template <typename T>
std::enable_if_t<detail::is_ugeninteger_v<T>, T> abs(T x) {
  return __sycl_std::__invoke_u_abs<T>(x);
}

// ugeninteger abs_diff (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_ugeninteger_v<T>, T> abs_diff(T x, T y) {
  return __sycl_std::__invoke_u_abs_diff<T>(x, y);
}

// ugeninteger abs_diff (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_igeninteger_v<T>, detail::make_unsigned_t<T>>
abs_diff(T x, T y) {
  return __sycl_std::__invoke_s_abs_diff<detail::make_unsigned_t<T>>(x, y);
}

// geninteger add_sat (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_igeninteger_v<T>, T> add_sat(T x, T y) {
  return __sycl_std::__invoke_s_add_sat<T>(x, y);
}

// geninteger add_sat (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_ugeninteger_v<T>, T> add_sat(T x, T y) {
  return __sycl_std::__invoke_u_add_sat<T>(x, y);
}

// geninteger hadd (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_igeninteger_v<T>, T> hadd(T x, T y) {
  return __sycl_std::__invoke_s_hadd<T>(x, y);
}

// geninteger hadd (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_ugeninteger_v<T>, T> hadd(T x, T y) {
  return __sycl_std::__invoke_u_hadd<T>(x, y);
}

// geninteger rhadd (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_igeninteger_v<T>, T> rhadd(T x, T y) {
  return __sycl_std::__invoke_s_rhadd<T>(x, y);
}

// geninteger rhadd (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_ugeninteger_v<T>, T> rhadd(T x, T y) {
  return __sycl_std::__invoke_u_rhadd<T>(x, y);
}

// geninteger clamp (geninteger x, geninteger minval, geninteger maxval)
template <typename T>
std::enable_if_t<detail::is_igeninteger_v<T>, T> clamp(T x, T minval,
                                                       T maxval) {
  return __sycl_std::__invoke_s_clamp<T>(x, minval, maxval);
}

// geninteger clamp (geninteger x, geninteger minval, geninteger maxval)
template <typename T>
std::enable_if_t<detail::is_ugeninteger_v<T>, T> clamp(T x, T minval,
                                                       T maxval) {
  return __sycl_std::__invoke_u_clamp<T>(x, minval, maxval);
}

// geninteger clz (geninteger x)
template <typename T> std::enable_if_t<detail::is_geninteger_v<T>, T> clz(T x) {
  return __sycl_std::__invoke_clz<T>(x);
}

// geninteger ctz (geninteger x)
template <typename T> std::enable_if_t<detail::is_geninteger_v<T>, T> ctz(T x) {
  return __sycl_std::__invoke_ctz<T>(x);
}

// geninteger mad_hi (geninteger a, geninteger b, geninteger c)
template <typename T>
std::enable_if_t<detail::is_igeninteger_v<T>, T> mad_hi(T x, T y, T z) {
  return __sycl_std::__invoke_s_mad_hi<T>(x, y, z);
}

// geninteger mad_hi (geninteger a, geninteger b, geninteger c)
template <typename T>
std::enable_if_t<detail::is_ugeninteger_v<T>, T> mad_hi(T x, T y, T z) {
  return __sycl_std::__invoke_u_mad_hi<T>(x, y, z);
}

// geninteger mad_sat (geninteger a, geninteger b, geninteger c)
template <typename T>
std::enable_if_t<detail::is_igeninteger_v<T>, T> mad_sat(T a, T b, T c) {
  return __sycl_std::__invoke_s_mad_sat<T>(a, b, c);
}

// geninteger mad_sat (geninteger a, geninteger b, geninteger c)
template <typename T>
std::enable_if_t<detail::is_ugeninteger_v<T>, T> mad_sat(T a, T b, T c) {
  return __sycl_std::__invoke_u_mad_sat<T>(a, b, c);
}

// igeninteger max (igeninteger x, igeninteger y)
template <typename T>
std::enable_if_t<detail::is_igeninteger_v<T>, T>(max)(T x, T y) {
  return __sycl_std::__invoke_s_max<T>(x, y);
}

// ugeninteger max (ugeninteger x, ugeninteger y)
template <typename T>
std::enable_if_t<detail::is_ugeninteger_v<T>, T>(max)(T x, T y) {
  return __sycl_std::__invoke_u_max<T>(x, y);
}

// igeninteger min (igeninteger x, igeninteger y)
template <typename T>
std::enable_if_t<detail::is_igeninteger_v<T>, T>(min)(T x, T y) {
  return __sycl_std::__invoke_s_min<T>(x, y);
}

// ugeninteger min (ugeninteger x, ugeninteger y)
template <typename T>
std::enable_if_t<detail::is_ugeninteger_v<T>, T>(min)(T x, T y) {
  return __sycl_std::__invoke_u_min<T>(x, y);
}

// geninteger mul_hi (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_igeninteger_v<T>, T> mul_hi(T x, T y) {
  return __sycl_std::__invoke_s_mul_hi<T>(x, y);
}

// geninteger mul_hi (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_ugeninteger_v<T>, T> mul_hi(T x, T y) {
  return __sycl_std::__invoke_u_mul_hi<T>(x, y);
}

// geninteger rotate (geninteger v, geninteger i)
template <typename T>
std::enable_if_t<detail::is_geninteger_v<T>, T> rotate(T v, T i) {
  return __sycl_std::__invoke_rotate<T>(v, i);
}

// geninteger sub_sat (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_igeninteger_v<T>, T> sub_sat(T x, T y) {
  return __sycl_std::__invoke_s_sub_sat<T>(x, y);
}

// geninteger sub_sat (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_ugeninteger_v<T>, T> sub_sat(T x, T y) {
  return __sycl_std::__invoke_u_sub_sat<T>(x, y);
}

// ugeninteger16bit upsample (ugeninteger8bit hi, ugeninteger8bit lo)
template <typename T>
std::enable_if_t<detail::is_ugeninteger8bit_v<T>, detail::make_larger_t<T>>
upsample(T hi, T lo) {
  return __sycl_std::__invoke_u_upsample<detail::make_larger_t<T>>(hi, lo);
}

// igeninteger16bit upsample (igeninteger8bit hi, ugeninteger8bit lo)
template <typename T, typename T2>
std::enable_if_t<detail::is_igeninteger8bit_v<T> &&
                     detail::is_ugeninteger8bit_v<T2>,
                 detail::make_larger_t<T>>
upsample(T hi, T2 lo) {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_s_upsample<detail::make_larger_t<T>>(hi, lo);
}

// ugeninteger32bit upsample (ugeninteger16bit hi, ugeninteger16bit lo)
template <typename T>
std::enable_if_t<detail::is_ugeninteger16bit_v<T>, detail::make_larger_t<T>>
upsample(T hi, T lo) {
  return __sycl_std::__invoke_u_upsample<detail::make_larger_t<T>>(hi, lo);
}

// igeninteger32bit upsample (igeninteger16bit hi, ugeninteger16bit lo)
template <typename T, typename T2>
std::enable_if_t<detail::is_igeninteger16bit_v<T> &&
                     detail::is_ugeninteger16bit_v<T2>,
                 detail::make_larger_t<T>>
upsample(T hi, T2 lo) {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_s_upsample<detail::make_larger_t<T>>(hi, lo);
}

// ugeninteger64bit upsample (ugeninteger32bit hi, ugeninteger32bit lo)
template <typename T>
std::enable_if_t<detail::is_ugeninteger32bit_v<T>, detail::make_larger_t<T>>
upsample(T hi, T lo) {
  return __sycl_std::__invoke_u_upsample<detail::make_larger_t<T>>(hi, lo);
}

// igeninteger64bit upsample (igeninteger32bit hi, ugeninteger32bit lo)
template <typename T, typename T2>
std::enable_if_t<detail::is_igeninteger32bit_v<T> &&
                     detail::is_ugeninteger32bit_v<T2>,
                 detail::make_larger_t<T>>
upsample(T hi, T2 lo) {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_s_upsample<detail::make_larger_t<T>>(hi, lo);
}

// geninteger popcount (geninteger x)
template <typename T>
std::enable_if_t<detail::is_geninteger_v<T>, T> popcount(T x) {
  return __sycl_std::__invoke_popcount<T>(x);
}

// geninteger32bit mad24 (geninteger32bit x, geninteger32bit y,
// geninteger32bit z)
template <typename T>
std::enable_if_t<detail::is_igeninteger32bit_v<T>, T> mad24(T x, T y, T z) {
  return __sycl_std::__invoke_s_mad24<T>(x, y, z);
}

// geninteger32bit mad24 (geninteger32bit x, geninteger32bit y,
// geninteger32bit z)
template <typename T>
std::enable_if_t<detail::is_ugeninteger32bit_v<T>, T> mad24(T x, T y, T z) {
  return __sycl_std::__invoke_u_mad24<T>(x, y, z);
}

// geninteger32bit mul24 (geninteger32bit x, geninteger32bit y)
template <typename T>
std::enable_if_t<detail::is_igeninteger32bit_v<T>, T> mul24(T x, T y) {
  return __sycl_std::__invoke_s_mul24<T>(x, y);
}

// geninteger32bit mul24 (geninteger32bit x, geninteger32bit y)
template <typename T>
std::enable_if_t<detail::is_ugeninteger32bit_v<T>, T> mul24(T x, T y) {
  return __sycl_std::__invoke_u_mul24<T>(x, y);
}

/* --------------- 4.13.6 Geometric Functions. ------------------------------*/
// float dot (float p0, float p1)
// double dot (double p0, double p1)
// half dot (half p0, half p1)
template <typename T>
std::enable_if_t<detail::is_sgenfloat_v<T>, T> dot(T p0, T p1) {
  return p0 * p1;
}

/* SYCL 1.2.1 ---- 4.13.7 Relational functions. -----------------------------*/
/* SYCL 2020  ---- 4.17.9 Relational functions. -----------------------------*/

template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat_v<T>, T>>
detail::common_rel_ret_t<T> isequal(T x, T y) {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_FOrdEqual<detail::internal_rel_ret_t<T>>(x, y));
}

template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat_v<T>, T>>
detail::common_rel_ret_t<T> isnotequal(T x, T y) {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_FUnordNotEqual<detail::internal_rel_ret_t<T>>(x, y));
}

template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat_v<T>, T>>
detail::common_rel_ret_t<T> isgreater(T x, T y) {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_FOrdGreaterThan<detail::internal_rel_ret_t<T>>(x,
                                                                          y));
}

template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat_v<T>, T>>
detail::common_rel_ret_t<T> isgreaterequal(T x, T y) {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_FOrdGreaterThanEqual<detail::internal_rel_ret_t<T>>(
          x, y));
}

template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat_v<T>, T>>
detail::common_rel_ret_t<T> isless(T x, T y) {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_FOrdLessThan<detail::internal_rel_ret_t<T>>(x, y));
}

template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat_v<T>, T>>
detail::common_rel_ret_t<T> islessequal(T x, T y) {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_FOrdLessThanEqual<detail::internal_rel_ret_t<T>>(x,
                                                                            y));
}

template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat_v<T>, T>>
detail::common_rel_ret_t<T> islessgreater(T x, T y) {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_FOrdNotEqual<detail::internal_rel_ret_t<T>>(x, y));
}

template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat_v<T>, T>>
detail::common_rel_ret_t<T> isfinite(T x) {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_IsFinite<detail::internal_rel_ret_t<T>>(x));
}

template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat_v<T>, T>>
detail::common_rel_ret_t<T> isinf(T x) {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_IsInf<detail::internal_rel_ret_t<T>>(x));
}

template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat_v<T>, T>>
detail::common_rel_ret_t<T> isnan(T x) {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_IsNan<detail::internal_rel_ret_t<T>>(x));
}

template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat_v<T>, T>>
detail::common_rel_ret_t<T> isnormal(T x) {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_IsNormal<detail::internal_rel_ret_t<T>>(x));
}

template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat_v<T>, T>>
detail::common_rel_ret_t<T> isordered(T x, T y) {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_Ordered<detail::internal_rel_ret_t<T>>(x, y));
}

template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat_v<T>, T>>
detail::common_rel_ret_t<T> isunordered(T x, T y) {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_Unordered<detail::internal_rel_ret_t<T>>(x, y));
}

template <typename T,
          typename = std::enable_if_t<detail::is_svgenfloat_v<T>, T>>
detail::common_rel_ret_t<T> signbit(T x) {
  return detail::RelConverter<T>::apply(
      __sycl_std::__invoke_SignBitSet<detail::internal_rel_ret_t<T>>(x));
}

// bool any (sigeninteger x)
template <typename T>
std::enable_if_t<detail::is_sigeninteger_v<T>, bool> any(T x) {
  return detail::msbIsSet(x);
}

// bool all (sigeninteger x)
template <typename T>
std::enable_if_t<detail::is_sigeninteger_v<T>, bool> all(T x) {
  return detail::msbIsSet(x);
}

// gentype bitselect (gentype a, gentype b, gentype c)
template <typename T>
std::enable_if_t<detail::is_gentype_v<T>, T> bitselect(T a, T b, T c) {
  return __sycl_std::__invoke_bitselect<T>(a, b, c);
}

// sgentype select (sgentype a, sgentype b, bool c)
template <typename T>
std::enable_if_t<detail::is_sgentype_v<T>, T> select(T a, T b, bool c) {
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
std::enable_if_t<detail::is_geninteger_v<T> && detail::is_igeninteger_v<T2>, T>
select(T a, T b, T2 c) {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_select<T>(a, b, c);
}

// geninteger select (geninteger a, geninteger b, ugeninteger c)
template <typename T, typename T2>
std::enable_if_t<detail::is_geninteger_v<T> && detail::is_ugeninteger_v<T2>, T>
select(T a, T b, T2 c) {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_select<T>(a, b, c);
}

// svgenfloatf select (svgenfloatf a, svgenfloatf b, genint c)
template <typename T, typename T2>
std::enable_if_t<detail::is_svgenfloatf_v<T> && detail::is_genint_v<T2>, T>
select(T a, T b, T2 c) {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_select<T>(a, b, c);
}

// svgenfloatf select (svgenfloatf a, svgenfloatf b, ugenint c)
template <typename T, typename T2>
std::enable_if_t<detail::is_svgenfloatf_v<T> && detail::is_ugenint_v<T2>, T>
select(T a, T b, T2 c) {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_select<T>(a, b, c);
}

// svgenfloatd select (svgenfloatd a, svgenfloatd b, igeninteger64 c)
template <typename T, typename T2>
std::enable_if_t<
    detail::is_svgenfloatd_v<T> && detail::is_igeninteger64bit_v<T2>, T>
select(T a, T b, T2 c) {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_select<T>(a, b, c);
}

// svgenfloatd select (svgenfloatd a, svgenfloatd b, ugeninteger64 c)
template <typename T, typename T2>
std::enable_if_t<
    detail::is_svgenfloatd_v<T> && detail::is_ugeninteger64bit_v<T2>, T>
select(T a, T b, T2 c) {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_select<T>(a, b, c);
}

// svgenfloath select (svgenfloath a, svgenfloath b, igeninteger16 c)
template <typename T, typename T2>
std::enable_if_t<
    detail::is_svgenfloath_v<T> && detail::is_igeninteger16bit_v<T2>, T>
select(T a, T b, T2 c) {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_select<T>(a, b, c);
}

// svgenfloath select (svgenfloath a, svgenfloath b, ugeninteger16 c)
template <typename T, typename T2>
std::enable_if_t<
    detail::is_svgenfloath_v<T> && detail::is_ugeninteger16bit_v<T2>, T>
select(T a, T b, T2 c) {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_select<T>(a, b, c);
}

namespace native {
/* ----------------- 4.13.3 Math functions. ---------------------------------*/

// svgenfloatf cos (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> cos(T x) {
  return __sycl_std::__invoke_native_cos<T>(x);
}

// svgenfloatf divide (svgenfloatf x, svgenfloatf y)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> divide(T x, T y) {
  return __sycl_std::__invoke_native_divide<T>(x, y);
}

// svgenfloatf exp (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> exp(T x) {
  return __sycl_std::__invoke_native_exp<T>(x);
}

// svgenfloatf exp2 (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> exp2(T x) {
  return __sycl_std::__invoke_native_exp2<T>(x);
}

// svgenfloatf exp10 (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> exp10(T x) {
  return __sycl_std::__invoke_native_exp10<T>(x);
}

// svgenfloatf log (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> log(T x) {
  return __sycl_std::__invoke_native_log<T>(x);
}

// svgenfloatf log2 (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> log2(T x) {
  return __sycl_std::__invoke_native_log2<T>(x);
}

// svgenfloatf log10 (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> log10(T x) {
  return __sycl_std::__invoke_native_log10<T>(x);
}

// svgenfloatf powr (svgenfloatf x, svgenfloatf y)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> powr(T x, T y) {
  return __sycl_std::__invoke_native_powr<T>(x, y);
}

// svgenfloatf recip (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> recip(T x) {
  return __sycl_std::__invoke_native_recip<T>(x);
}

// svgenfloatf rsqrt (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> rsqrt(T x) {
  return __sycl_std::__invoke_native_rsqrt<T>(x);
}

// svgenfloatf sin (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> sin(T x) {
  return __sycl_std::__invoke_native_sin<T>(x);
}

// svgenfloatf sqrt (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> sqrt(T x) {
  return __sycl_std::__invoke_native_sqrt<T>(x);
}

// svgenfloatf tan (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> tan(T x) {
  return __sycl_std::__invoke_native_tan<T>(x);
}

} // namespace native
namespace half_precision {
/* ----------------- 4.13.3 Math functions. ---------------------------------*/

// svgenfloatf cos (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> cos(T x) {
  return __sycl_std::__invoke_half_cos<T>(x);
}

// svgenfloatf divide (svgenfloatf x, svgenfloatf y)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> divide(T x, T y) {
  return __sycl_std::__invoke_half_divide<T>(x, y);
}

// svgenfloatf exp (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> exp(T x) {
  return __sycl_std::__invoke_half_exp<T>(x);
}

// svgenfloatf exp2 (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> exp2(T x) {
  return __sycl_std::__invoke_half_exp2<T>(x);
}

// svgenfloatf exp10 (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> exp10(T x) {
  return __sycl_std::__invoke_half_exp10<T>(x);
}

// svgenfloatf log (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> log(T x) {
  return __sycl_std::__invoke_half_log<T>(x);
}

// svgenfloatf log2 (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> log2(T x) {
  return __sycl_std::__invoke_half_log2<T>(x);
}

// svgenfloatf log10 (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> log10(T x) {
  return __sycl_std::__invoke_half_log10<T>(x);
}

// svgenfloatf powr (svgenfloatf x, svgenfloatf y)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> powr(T x, T y) {
  return __sycl_std::__invoke_half_powr<T>(x, y);
}

// svgenfloatf recip (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> recip(T x) {
  return __sycl_std::__invoke_half_recip<T>(x);
}

// svgenfloatf rsqrt (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> rsqrt(T x) {
  return __sycl_std::__invoke_half_rsqrt<T>(x);
}

// svgenfloatf sin (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> sin(T x) {
  return __sycl_std::__invoke_half_sin<T>(x);
}

// svgenfloatf sqrt (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> sqrt(T x) {
  return __sycl_std::__invoke_half_sqrt<T>(x);
}

// svgenfloatf tan (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> tan(T x) {
  return __sycl_std::__invoke_half_tan<T>(x);
}

} // namespace half_precision

#ifdef __FAST_MATH__
/* ----------------- -ffast-math functions. ---------------------------------*/

// svgenfloatf cos (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> cos(T x) {
  return native::cos(x);
}

// svgenfloat sincos (svgenfloat x, genfloatptr cosval)
// This is a performance optimization to ensure that sincos isn't slower than a
// pair of sin/cos executed separately. Theoretically, calling non-native sincos
// might be faster than calling native::sin plus native::cos separately and we'd
// need some kind of cost model to make the right decision (and move this
// entirely to the JIT/AOT compilers). However, in practice, this simpler
// solution seems to work just fine and matches how sin/cos above are optimized
// for the fast math path.
template <typename T, typename T2>
std::enable_if_t<detail::is_svgenfloatf_v<T> && detail::is_genfloatptr_v<T2>, T>
sincos(T x, T2 cosval) {
  detail::check_vector_size<T, T2>();
  *cosval = native::cos(x);
  return native::sin(x);
}

// svgenfloatf exp (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> exp(T x) {
  return native::exp(x);
}

// svgenfloatf exp2 (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> exp2(T x) {
  return native::exp2(x);
}

// svgenfloatf exp10 (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> exp10(T x) {
  return native::exp10(x);
}

// svgenfloatf log(svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> log(T x) {
  return native::log(x);
}

// svgenfloatf log2 (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> log2(T x) {
  return native::log2(x);
}

// svgenfloatf log10 (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> log10(T x) {
  return native::log10(x);
}

// svgenfloatf powr (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> powr(T x, T y) {
  return native::powr(x, y);
}

// svgenfloatf rsqrt (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> rsqrt(T x) {
  return native::rsqrt(x);
}

// svgenfloatf sin (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> sin(T x) {
  return native::sin(x);
}

// svgenfloatf sqrt (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> sqrt(T x) {
  return native::sqrt(x);
}

// svgenfloatf tan (svgenfloatf x)
template <typename T>
std::enable_if_t<detail::is_svgenfloatf_v<T>, T> tan(T x) {
  return native::tan(x);
}

#endif // __FAST_MATH__
} // namespace _V1
} // namespace sycl
