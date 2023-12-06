//==----------- builtins_math.cpp - SYCL built-in math functions -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file defines the host versions of functions defined
// in SYCL SPEC section - 4.13.3 Math functions.

// Define _USE_MATH_DEFINES to enforce math defines of macros like M_PI in
// <cmath>. _USE_MATH_DEFINES is defined here before includes of SYCL header
// files to avoid include of <cmath> via those SYCL headers with unset
// _USE_MATH_DEFINES.
#define _USE_MATH_DEFINES

#include "builtins_helper.hpp"
#include <sycl/detail/export.hpp>

#include <cmath>

namespace s = sycl;
namespace d = s::detail;

namespace __host_std {

namespace {
template <typename T> inline T __acospi(T x) { return std::acos(x) / M_PI; }

template <typename T> inline T __asinpi(T x) { return std::asin(x) / M_PI; }

template <typename T> inline T __atanpi(T x) { return std::atan(x) / M_PI; }

template <typename T> inline T __atan2pi(T x, T y) {
  return std::atan2(x, y) / M_PI;
}

template <typename T> inline T __cospi(T x) {
  return std::sin(M_PI * (0.5 - x));
}

template <typename T> T inline __fract(T x, T *iptr) {
  T f = std::floor(x);
  *(iptr) = f;
  return std::fmin(x - f, std::nextafter(T(1.0), T(0.0)));
}

template <typename T> inline T __lgamma_r(T x, s::cl_int *signp) {
  T g = std::tgamma(x);
  *signp = std::signbit(d::cast_if_host_half(g)) ? -1 : 1;
  return std::log(std::abs(g));
}

template <typename T> inline T __mad(T a, T b, T c) { return (a * b) + c; }

template <typename T> inline T __maxmag(T x, T y) {
  if (std::fabs(x) > std::fabs(y))
    return x;
  if (std::fabs(y) > std::fabs(x))
    return y;
  return std::fmax(x, y);
}

template <typename T> inline T __minmag(T x, T y) {
  if (std::fabs(x) < std::fabs(y))
    return x;
  if (std::fabs(y) < std::fabs(x))
    return y;
  return std::fmin(x, y);
}

template <typename T> inline T __powr(T x, T y) {
  return (x >= T(0)) ? T(std::pow(x, y)) : x;
}

template <typename T> inline T __rootn(T x, s::cl_int y) {
  return std::pow(x, T(1.0) / y);
}

template <typename T> inline T __rsqrt(T x) { return T(1.0) / std::sqrt(x); }

template <typename T> inline T __sincos(T x, T *cosval) {
  (*cosval) = std::cos(x);
  return std::sin(x);
}

template <typename T> inline T __sinpi(T x) { return std::sin(M_PI * x); }

template <typename T> inline T __tanpi(T x) {
  // For uniformity, place in range [0.0, 1.0).
  double y = x - std::floor(x);
  // Flip for better accuracy.
  return 1.0 / std::tan((0.5 - y) * M_PI);
}

} // namespace

// ----------------- 4.13.3 Math functions. Host implementations ---------------
// acos
__SYCL_EXPORT s::cl_float sycl_host_acos(s::cl_float x) __NOEXC {
  return std::acos(x);
}
__SYCL_EXPORT s::cl_double sycl_host_acos(s::cl_double x) __NOEXC {
  return std::acos(x);
}
__SYCL_EXPORT s::cl_half sycl_host_acos(s::cl_half x) __NOEXC {
  return std::acos(x);
}
MAKE_1V(sycl_host_acos, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_acos, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_acos, s::cl_half, s::cl_half)

// acosh
__SYCL_EXPORT s::cl_float sycl_host_acosh(s::cl_float x) __NOEXC {
  return std::acosh(x);
}
__SYCL_EXPORT s::cl_double sycl_host_acosh(s::cl_double x) __NOEXC {
  return std::acosh(x);
}
__SYCL_EXPORT s::cl_half sycl_host_acosh(s::cl_half x) __NOEXC {
  return std::acosh(x);
}
MAKE_1V(sycl_host_acosh, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_acosh, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_acosh, s::cl_half, s::cl_half)

// acospi
__SYCL_EXPORT s::cl_float sycl_host_acospi(s::cl_float x) __NOEXC {
  return __acospi(x);
}
__SYCL_EXPORT s::cl_double sycl_host_acospi(s::cl_double x) __NOEXC {
  return __acospi(x);
}
__SYCL_EXPORT s::cl_half sycl_host_acospi(s::cl_half x) __NOEXC {
  return __acospi(x);
}
MAKE_1V(sycl_host_acospi, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_acospi, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_acospi, s::cl_half, s::cl_half)

// asin
__SYCL_EXPORT s::cl_float sycl_host_asin(s::cl_float x) __NOEXC {
  return std::asin(x);
}
__SYCL_EXPORT s::cl_double sycl_host_asin(s::cl_double x) __NOEXC {
  return std::asin(x);
}
__SYCL_EXPORT s::cl_half sycl_host_asin(s::cl_half x) __NOEXC {
  return std::asin(x);
}
MAKE_1V(sycl_host_asin, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_asin, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_asin, s::cl_half, s::cl_half)

// asinh
__SYCL_EXPORT s::cl_float sycl_host_asinh(s::cl_float x) __NOEXC {
  return std::asinh(x);
}
__SYCL_EXPORT s::cl_double sycl_host_asinh(s::cl_double x) __NOEXC {
  return std::asinh(x);
}
__SYCL_EXPORT s::cl_half sycl_host_asinh(s::cl_half x) __NOEXC {
  return std::asinh(x);
}
MAKE_1V(sycl_host_asinh, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_asinh, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_asinh, s::cl_half, s::cl_half)

// asinpi
__SYCL_EXPORT s::cl_float sycl_host_asinpi(s::cl_float x) __NOEXC {
  return __asinpi(x);
}
__SYCL_EXPORT s::cl_double sycl_host_asinpi(s::cl_double x) __NOEXC {
  return __asinpi(x);
}
__SYCL_EXPORT s::cl_half sycl_host_asinpi(s::cl_half x) __NOEXC {
  return __asinpi(x);
}
MAKE_1V(sycl_host_asinpi, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_asinpi, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_asinpi, s::cl_half, s::cl_half)

// atan
__SYCL_EXPORT s::cl_float sycl_host_atan(s::cl_float x) __NOEXC {
  return std::atan(x);
}
__SYCL_EXPORT s::cl_double sycl_host_atan(s::cl_double x) __NOEXC {
  return std::atan(x);
}
__SYCL_EXPORT s::cl_half sycl_host_atan(s::cl_half x) __NOEXC {
  return std::atan(x);
}
MAKE_1V(sycl_host_atan, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_atan, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_atan, s::cl_half, s::cl_half)

// atan2
__SYCL_EXPORT s::cl_float sycl_host_atan2(s::cl_float x,
                                          s::cl_float y) __NOEXC {
  return std::atan2(x, y);
}
__SYCL_EXPORT s::cl_double sycl_host_atan2(s::cl_double x,
                                           s::cl_double y) __NOEXC {
  return std::atan2(x, y);
}
__SYCL_EXPORT s::cl_half sycl_host_atan2(s::cl_half x, s::cl_half y) __NOEXC {
  return std::atan2(x, y);
}
MAKE_1V_2V(sycl_host_atan2, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(sycl_host_atan2, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(sycl_host_atan2, s::cl_half, s::cl_half, s::cl_half)

// atanh
__SYCL_EXPORT s::cl_float sycl_host_atanh(s::cl_float x) __NOEXC {
  return std::atanh(x);
}
__SYCL_EXPORT s::cl_double sycl_host_atanh(s::cl_double x) __NOEXC {
  return std::atanh(x);
}
__SYCL_EXPORT s::cl_half sycl_host_atanh(s::cl_half x) __NOEXC {
  return std::atanh(x);
}
MAKE_1V(sycl_host_atanh, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_atanh, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_atanh, s::cl_half, s::cl_half)

// atanpi
__SYCL_EXPORT s::cl_float sycl_host_atanpi(s::cl_float x) __NOEXC {
  return __atanpi(x);
}
__SYCL_EXPORT s::cl_double sycl_host_atanpi(s::cl_double x) __NOEXC {
  return __atanpi(x);
}
__SYCL_EXPORT s::cl_half sycl_host_atanpi(s::cl_half x) __NOEXC {
  return __atanpi(x);
}
MAKE_1V(sycl_host_atanpi, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_atanpi, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_atanpi, s::cl_half, s::cl_half)

// atan2pi
__SYCL_EXPORT s::cl_float sycl_host_atan2pi(s::cl_float x,
                                            s::cl_float y) __NOEXC {
  return __atan2pi(x, y);
}
__SYCL_EXPORT s::cl_double sycl_host_atan2pi(s::cl_double x,
                                             s::cl_double y) __NOEXC {
  return __atan2pi(x, y);
}
__SYCL_EXPORT s::cl_half sycl_host_atan2pi(s::cl_half x, s::cl_half y) __NOEXC {
  return __atan2pi(x, y);
}
MAKE_1V_2V(sycl_host_atan2pi, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(sycl_host_atan2pi, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(sycl_host_atan2pi, s::cl_half, s::cl_half, s::cl_half)

// cbrt
__SYCL_EXPORT s::cl_float sycl_host_cbrt(s::cl_float x) __NOEXC {
  return std::cbrt(x);
}
__SYCL_EXPORT s::cl_double sycl_host_cbrt(s::cl_double x) __NOEXC {
  return std::cbrt(x);
}
__SYCL_EXPORT s::cl_half sycl_host_cbrt(s::cl_half x) __NOEXC {
  return std::cbrt(x);
}
MAKE_1V(sycl_host_cbrt, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_cbrt, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_cbrt, s::cl_half, s::cl_half)

// ceil
__SYCL_EXPORT s::cl_float sycl_host_ceil(s::cl_float x) __NOEXC {
  return std::ceil(x);
}
__SYCL_EXPORT s::cl_double sycl_host_ceil(s::cl_double x) __NOEXC {
  return std::ceil(x);
}
__SYCL_EXPORT s::cl_half sycl_host_ceil(s::cl_half x) __NOEXC {
  return std::ceil(x);
}
MAKE_1V(sycl_host_ceil, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_ceil, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_ceil, s::cl_half, s::cl_half)

// copysign
__SYCL_EXPORT s::cl_float sycl_host_copysign(s::cl_float x,
                                             s::cl_float y) __NOEXC {
  return std::copysign(x, y);
}
__SYCL_EXPORT s::cl_double sycl_host_copysign(s::cl_double x,
                                              s::cl_double y) __NOEXC {
  return std::copysign(x, y);
}
__SYCL_EXPORT s::cl_half sycl_host_copysign(s::cl_half x,
                                            s::cl_half y) __NOEXC {
  return std::copysign(x, y);
}
MAKE_1V_2V(sycl_host_copysign, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(sycl_host_copysign, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(sycl_host_copysign, s::cl_half, s::cl_half, s::cl_half)

// cos
__SYCL_EXPORT s::cl_float sycl_host_cos(s::cl_float x) __NOEXC {
  return std::cos(x);
}
__SYCL_EXPORT s::cl_double sycl_host_cos(s::cl_double x) __NOEXC {
  return std::cos(x);
}
__SYCL_EXPORT s::cl_half sycl_host_cos(s::cl_half x) __NOEXC {
  return std::cos(x);
}
MAKE_1V(sycl_host_cos, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_cos, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_cos, s::cl_half, s::cl_half)

// cosh
__SYCL_EXPORT s::cl_float sycl_host_cosh(s::cl_float x) __NOEXC {
  return std::cosh(x);
}
__SYCL_EXPORT s::cl_double sycl_host_cosh(s::cl_double x) __NOEXC {
  return std::cosh(x);
}
__SYCL_EXPORT s::cl_half sycl_host_cosh(s::cl_half x) __NOEXC {
  return std::cosh(x);
}
MAKE_1V(sycl_host_cosh, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_cosh, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_cosh, s::cl_half, s::cl_half)

// cospi
__SYCL_EXPORT s::cl_float sycl_host_cospi(s::cl_float x) __NOEXC {
  return __cospi(x);
}
__SYCL_EXPORT s::cl_double sycl_host_cospi(s::cl_double x) __NOEXC {
  return __cospi(x);
}
__SYCL_EXPORT s::cl_half sycl_host_cospi(s::cl_half x) __NOEXC {
  return __cospi(x);
}
MAKE_1V(sycl_host_cospi, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_cospi, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_cospi, s::cl_half, s::cl_half)

// erfc
__SYCL_EXPORT s::cl_float sycl_host_erfc(s::cl_float x) __NOEXC {
  return std::erfc(x);
}
__SYCL_EXPORT s::cl_double sycl_host_erfc(s::cl_double x) __NOEXC {
  return std::erfc(x);
}
__SYCL_EXPORT s::cl_half sycl_host_erfc(s::cl_half x) __NOEXC {
  return std::erfc(x);
}
MAKE_1V(sycl_host_erfc, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_erfc, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_erfc, s::cl_half, s::cl_half)

// erf
__SYCL_EXPORT s::cl_float sycl_host_erf(s::cl_float x) __NOEXC {
  return std::erf(x);
}
__SYCL_EXPORT s::cl_double sycl_host_erf(s::cl_double x) __NOEXC {
  return std::erf(x);
}
__SYCL_EXPORT s::cl_half sycl_host_erf(s::cl_half x) __NOEXC {
  return std::erf(x);
}
MAKE_1V(sycl_host_erf, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_erf, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_erf, s::cl_half, s::cl_half)

// exp
__SYCL_EXPORT s::cl_float sycl_host_exp(s::cl_float x) __NOEXC {
  return std::exp(x);
}
__SYCL_EXPORT s::cl_double sycl_host_exp(s::cl_double x) __NOEXC {
  return std::exp(x);
}
__SYCL_EXPORT s::cl_half sycl_host_exp(s::cl_half x) __NOEXC {
  return std::exp(x);
}
MAKE_1V(sycl_host_exp, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_exp, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_exp, s::cl_half, s::cl_half)

// exp2
__SYCL_EXPORT s::cl_float sycl_host_exp2(s::cl_float x) __NOEXC {
  return std::exp2(x);
}
__SYCL_EXPORT s::cl_double sycl_host_exp2(s::cl_double x) __NOEXC {
  return std::exp2(x);
}
__SYCL_EXPORT s::cl_half sycl_host_exp2(s::cl_half x) __NOEXC {
  return std::exp2(x);
}
MAKE_1V(sycl_host_exp2, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_exp2, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_exp2, s::cl_half, s::cl_half)

// exp10
__SYCL_EXPORT s::cl_float sycl_host_exp10(s::cl_float x) __NOEXC {
  return std::pow(10, x);
}
__SYCL_EXPORT s::cl_double sycl_host_exp10(s::cl_double x) __NOEXC {
  return std::pow(10, x);
}
__SYCL_EXPORT s::cl_half sycl_host_exp10(s::cl_half x) __NOEXC {
  return std::pow(10, x);
}
MAKE_1V(sycl_host_exp10, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_exp10, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_exp10, s::cl_half, s::cl_half)

// expm1
__SYCL_EXPORT s::cl_float sycl_host_expm1(s::cl_float x) __NOEXC {
  return std::expm1(x);
}
__SYCL_EXPORT s::cl_double sycl_host_expm1(s::cl_double x) __NOEXC {
  return std::expm1(x);
}
__SYCL_EXPORT s::cl_half sycl_host_expm1(s::cl_half x) __NOEXC {
  return std::expm1(x);
}
MAKE_1V(sycl_host_expm1, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_expm1, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_expm1, s::cl_half, s::cl_half)

// fabs
__SYCL_EXPORT s::cl_float sycl_host_fabs(s::cl_float x) __NOEXC {
  return std::fabs(x);
}
__SYCL_EXPORT s::cl_double sycl_host_fabs(s::cl_double x) __NOEXC {
  return std::fabs(x);
}
__SYCL_EXPORT s::cl_half sycl_host_fabs(s::cl_half x) __NOEXC {
  return std::fabs(x);
}
MAKE_1V(sycl_host_fabs, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_fabs, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_fabs, s::cl_half, s::cl_half)

// fdim
__SYCL_EXPORT s::cl_float sycl_host_fdim(s::cl_float x, s::cl_float y) __NOEXC {
  return std::fdim(x, y);
}
__SYCL_EXPORT s::cl_double sycl_host_fdim(s::cl_double x,
                                          s::cl_double y) __NOEXC {
  return std::fdim(x, y);
}
__SYCL_EXPORT s::cl_half sycl_host_fdim(s::cl_half x, s::cl_half y) __NOEXC {
  return std::fdim(x, y);
}
MAKE_1V_2V(sycl_host_fdim, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(sycl_host_fdim, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(sycl_host_fdim, s::cl_half, s::cl_half, s::cl_half)

// floor
__SYCL_EXPORT s::cl_float sycl_host_floor(s::cl_float x) __NOEXC {
  return std::floor(x);
}
__SYCL_EXPORT s::cl_double sycl_host_floor(s::cl_double x) __NOEXC {
  return std::floor(x);
}
__SYCL_EXPORT s::cl_half sycl_host_floor(s::cl_half x) __NOEXC {
  return std::floor(x);
}
MAKE_1V(sycl_host_floor, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_floor, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_floor, s::cl_half, s::cl_half)

// fma
__SYCL_EXPORT s::cl_float sycl_host_fma(s::cl_float a, s::cl_float b,
                                        s::cl_float c) __NOEXC {
  return std::fma(a, b, c);
}
__SYCL_EXPORT s::cl_double sycl_host_fma(s::cl_double a, s::cl_double b,
                                         s::cl_double c) __NOEXC {
  return std::fma(a, b, c);
}
__SYCL_EXPORT s::cl_half sycl_host_fma(s::cl_half a, s::cl_half b,
                                       s::cl_half c) __NOEXC {
  return std::fma(a, b, c);
}
MAKE_1V_2V_3V(sycl_host_fma, s::cl_float, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V_3V(sycl_host_fma, s::cl_double, s::cl_double, s::cl_double,
              s::cl_double)
MAKE_1V_2V_3V(sycl_host_fma, s::cl_half, s::cl_half, s::cl_half, s::cl_half)

// fmax
__SYCL_EXPORT s::cl_float sycl_host_fmax(s::cl_float x, s::cl_float y) __NOEXC {
  return std::fmax(x, y);
}
__SYCL_EXPORT s::cl_double sycl_host_fmax(s::cl_double x,
                                          s::cl_double y) __NOEXC {
  return std::fmax(x, y);
}
__SYCL_EXPORT s::cl_half sycl_host_fmax(s::cl_half x, s::cl_half y) __NOEXC {
  return std::fmax(x, y);
}
MAKE_1V_2V(sycl_host_fmax, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(sycl_host_fmax, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(sycl_host_fmax, s::cl_half, s::cl_half, s::cl_half)

// fmin
__SYCL_EXPORT s::cl_float sycl_host_fmin(s::cl_float x, s::cl_float y) __NOEXC {
  return std::fmin(x, y);
}
__SYCL_EXPORT s::cl_double sycl_host_fmin(s::cl_double x,
                                          s::cl_double y) __NOEXC {
  return std::fmin(x, y);
}
__SYCL_EXPORT s::cl_half sycl_host_fmin(s::cl_half x, s::cl_half y) __NOEXC {
  return std::fmin(x, y);
}
MAKE_1V_2V(sycl_host_fmin, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(sycl_host_fmin, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(sycl_host_fmin, s::cl_half, s::cl_half, s::cl_half)

// fmod
__SYCL_EXPORT s::cl_float sycl_host_fmod(s::cl_float x, s::cl_float y) __NOEXC {
  return std::fmod(x, y);
}
__SYCL_EXPORT s::cl_double sycl_host_fmod(s::cl_double x,
                                          s::cl_double y) __NOEXC {
  return std::fmod(x, y);
}
__SYCL_EXPORT s::cl_half sycl_host_fmod(s::cl_half x, s::cl_half y) __NOEXC {
  return std::fmod(x, y);
}
MAKE_1V_2V(sycl_host_fmod, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(sycl_host_fmod, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(sycl_host_fmod, s::cl_half, s::cl_half, s::cl_half)

// nextafter
__SYCL_EXPORT s::cl_float sycl_host_nextafter(s::cl_float x,
                                              s::cl_float y) __NOEXC {
  return std::nextafter(x, y);
}
__SYCL_EXPORT s::cl_double sycl_host_nextafter(s::cl_double x,
                                               s::cl_double y) __NOEXC {
  return std::nextafter(x, y);
}
__SYCL_EXPORT s::cl_half sycl_host_nextafter(s::cl_half x,
                                             s::cl_half y) __NOEXC {
  if (std::isnan(d::cast_if_host_half(x)))
    return x;
  if (std::isnan(d::cast_if_host_half(y)) || x == y)
    return y;

  uint16_t x_bits = s::bit_cast<uint16_t>(x);
  uint16_t x_sign = x_bits & 0x8000;
  int16_t movement = (x > y ? -1 : 1) * (x_sign ? -1 : 1);
  if (x_bits == x_sign && movement == -1) {
    // Special case where we underflow in the decrement, in which case we turn
    // it around and flip the sign. The overflow case does not need special
    // handling.
    movement = 1;
    x_bits ^= 0x8000;
  }
  x_bits += movement;
  return s::bit_cast<s::cl_half>(x_bits);
}
MAKE_1V_2V(sycl_host_nextafter, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(sycl_host_nextafter, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(sycl_host_nextafter, s::cl_half, s::cl_half, s::cl_half)

// fract
__SYCL_EXPORT s::cl_float sycl_host_fract(s::cl_float x,
                                          s::cl_float *iptr) __NOEXC {
  return __fract(x, iptr);
}
__SYCL_EXPORT s::cl_double sycl_host_fract(s::cl_double x,
                                           s::cl_double *iptr) __NOEXC {
  return __fract(x, iptr);
}
__SYCL_EXPORT s::cl_half sycl_host_fract(s::cl_half x,
                                         s::cl_half *iptr) __NOEXC {
  return __fract(x, iptr);
}
MAKE_1V_2P(sycl_host_fract, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2P(sycl_host_fract, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2P(sycl_host_fract, s::cl_half, s::cl_half, s::cl_half)

// frexp
__SYCL_EXPORT s::cl_float sycl_host_frexp(s::cl_float x,
                                          s::cl_int *exp) __NOEXC {
  return std::frexp(x, exp);
}
__SYCL_EXPORT s::cl_double sycl_host_frexp(s::cl_double x,
                                           s::cl_int *exp) __NOEXC {
  return std::frexp(x, exp);
}
__SYCL_EXPORT s::cl_half sycl_host_frexp(s::cl_half x, s::cl_int *exp) __NOEXC {
  return std::frexp(x, exp);
}
MAKE_1V_2P(sycl_host_frexp, s::cl_float, s::cl_float, s::cl_int)
MAKE_1V_2P(sycl_host_frexp, s::cl_double, s::cl_double, s::cl_int)
MAKE_1V_2P(sycl_host_frexp, s::cl_half, s::cl_half, s::cl_int)

// hypot
__SYCL_EXPORT s::cl_float sycl_host_hypot(s::cl_float x,
                                          s::cl_float y) __NOEXC {
  return std::hypot(x, y);
}
__SYCL_EXPORT s::cl_double sycl_host_hypot(s::cl_double x,
                                           s::cl_double y) __NOEXC {
  return std::hypot(x, y);
}
__SYCL_EXPORT s::cl_half sycl_host_hypot(s::cl_half x, s::cl_half y) __NOEXC {
  return std::hypot(x, y);
}
MAKE_1V_2V(sycl_host_hypot, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(sycl_host_hypot, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(sycl_host_hypot, s::cl_half, s::cl_half, s::cl_half)

// ilogb
__SYCL_EXPORT s::cl_int sycl_host_ilogb(s::cl_float x) __NOEXC {
  return std::ilogb(x);
}
__SYCL_EXPORT s::cl_int sycl_host_ilogb(s::cl_double x) __NOEXC {
  return std::ilogb(x);
}
__SYCL_EXPORT s::cl_int sycl_host_ilogb(s::cl_half x) __NOEXC {
  return std::ilogb(x);
}
MAKE_1V(sycl_host_ilogb, s::cl_int, s::cl_float)
MAKE_1V(sycl_host_ilogb, s::cl_int, s::cl_double)
MAKE_1V(sycl_host_ilogb, s::cl_int, s::cl_half)

// ldexp
__SYCL_EXPORT s::cl_float sycl_host_ldexp(s::cl_float x, s::cl_int k) __NOEXC {
  return std::ldexp(x, k);
}
__SYCL_EXPORT s::cl_double sycl_host_ldexp(s::cl_double x,
                                           s::cl_int k) __NOEXC {
  return std::ldexp(x, k);
}
__SYCL_EXPORT s::cl_half sycl_host_ldexp(s::cl_half x, s::cl_int k) __NOEXC {
  return std::ldexp(x, k);
}
MAKE_1V_2V(sycl_host_ldexp, s::cl_float, s::cl_float, s::cl_int)
MAKE_1V_2V(sycl_host_ldexp, s::cl_double, s::cl_double, s::cl_int)
MAKE_1V_2V(sycl_host_ldexp, s::cl_half, s::cl_half, s::cl_int)

// lgamma
__SYCL_EXPORT s::cl_float sycl_host_lgamma(s::cl_float x) __NOEXC {
  return std::lgamma(x);
}
__SYCL_EXPORT s::cl_double sycl_host_lgamma(s::cl_double x) __NOEXC {
  return std::lgamma(x);
}
__SYCL_EXPORT s::cl_half sycl_host_lgamma(s::cl_half x) __NOEXC {
  return std::lgamma(x);
}
MAKE_1V(sycl_host_lgamma, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_lgamma, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_lgamma, s::cl_half, s::cl_half)

// lgamma_r
__SYCL_EXPORT s::cl_float sycl_host_lgamma_r(s::cl_float x,
                                             s::cl_int *signp) __NOEXC {
  return __lgamma_r(x, signp);
}
__SYCL_EXPORT s::cl_double sycl_host_lgamma_r(s::cl_double x,
                                              s::cl_int *signp) __NOEXC {
  return __lgamma_r(x, signp);
}
__SYCL_EXPORT s::cl_half sycl_host_lgamma_r(s::cl_half x,
                                            s::cl_int *signp) __NOEXC {
  return __lgamma_r(x, signp);
}
MAKE_1V_2P(sycl_host_lgamma_r, s::cl_float, s::cl_float, s::cl_int)
MAKE_1V_2P(sycl_host_lgamma_r, s::cl_double, s::cl_double, s::cl_int)
MAKE_1V_2P(sycl_host_lgamma_r, s::cl_half, s::cl_half, s::cl_int)

// log
__SYCL_EXPORT s::cl_float sycl_host_log(s::cl_float x) __NOEXC {
  return std::log(x);
}
__SYCL_EXPORT s::cl_double sycl_host_log(s::cl_double x) __NOEXC {
  return std::log(x);
}
__SYCL_EXPORT s::cl_half sycl_host_log(s::cl_half x) __NOEXC {
  return std::log(x);
}
MAKE_1V(sycl_host_log, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_log, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_log, s::cl_half, s::cl_half)

// log2
__SYCL_EXPORT s::cl_float sycl_host_log2(s::cl_float x) __NOEXC {
  return std::log2(x);
}
__SYCL_EXPORT s::cl_double sycl_host_log2(s::cl_double x) __NOEXC {
  return std::log2(x);
}
__SYCL_EXPORT s::cl_half sycl_host_log2(s::cl_half x) __NOEXC {
  return std::log2(x);
}
MAKE_1V(sycl_host_log2, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_log2, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_log2, s::cl_half, s::cl_half)

// log10
__SYCL_EXPORT s::cl_float sycl_host_log10(s::cl_float x) __NOEXC {
  return std::log10(x);
}
__SYCL_EXPORT s::cl_double sycl_host_log10(s::cl_double x) __NOEXC {
  return std::log10(x);
}
__SYCL_EXPORT s::cl_half sycl_host_log10(s::cl_half x) __NOEXC {
  return std::log10(x);
}
MAKE_1V(sycl_host_log10, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_log10, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_log10, s::cl_half, s::cl_half)

// log1p
__SYCL_EXPORT s::cl_float sycl_host_log1p(s::cl_float x) __NOEXC {
  return std::log1p(x);
}
__SYCL_EXPORT s::cl_double sycl_host_log1p(s::cl_double x) __NOEXC {
  return std::log1p(x);
}
__SYCL_EXPORT s::cl_half sycl_host_log1p(s::cl_half x) __NOEXC {
  return std::log1p(x);
}
MAKE_1V(sycl_host_log1p, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_log1p, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_log1p, s::cl_half, s::cl_half)

// logb
__SYCL_EXPORT s::cl_float sycl_host_logb(s::cl_float x) __NOEXC {
  return std::logb(x);
}
__SYCL_EXPORT s::cl_double sycl_host_logb(s::cl_double x) __NOEXC {
  return std::logb(x);
}
__SYCL_EXPORT s::cl_half sycl_host_logb(s::cl_half x) __NOEXC {
  return std::logb(x);
}
MAKE_1V(sycl_host_logb, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_logb, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_logb, s::cl_half, s::cl_half)

// mad
__SYCL_EXPORT s::cl_float sycl_host_mad(s::cl_float a, s::cl_float b,
                                        s::cl_float c) __NOEXC {
  return __mad(a, b, c);
}
__SYCL_EXPORT s::cl_double sycl_host_mad(s::cl_double a, s::cl_double b,
                                         s::cl_double c) __NOEXC {
  return __mad(a, b, c);
}
__SYCL_EXPORT s::cl_half sycl_host_mad(s::cl_half a, s::cl_half b,
                                       s::cl_half c) __NOEXC {
  return __mad(a, b, c);
}
MAKE_1V_2V_3V(sycl_host_mad, s::cl_float, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V_3V(sycl_host_mad, s::cl_double, s::cl_double, s::cl_double,
              s::cl_double)
MAKE_1V_2V_3V(sycl_host_mad, s::cl_half, s::cl_half, s::cl_half, s::cl_half)

// maxmag
__SYCL_EXPORT s::cl_float sycl_host_maxmag(s::cl_float x,
                                           s::cl_float y) __NOEXC {
  return __maxmag(x, y);
}
__SYCL_EXPORT s::cl_double sycl_host_maxmag(s::cl_double x,
                                            s::cl_double y) __NOEXC {
  return __maxmag(x, y);
}
__SYCL_EXPORT s::cl_half sycl_host_maxmag(s::cl_half x, s::cl_half y) __NOEXC {
  return __maxmag(x, y);
}
MAKE_1V_2V(sycl_host_maxmag, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(sycl_host_maxmag, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(sycl_host_maxmag, s::cl_half, s::cl_half, s::cl_half)

// minmag
__SYCL_EXPORT s::cl_float sycl_host_minmag(s::cl_float x,
                                           s::cl_float y) __NOEXC {
  return __minmag(x, y);
}
__SYCL_EXPORT s::cl_double sycl_host_minmag(s::cl_double x,
                                            s::cl_double y) __NOEXC {
  return __minmag(x, y);
}
__SYCL_EXPORT s::cl_half sycl_host_minmag(s::cl_half x, s::cl_half y) __NOEXC {
  return __minmag(x, y);
}
MAKE_1V_2V(sycl_host_minmag, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(sycl_host_minmag, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(sycl_host_minmag, s::cl_half, s::cl_half, s::cl_half)

// modf
__SYCL_EXPORT s::cl_float sycl_host_modf(s::cl_float x,
                                         s::cl_float *iptr) __NOEXC {
  return std::modf(x, iptr);
}
__SYCL_EXPORT s::cl_double sycl_host_modf(s::cl_double x,
                                          s::cl_double *iptr) __NOEXC {
  return std::modf(x, iptr);
}
__SYCL_EXPORT s::cl_half sycl_host_modf(s::cl_half x,
                                        s::cl_half *iptr) __NOEXC {
  float ptr_val_float = *iptr;
  auto ret = std::modf(x, &ptr_val_float);
  *iptr = ptr_val_float;
  return ret;
}
MAKE_1V_2P(sycl_host_modf, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2P(sycl_host_modf, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2P(sycl_host_modf, s::cl_half, s::cl_half, s::cl_half)

// nan
__SYCL_EXPORT s::cl_float sycl_host_nan(s::cl_uint nancode) __NOEXC {
  (void)nancode;
  return d::quiet_NaN<s::cl_float>();
}
__SYCL_EXPORT s::cl_double sycl_host_nan(s::cl_ulong nancode) __NOEXC {
  (void)nancode;
  return d::quiet_NaN<s::cl_double>();
}
__SYCL_EXPORT s::cl_half sycl_host_nan(s::cl_ushort nancode) __NOEXC {
  (void)nancode;
  return s::cl_half(d::quiet_NaN<s::cl_float>());
}
MAKE_1V(sycl_host_nan, s::cl_float, s::cl_uint)
MAKE_1V(sycl_host_nan, s::cl_double, s::cl_ulong)
MAKE_1V(sycl_host_nan, s::cl_half, s::cl_ushort)

// pow
__SYCL_EXPORT s::cl_float sycl_host_pow(s::cl_float x, s::cl_float y) __NOEXC {
  return std::pow(x, y);
}
__SYCL_EXPORT s::cl_double sycl_host_pow(s::cl_double x,
                                         s::cl_double y) __NOEXC {
  return std::pow(x, y);
}
__SYCL_EXPORT s::cl_half sycl_host_pow(s::cl_half x, s::cl_half y) __NOEXC {
  return std::pow(x, y);
}
MAKE_1V_2V(sycl_host_pow, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(sycl_host_pow, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(sycl_host_pow, s::cl_half, s::cl_half, s::cl_half)

// pown
__SYCL_EXPORT s::cl_float sycl_host_pown(s::cl_float x, s::cl_int y) __NOEXC {
  return std::pow(x, y);
}
__SYCL_EXPORT s::cl_double sycl_host_pown(s::cl_double x, s::cl_int y) __NOEXC {
  return std::pow(x, y);
}
__SYCL_EXPORT s::cl_half sycl_host_pown(s::cl_half x, s::cl_int y) __NOEXC {
  return std::pow(x, y);
}
MAKE_1V_2V(sycl_host_pown, s::cl_float, s::cl_float, s::cl_int)
MAKE_1V_2V(sycl_host_pown, s::cl_double, s::cl_double, s::cl_int)
MAKE_1V_2V(sycl_host_pown, s::cl_half, s::cl_half, s::cl_int)

// powr
__SYCL_EXPORT s::cl_float sycl_host_powr(s::cl_float x, s::cl_float y) __NOEXC {
  return __powr(x, y);
}
__SYCL_EXPORT s::cl_double sycl_host_powr(s::cl_double x,
                                          s::cl_double y) __NOEXC {
  return __powr(x, y);
}
__SYCL_EXPORT s::cl_half sycl_host_powr(s::cl_half x, s::cl_half y) __NOEXC {
  return __powr(x, y);
}
MAKE_1V_2V(sycl_host_powr, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(sycl_host_powr, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(sycl_host_powr, s::cl_half, s::cl_half, s::cl_half)

// remainder
__SYCL_EXPORT s::cl_float sycl_host_remainder(s::cl_float x,
                                              s::cl_float y) __NOEXC {
  return std::remainder(x, y);
}
__SYCL_EXPORT s::cl_double sycl_host_remainder(s::cl_double x,
                                               s::cl_double y) __NOEXC {
  return std::remainder(x, y);
}
__SYCL_EXPORT s::cl_half sycl_host_remainder(s::cl_half x,
                                             s::cl_half y) __NOEXC {
  return std::remainder(x, y);
}
MAKE_1V_2V(sycl_host_remainder, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(sycl_host_remainder, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(sycl_host_remainder, s::cl_half, s::cl_half, s::cl_half)

// remquo
__SYCL_EXPORT s::cl_float sycl_host_remquo(s::cl_float x, s::cl_float y,
                                           s::cl_int *quo) __NOEXC {
  s::cl_float rem = std::remainder(x, y);
  *quo = static_cast<int>(std::round((x - rem) / y));
  return rem;
}
__SYCL_EXPORT s::cl_double sycl_host_remquo(s::cl_double x, s::cl_double y,
                                            s::cl_int *quo) __NOEXC {
  s::cl_double rem = std::remainder(x, y);
  *quo = static_cast<int>(std::round((x - rem) / y));
  return rem;
}
__SYCL_EXPORT s::cl_half sycl_host_remquo(s::cl_half x, s::cl_half y,
                                          s::cl_int *quo) __NOEXC {
  s::cl_half rem = std::remainder(x, y);
  *quo = static_cast<int>(std::round((x - rem) / y));
  return rem;
}
MAKE_1V_2V_3P(sycl_host_remquo, s::cl_float, s::cl_float, s::cl_float,
              s::cl_int)
MAKE_1V_2V_3P(sycl_host_remquo, s::cl_double, s::cl_double, s::cl_double,
              s::cl_int)
MAKE_1V_2V_3P(sycl_host_remquo, s::cl_half, s::cl_half, s::cl_half, s::cl_int)

// rint
__SYCL_EXPORT s::cl_float sycl_host_rint(s::cl_float x) __NOEXC {
  return std::rint(x);
}
__SYCL_EXPORT s::cl_double sycl_host_rint(s::cl_double x) __NOEXC {
  return std::rint(x);
}
__SYCL_EXPORT s::cl_half sycl_host_rint(s::cl_half x) __NOEXC {
  return std::rint(x);
}
MAKE_1V(sycl_host_rint, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_rint, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_rint, s::cl_half, s::cl_half)

// rootn
__SYCL_EXPORT s::cl_float sycl_host_rootn(s::cl_float x, s::cl_int y) __NOEXC {
  return __rootn(x, y);
}
__SYCL_EXPORT s::cl_double sycl_host_rootn(s::cl_double x,
                                           s::cl_int y) __NOEXC {
  return __rootn(x, y);
}
__SYCL_EXPORT s::cl_half sycl_host_rootn(s::cl_half x, s::cl_int y) __NOEXC {
  return __rootn(x, y);
}
MAKE_1V_2V(sycl_host_rootn, s::cl_float, s::cl_float, s::cl_int)
MAKE_1V_2V(sycl_host_rootn, s::cl_double, s::cl_double, s::cl_int)
MAKE_1V_2V(sycl_host_rootn, s::cl_half, s::cl_half, s::cl_int)

// round
__SYCL_EXPORT s::cl_float sycl_host_round(s::cl_float x) __NOEXC {
  return std::round(x);
}
__SYCL_EXPORT s::cl_double sycl_host_round(s::cl_double x) __NOEXC {
  return std::round(x);
}
__SYCL_EXPORT s::cl_half sycl_host_round(s::cl_half x) __NOEXC {
  return std::round(x);
}
MAKE_1V(sycl_host_round, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_round, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_round, s::cl_half, s::cl_half)

// rsqrt
__SYCL_EXPORT s::cl_float sycl_host_rsqrt(s::cl_float x) __NOEXC {
  return __rsqrt(x);
}
__SYCL_EXPORT s::cl_double sycl_host_rsqrt(s::cl_double x) __NOEXC {
  return __rsqrt(x);
}
__SYCL_EXPORT s::cl_half sycl_host_rsqrt(s::cl_half x) __NOEXC {
  return __rsqrt(x);
}
MAKE_1V(sycl_host_rsqrt, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_rsqrt, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_rsqrt, s::cl_half, s::cl_half)

// sin
__SYCL_EXPORT s::cl_float sycl_host_sin(s::cl_float x) __NOEXC {
  return std::sin(x);
}
__SYCL_EXPORT s::cl_double sycl_host_sin(s::cl_double x) __NOEXC {
  return std::sin(x);
}
__SYCL_EXPORT s::cl_half sycl_host_sin(s::cl_half x) __NOEXC {
  return std::sin(x);
}
MAKE_1V(sycl_host_sin, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_sin, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_sin, s::cl_half, s::cl_half)

// sincos
__SYCL_EXPORT s::cl_float sycl_host_sincos(s::cl_float x,
                                           s::cl_float *cosval) __NOEXC {
  return __sincos(x, cosval);
}
__SYCL_EXPORT s::cl_double sycl_host_sincos(s::cl_double x,
                                            s::cl_double *cosval) __NOEXC {
  return __sincos(x, cosval);
}
__SYCL_EXPORT s::cl_half sycl_host_sincos(s::cl_half x,
                                          s::cl_half *cosval) __NOEXC {
  return __sincos(x, cosval);
}
MAKE_1V_2P(sycl_host_sincos, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2P(sycl_host_sincos, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2P(sycl_host_sincos, s::cl_half, s::cl_half, s::cl_half)

// sinh
__SYCL_EXPORT s::cl_float sycl_host_sinh(s::cl_float x) __NOEXC {
  return std::sinh(x);
}
__SYCL_EXPORT s::cl_double sycl_host_sinh(s::cl_double x) __NOEXC {
  return std::sinh(x);
}
__SYCL_EXPORT s::cl_half sycl_host_sinh(s::cl_half x) __NOEXC {
  return std::sinh(x);
}
MAKE_1V(sycl_host_sinh, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_sinh, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_sinh, s::cl_half, s::cl_half)

// sinpi
__SYCL_EXPORT s::cl_float sycl_host_sinpi(s::cl_float x) __NOEXC {
  return __sinpi(x);
}
__SYCL_EXPORT s::cl_double sycl_host_sinpi(s::cl_double x) __NOEXC {
  return __sinpi(x);
}
__SYCL_EXPORT s::cl_half sycl_host_sinpi(s::cl_half x) __NOEXC {
  return __sinpi(x);
}
MAKE_1V(sycl_host_sinpi, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_sinpi, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_sinpi, s::cl_half, s::cl_half)

// sqrt
__SYCL_EXPORT s::cl_float sycl_host_sqrt(s::cl_float x) __NOEXC {
  return std::sqrt(x);
}
__SYCL_EXPORT s::cl_double sycl_host_sqrt(s::cl_double x) __NOEXC {
  return std::sqrt(x);
}
__SYCL_EXPORT s::cl_half sycl_host_sqrt(s::cl_half x) __NOEXC {
  return std::sqrt(x);
}
MAKE_1V(sycl_host_sqrt, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_sqrt, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_sqrt, s::cl_half, s::cl_half)

// tan
__SYCL_EXPORT s::cl_float sycl_host_tan(s::cl_float x) __NOEXC {
  return std::tan(x);
}
__SYCL_EXPORT s::cl_double sycl_host_tan(s::cl_double x) __NOEXC {
  return std::tan(x);
}
__SYCL_EXPORT s::cl_half sycl_host_tan(s::cl_half x) __NOEXC {
  return std::tan(x);
}
MAKE_1V(sycl_host_tan, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_tan, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_tan, s::cl_half, s::cl_half)

// tanh
__SYCL_EXPORT s::cl_float sycl_host_tanh(s::cl_float x) __NOEXC {
  return std::tanh(x);
}
__SYCL_EXPORT s::cl_double sycl_host_tanh(s::cl_double x) __NOEXC {
  return std::tanh(x);
}
__SYCL_EXPORT s::cl_half sycl_host_tanh(s::cl_half x) __NOEXC {
  return std::tanh(x);
}
MAKE_1V(sycl_host_tanh, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_tanh, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_tanh, s::cl_half, s::cl_half)

// tanpi
__SYCL_EXPORT s::cl_float sycl_host_tanpi(s::cl_float x) __NOEXC {
  return __tanpi(x);
}
__SYCL_EXPORT s::cl_double sycl_host_tanpi(s::cl_double x) __NOEXC {
  return __tanpi(x);
}
__SYCL_EXPORT s::cl_half sycl_host_tanpi(s::cl_half x) __NOEXC {
  return __tanpi<float>(x);
}
MAKE_1V(sycl_host_tanpi, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_tanpi, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_tanpi, s::cl_half, s::cl_half)

// tgamma
__SYCL_EXPORT s::cl_float sycl_host_tgamma(s::cl_float x) __NOEXC {
  return std::tgamma(x);
}
__SYCL_EXPORT s::cl_double sycl_host_tgamma(s::cl_double x) __NOEXC {
  return std::tgamma(x);
}
__SYCL_EXPORT s::cl_half sycl_host_tgamma(s::cl_half x) __NOEXC {
  return std::tgamma(x);
}
MAKE_1V(sycl_host_tgamma, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_tgamma, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_tgamma, s::cl_half, s::cl_half)

// trunc
__SYCL_EXPORT s::cl_float sycl_host_trunc(s::cl_float x) __NOEXC {
  return std::trunc(x);
}
__SYCL_EXPORT s::cl_double sycl_host_trunc(s::cl_double x) __NOEXC {
  return std::trunc(x);
}
__SYCL_EXPORT s::cl_half sycl_host_trunc(s::cl_half x) __NOEXC {
  return std::trunc(x);
}
MAKE_1V(sycl_host_trunc, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_trunc, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_trunc, s::cl_half, s::cl_half)

// --------------- 4.13.3 Native Math functions. Host implementations. ---------
// native_cos
__SYCL_EXPORT s::cl_float sycl_host_native_cos(s::cl_float x) __NOEXC {
  return std::cos(x);
}
MAKE_1V(sycl_host_native_cos, s::cl_float, s::cl_float)

// native_divide
__SYCL_EXPORT s::cl_float sycl_host_native_divide(s::cl_float x,
                                                  s::cl_float y) __NOEXC {
  return x / y;
}
MAKE_1V_2V(sycl_host_native_divide, s::cl_float, s::cl_float, s::cl_float)

// native_exp
__SYCL_EXPORT s::cl_float sycl_host_native_exp(s::cl_float x) __NOEXC {
  return std::exp(x);
}
MAKE_1V(sycl_host_native_exp, s::cl_float, s::cl_float)

// native_exp2
__SYCL_EXPORT s::cl_float sycl_host_native_exp2(s::cl_float x) __NOEXC {
  return std::exp2(x);
}
MAKE_1V(sycl_host_native_exp2, s::cl_float, s::cl_float)

// native_exp10
__SYCL_EXPORT s::cl_float sycl_host_native_exp10(s::cl_float x) __NOEXC {
  return std::pow(10, x);
}
MAKE_1V(sycl_host_native_exp10, s::cl_float, s::cl_float)

// native_log
__SYCL_EXPORT s::cl_float sycl_host_native_log(s::cl_float x) __NOEXC {
  return std::log(x);
}
MAKE_1V(sycl_host_native_log, s::cl_float, s::cl_float)

// native_log2
__SYCL_EXPORT s::cl_float sycl_host_native_log2(s::cl_float x) __NOEXC {
  return std::log2(x);
}
MAKE_1V(sycl_host_native_log2, s::cl_float, s::cl_float)

// native_log10
__SYCL_EXPORT s::cl_float sycl_host_native_log10(s::cl_float x) __NOEXC {
  return std::log10(x);
}
MAKE_1V(sycl_host_native_log10, s::cl_float, s::cl_float)

// native_powr
__SYCL_EXPORT s::cl_float sycl_host_native_powr(s::cl_float x,
                                                s::cl_float y) __NOEXC {
  return (x >= 0 ? std::pow(x, y) : x);
}
MAKE_1V_2V(sycl_host_native_powr, s::cl_float, s::cl_float, s::cl_float)

// native_recip
__SYCL_EXPORT s::cl_float sycl_host_native_recip(s::cl_float x) __NOEXC {
  return 1.0 / x;
}
MAKE_1V(sycl_host_native_recip, s::cl_float, s::cl_float)

// native_rsqrt
__SYCL_EXPORT s::cl_float sycl_host_native_rsqrt(s::cl_float x) __NOEXC {
  return 1.0 / std::sqrt(x);
}
MAKE_1V(sycl_host_native_rsqrt, s::cl_float, s::cl_float)

// native_sin
__SYCL_EXPORT s::cl_float sycl_host_native_sin(s::cl_float x) __NOEXC {
  return std::sin(x);
}
MAKE_1V(sycl_host_native_sin, s::cl_float, s::cl_float)

// native_sqrt
__SYCL_EXPORT s::cl_float sycl_host_native_sqrt(s::cl_float x) __NOEXC {
  return std::sqrt(x);
}
MAKE_1V(sycl_host_native_sqrt, s::cl_float, s::cl_float)

// native_tan
__SYCL_EXPORT s::cl_float sycl_host_native_tan(s::cl_float x) __NOEXC {
  return std::tan(x);
}
MAKE_1V(sycl_host_native_tan, s::cl_float, s::cl_float)

// ---------- 4.13.3 Half Precision Math functions. Host implementations. ------
// half_cos
__SYCL_EXPORT s::cl_float sycl_host_half_cos(s::cl_float x) __NOEXC {
  return std::cos(x);
}
MAKE_1V(sycl_host_half_cos, s::cl_float, s::cl_float)

// half_divide
__SYCL_EXPORT s::cl_float sycl_host_half_divide(s::cl_float x,
                                                s::cl_float y) __NOEXC {
  return x / y;
}
MAKE_1V_2V(sycl_host_half_divide, s::cl_float, s::cl_float, s::cl_float)

// half_exp
__SYCL_EXPORT s::cl_float sycl_host_half_exp(s::cl_float x) __NOEXC {
  return std::exp(x);
}
MAKE_1V(sycl_host_half_exp, s::cl_float, s::cl_float)
// half_exp2
__SYCL_EXPORT s::cl_float sycl_host_half_exp2(s::cl_float x) __NOEXC {
  return std::exp2(x);
}
MAKE_1V(sycl_host_half_exp2, s::cl_float, s::cl_float)

// half_exp10
__SYCL_EXPORT s::cl_float sycl_host_half_exp10(s::cl_float x) __NOEXC {
  return std::pow(10, x);
}
MAKE_1V(sycl_host_half_exp10, s::cl_float, s::cl_float)
// half_log
__SYCL_EXPORT s::cl_float sycl_host_half_log(s::cl_float x) __NOEXC {
  return std::log(x);
}
MAKE_1V(sycl_host_half_log, s::cl_float, s::cl_float)

// half_log2
__SYCL_EXPORT s::cl_float sycl_host_half_log2(s::cl_float x) __NOEXC {
  return std::log2(x);
}
MAKE_1V(sycl_host_half_log2, s::cl_float, s::cl_float)

// half_log10
__SYCL_EXPORT s::cl_float sycl_host_half_log10(s::cl_float x) __NOEXC {
  return std::log10(x);
}
MAKE_1V(sycl_host_half_log10, s::cl_float, s::cl_float)

// half_powr
__SYCL_EXPORT s::cl_float sycl_host_half_powr(s::cl_float x,
                                              s::cl_float y) __NOEXC {
  return (x >= 0 ? std::pow(x, y) : x);
}
MAKE_1V_2V(sycl_host_half_powr, s::cl_float, s::cl_float, s::cl_float)

// half_recip
__SYCL_EXPORT s::cl_float sycl_host_half_recip(s::cl_float x) __NOEXC {
  return 1.0 / x;
}
MAKE_1V(sycl_host_half_recip, s::cl_float, s::cl_float)

// half_rsqrt
__SYCL_EXPORT s::cl_float sycl_host_half_rsqrt(s::cl_float x) __NOEXC {
  return 1.0 / std::sqrt(x);
}
MAKE_1V(sycl_host_half_rsqrt, s::cl_float, s::cl_float)

// half_sin
__SYCL_EXPORT s::cl_float sycl_host_half_sin(s::cl_float x) __NOEXC {
  return std::sin(x);
}
MAKE_1V(sycl_host_half_sin, s::cl_float, s::cl_float)

// half_sqrt
__SYCL_EXPORT s::cl_float sycl_host_half_sqrt(s::cl_float x) __NOEXC {
  return std::sqrt(x);
}
MAKE_1V(sycl_host_half_sqrt, s::cl_float, s::cl_float)

// half_tan
__SYCL_EXPORT s::cl_float sycl_host_half_tan(s::cl_float x) __NOEXC {
  return std::tan(x);
}
MAKE_1V(sycl_host_half_tan, s::cl_float, s::cl_float)

} // namespace __host_std
