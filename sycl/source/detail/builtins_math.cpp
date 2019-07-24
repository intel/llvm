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

#include <cmath>

namespace s = cl::sycl;
namespace d = s::detail;

namespace cl {
namespace __host_std {

namespace {
template <typename T> inline T __acospi(T x) { return std::acos(x) / M_PI; }

template <typename T> inline T __asinpi(T x) { return std::asin(x) / M_PI; }

template <typename T> inline T __atanpi(T x) { return std::atan(x) / M_PI; }

template <typename T> inline T __atan2pi(T x, T y) {
  return std::atan2(x, y) / M_PI;
}

template <typename T> inline T __cospi(T x) { return std::cos(M_PI * x); }

template <typename T> T inline __fract(T x, T *iptr) {
  T f = std::floor(x);
  *(iptr) = f;
  return std::fmin(x - f, nextafter(T(1.0), T(0.0)));
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

template <typename T> inline T __tanpi(T x) { return std::tan(M_PI * x); }

} // namespace

// ----------------- 4.13.3 Math functions. Host implementations ---------------
// acos
cl_float acos(s::cl_float x) __NOEXC { return std::acos(x); }
cl_double acos(s::cl_double x) __NOEXC { return std::acos(x); }
cl_half acos(s::cl_half x) __NOEXC { return std::acos(x); }
MAKE_1V(acos, s::cl_float, s::cl_float)
MAKE_1V(acos, s::cl_double, s::cl_double)
MAKE_1V(acos, s::cl_half, s::cl_half)

// acosh
cl_float acosh(s::cl_float x) __NOEXC { return std::acosh(x); }
cl_double acosh(s::cl_double x) __NOEXC { return std::acosh(x); }
cl_half acosh(s::cl_half x) __NOEXC { return std::acosh(x); }
MAKE_1V(acosh, s::cl_float, s::cl_float)
MAKE_1V(acosh, s::cl_double, s::cl_double)
MAKE_1V(acosh, s::cl_half, s::cl_half)

// acospi
cl_float acospi(s::cl_float x) __NOEXC { return __acospi(x); }
cl_double acospi(s::cl_double x) __NOEXC { return __acospi(x); }
cl_half acospi(s::cl_half x) __NOEXC { return __acospi(x); }
MAKE_1V(acospi, s::cl_float, s::cl_float)
MAKE_1V(acospi, s::cl_double, s::cl_double)
MAKE_1V(acospi, s::cl_half, s::cl_half)

// asin
cl_float asin(s::cl_float x) __NOEXC { return std::asin(x); }
cl_double asin(s::cl_double x) __NOEXC { return std::asin(x); }
cl_half asin(s::cl_half x) __NOEXC { return std::asin(x); }
MAKE_1V(asin, s::cl_float, s::cl_float)
MAKE_1V(asin, s::cl_double, s::cl_double)
MAKE_1V(asin, s::cl_half, s::cl_half)

// asinh
cl_float asinh(s::cl_float x) __NOEXC { return std::asinh(x); }
cl_double asinh(s::cl_double x) __NOEXC { return std::asinh(x); }
cl_half asinh(s::cl_half x) __NOEXC { return std::asinh(x); }
MAKE_1V(asinh, s::cl_float, s::cl_float)
MAKE_1V(asinh, s::cl_double, s::cl_double)
MAKE_1V(asinh, s::cl_half, s::cl_half)

// asinpi
cl_float asinpi(s::cl_float x) __NOEXC { return __asinpi(x); }
cl_double asinpi(s::cl_double x) __NOEXC { return __asinpi(x); }
cl_half asinpi(s::cl_half x) __NOEXC { return __asinpi(x); }
MAKE_1V(asinpi, s::cl_float, s::cl_float)
MAKE_1V(asinpi, s::cl_double, s::cl_double)
MAKE_1V(asinpi, s::cl_half, s::cl_half)

// atan
cl_float atan(s::cl_float x) __NOEXC { return std::atan(x); }
cl_double atan(s::cl_double x) __NOEXC { return std::atan(x); }
cl_half atan(s::cl_half x) __NOEXC { return std::atan(x); }
MAKE_1V(atan, s::cl_float, s::cl_float)
MAKE_1V(atan, s::cl_double, s::cl_double)
MAKE_1V(atan, s::cl_half, s::cl_half)

// atan2
cl_float atan2(s::cl_float x, s::cl_float y) __NOEXC {
  return std::atan2(x, y);
}
cl_double atan2(s::cl_double x, s::cl_double y) __NOEXC {
  return std::atan2(x, y);
}
cl_half atan2(s::cl_half x, s::cl_half y) __NOEXC { return std::atan2(x, y); }
MAKE_1V_2V(atan2, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(atan2, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(atan2, s::cl_half, s::cl_half, s::cl_half)

// atanh
cl_float atanh(s::cl_float x) __NOEXC { return std::atanh(x); }
cl_double atanh(s::cl_double x) __NOEXC { return std::atanh(x); }
cl_half atanh(s::cl_half x) __NOEXC { return std::atanh(x); }
MAKE_1V(atanh, s::cl_float, s::cl_float)
MAKE_1V(atanh, s::cl_double, s::cl_double)
MAKE_1V(atanh, s::cl_half, s::cl_half)

// atanpi
cl_float atanpi(s::cl_float x) __NOEXC { return __atanpi(x); }
cl_double atanpi(s::cl_double x) __NOEXC { return __atanpi(x); }
cl_half atanpi(s::cl_half x) __NOEXC { return __atanpi(x); }
MAKE_1V(atanpi, s::cl_float, s::cl_float)
MAKE_1V(atanpi, s::cl_double, s::cl_double)
MAKE_1V(atanpi, s::cl_half, s::cl_half)

// atan2pi
cl_float atan2pi(s::cl_float x, s::cl_float y) __NOEXC {
  return __atan2pi(x, y);
}
cl_double atan2pi(s::cl_double x, s::cl_double y) __NOEXC {
  return __atan2pi(x, y);
}
cl_half atan2pi(s::cl_half x, s::cl_half y) __NOEXC { return __atan2pi(x, y); }
MAKE_1V_2V(atan2pi, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(atan2pi, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(atan2pi, s::cl_half, s::cl_half, s::cl_half)

// cbrt
cl_float cbrt(s::cl_float x) __NOEXC { return std::cbrt(x); }
cl_double cbrt(s::cl_double x) __NOEXC { return std::cbrt(x); }
cl_half cbrt(s::cl_half x) __NOEXC { return std::cbrt(x); }
MAKE_1V(cbrt, s::cl_float, s::cl_float)
MAKE_1V(cbrt, s::cl_double, s::cl_double)
MAKE_1V(cbrt, s::cl_half, s::cl_half)

// ceil
cl_float ceil(s::cl_float x) __NOEXC { return std::ceil(x); }
cl_double ceil(s::cl_double x) __NOEXC { return std::ceil(x); }
cl_half ceil(s::cl_half x) __NOEXC { return std::ceil(x); }
MAKE_1V(ceil, s::cl_float, s::cl_float)
MAKE_1V(ceil, s::cl_double, s::cl_double)
MAKE_1V(ceil, s::cl_half, s::cl_half)

// copysign
cl_float copysign(s::cl_float x, s::cl_float y) __NOEXC {
  return std::copysign(x, y);
}
cl_double copysign(s::cl_double x, s::cl_double y) __NOEXC {
  return std::copysign(x, y);
}
cl_half copysign(s::cl_half x, s::cl_half y) __NOEXC {
  return std::copysign(x, y);
}
MAKE_1V_2V(copysign, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(copysign, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(copysign, s::cl_half, s::cl_half, s::cl_half)

// cos
cl_float cos(s::cl_float x) __NOEXC { return std::cos(x); }
cl_double cos(s::cl_double x) __NOEXC { return std::cos(x); }
cl_half cos(s::cl_half x) __NOEXC { return std::cos(x); }
MAKE_1V(cos, s::cl_float, s::cl_float)
MAKE_1V(cos, s::cl_double, s::cl_double)
MAKE_1V(cos, s::cl_half, s::cl_half)

// cosh
cl_float cosh(s::cl_float x) __NOEXC { return std::cosh(x); }
cl_double cosh(s::cl_double x) __NOEXC { return std::cosh(x); }
cl_half cosh(s::cl_half x) __NOEXC { return std::cosh(x); }
MAKE_1V(cosh, s::cl_float, s::cl_float)
MAKE_1V(cosh, s::cl_double, s::cl_double)
MAKE_1V(cosh, s::cl_half, s::cl_half)

// cospi
cl_float cospi(s::cl_float x) __NOEXC { return __cospi(x); }
cl_double cospi(s::cl_double x) __NOEXC { return __cospi(x); }
cl_half cospi(s::cl_half x) __NOEXC { return __cospi(x); }
MAKE_1V(cospi, s::cl_float, s::cl_float)
MAKE_1V(cospi, s::cl_double, s::cl_double)
MAKE_1V(cospi, s::cl_half, s::cl_half)

// erfc
cl_float erfc(s::cl_float x) __NOEXC { return std::erfc(x); }
cl_double erfc(s::cl_double x) __NOEXC { return std::erfc(x); }
cl_half erfc(s::cl_half x) __NOEXC { return std::erfc(x); }
MAKE_1V(erfc, s::cl_float, s::cl_float)
MAKE_1V(erfc, s::cl_double, s::cl_double)
MAKE_1V(erfc, s::cl_half, s::cl_half)

// erf
cl_float erf(s::cl_float x) __NOEXC { return std::erf(x); }
cl_double erf(s::cl_double x) __NOEXC { return std::erf(x); }
cl_half erf(s::cl_half x) __NOEXC { return std::erf(x); }
MAKE_1V(erf, s::cl_float, s::cl_float)
MAKE_1V(erf, s::cl_double, s::cl_double)
MAKE_1V(erf, s::cl_half, s::cl_half)

// exp
cl_float exp(s::cl_float x) __NOEXC { return std::exp(x); }
cl_double exp(s::cl_double x) __NOEXC { return std::exp(x); }
cl_half exp(s::cl_half x) __NOEXC { return std::exp(x); }
MAKE_1V(exp, s::cl_float, s::cl_float)
MAKE_1V(exp, s::cl_double, s::cl_double)
MAKE_1V(exp, s::cl_half, s::cl_half)

// exp2
cl_float exp2(s::cl_float x) __NOEXC { return std::exp2(x); }
cl_double exp2(s::cl_double x) __NOEXC { return std::exp2(x); }
cl_half exp2(s::cl_half x) __NOEXC { return std::exp2(x); }
MAKE_1V(exp2, s::cl_float, s::cl_float)
MAKE_1V(exp2, s::cl_double, s::cl_double)
MAKE_1V(exp2, s::cl_half, s::cl_half)

// exp10
cl_float exp10(s::cl_float x) __NOEXC { return std::pow(10, x); }
cl_double exp10(s::cl_double x) __NOEXC { return std::pow(10, x); }
cl_half exp10(s::cl_half x) __NOEXC { return std::pow(10, x); }
MAKE_1V(exp10, s::cl_float, s::cl_float)
MAKE_1V(exp10, s::cl_double, s::cl_double)
MAKE_1V(exp10, s::cl_half, s::cl_half)

// expm1
cl_float expm1(s::cl_float x) __NOEXC { return std::expm1(x); }
cl_double expm1(s::cl_double x) __NOEXC { return std::expm1(x); }
cl_half expm1(s::cl_half x) __NOEXC { return std::expm1(x); }
MAKE_1V(expm1, s::cl_float, s::cl_float)
MAKE_1V(expm1, s::cl_double, s::cl_double)
MAKE_1V(expm1, s::cl_half, s::cl_half)

// fabs
cl_float fabs(s::cl_float x) __NOEXC { return std::fabs(x); }
cl_double fabs(s::cl_double x) __NOEXC { return std::fabs(x); }
cl_half fabs(s::cl_half x) __NOEXC { return std::fabs(x); }
MAKE_1V(fabs, s::cl_float, s::cl_float)
MAKE_1V(fabs, s::cl_double, s::cl_double)
MAKE_1V(fabs, s::cl_half, s::cl_half)

// fdim
cl_float fdim(s::cl_float x, s::cl_float y) __NOEXC { return std::fdim(x, y); }
cl_double fdim(s::cl_double x, s::cl_double y) __NOEXC {
  return std::fdim(x, y);
}
cl_half fdim(s::cl_half x, s::cl_half y) __NOEXC { return std::fdim(x, y); }
MAKE_1V_2V(fdim, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(fdim, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(fdim, s::cl_half, s::cl_half, s::cl_half)

// floor
cl_float floor(s::cl_float x) __NOEXC { return std::floor(x); }
cl_double floor(s::cl_double x) __NOEXC { return std::floor(x); }
cl_half floor(s::cl_half x) __NOEXC { return std::floor(x); }
MAKE_1V(floor, s::cl_float, s::cl_float)
MAKE_1V(floor, s::cl_double, s::cl_double)
MAKE_1V(floor, s::cl_half, s::cl_half)

// fma
cl_float fma(s::cl_float a, s::cl_float b, s::cl_float c) __NOEXC {
  return std::fma(a, b, c);
}
cl_double fma(s::cl_double a, s::cl_double b, s::cl_double c) __NOEXC {
  return std::fma(a, b, c);
}
cl_half fma(s::cl_half a, s::cl_half b, s::cl_half c) __NOEXC {
  return std::fma(a, b, c);
}
MAKE_1V_2V_3V(fma, s::cl_float, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V_3V(fma, s::cl_double, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V_3V(fma, s::cl_half, s::cl_half, s::cl_half, s::cl_half)

// fmax
cl_float fmax(s::cl_float x, s::cl_float y) __NOEXC { return std::fmax(x, y); }
cl_double fmax(s::cl_double x, s::cl_double y) __NOEXC {
  return std::fmax(x, y);
}
cl_half fmax(s::cl_half x, s::cl_half y) __NOEXC { return std::fmax(x, y); }
MAKE_1V_2V(fmax, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(fmax, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(fmax, s::cl_half, s::cl_half, s::cl_half)

// fmin
cl_float fmin(s::cl_float x, s::cl_float y) __NOEXC { return std::fmin(x, y); }
cl_double fmin(s::cl_double x, s::cl_double y) __NOEXC {
  return std::fmin(x, y);
}
cl_half fmin(s::cl_half x, s::cl_half y) __NOEXC { return std::fmin(x, y); }
MAKE_1V_2V(fmin, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(fmin, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(fmin, s::cl_half, s::cl_half, s::cl_half)

// fmod
cl_float fmod(s::cl_float x, s::cl_float y) __NOEXC { return std::fmod(x, y); }
cl_double fmod(s::cl_double x, s::cl_double y) __NOEXC {
  return std::fmod(x, y);
}
cl_half fmod(s::cl_half x, s::cl_half y) __NOEXC { return std::fmod(x, y); }
MAKE_1V_2V(fmod, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(fmod, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(fmod, s::cl_half, s::cl_half, s::cl_half)

// nextafter
cl_float nextafter(s::cl_float x, s::cl_float y) __NOEXC {
  return std::nextafter(x, y);
}
cl_double nextafter(s::cl_double x, s::cl_double y) __NOEXC {
  return std::nextafter(x, y);
}
cl_half nextafter(s::cl_half x, s::cl_half y) __NOEXC {
  return std::nextafter(x, y);
}
MAKE_1V_2V(nextafter, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(nextafter, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(nextafter, s::cl_half, s::cl_half, s::cl_half)

// fract
cl_float fract(s::cl_float x, s::cl_float *iptr) __NOEXC {
  return __fract(x, iptr);
}
cl_double fract(s::cl_double x, s::cl_double *iptr) __NOEXC {
  return __fract(x, iptr);
}
cl_half fract(s::cl_half x, s::cl_half *iptr) __NOEXC {
  return __fract(x, iptr);
}
MAKE_1V_2P(fract, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2P(fract, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2P(fract, s::cl_half, s::cl_half, s::cl_half)

// frexp
cl_float frexp(s::cl_float x, s::cl_int *exp) __NOEXC {
  return std::frexp(x, exp);
}
cl_double frexp(s::cl_double x, s::cl_int *exp) __NOEXC {
  return std::frexp(x, exp);
}
cl_half frexp(s::cl_half x, s::cl_int *exp) __NOEXC {
  return std::frexp(x, exp);
}
MAKE_1V_2P(frexp, s::cl_float, s::cl_float, s::cl_int)
MAKE_1V_2P(frexp, s::cl_double, s::cl_double, s::cl_int)
MAKE_1V_2P(frexp, s::cl_half, s::cl_half, s::cl_int)

// hypot
cl_float hypot(s::cl_float x, s::cl_float y) __NOEXC {
  return std::hypot(x, y);
}
cl_double hypot(s::cl_double x, s::cl_double y) __NOEXC {
  return std::hypot(x, y);
}
cl_half hypot(s::cl_half x, s::cl_half y) __NOEXC { return std::hypot(x, y); }
MAKE_1V_2V(hypot, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(hypot, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(hypot, s::cl_half, s::cl_half, s::cl_half)

// ilogb
cl_int ilogb(s::cl_float x) __NOEXC { return std::ilogb(x); }
cl_int ilogb(s::cl_double x) __NOEXC { return std::ilogb(x); }
cl_int ilogb(s::cl_half x) __NOEXC { return std::ilogb(x); }
MAKE_1V(ilogb, s::cl_int, s::cl_float)
MAKE_1V(ilogb, s::cl_int, s::cl_double)
MAKE_1V(ilogb, s::cl_int, s::cl_half)

// ldexp
cl_float ldexp(s::cl_float x, s::cl_int k) __NOEXC { return std::ldexp(x, k); }
cl_double ldexp(s::cl_double x, s::cl_int k) __NOEXC {
  return std::ldexp(x, k);
}
cl_half ldexp(s::cl_half x, s::cl_int k) __NOEXC { return std::ldexp(x, k); }
MAKE_1V_2V(ldexp, s::cl_float, s::cl_float, s::cl_int)
MAKE_1V_2V(ldexp, s::cl_double, s::cl_double, s::cl_int)
MAKE_1V_2V(ldexp, s::cl_half, s::cl_half, s::cl_int)

// lgamma
cl_float lgamma(s::cl_float x) __NOEXC { return std::lgamma(x); }
cl_double lgamma(s::cl_double x) __NOEXC { return std::lgamma(x); }
cl_half lgamma(s::cl_half x) __NOEXC { return std::lgamma(x); }
MAKE_1V(lgamma, s::cl_float, s::cl_float)
MAKE_1V(lgamma, s::cl_double, s::cl_double)
MAKE_1V(lgamma, s::cl_half, s::cl_half)

// lgamma_r
cl_float lgamma_r(s::cl_float x, s::cl_int *signp) __NOEXC {
  return __lgamma_r(x, signp);
}
cl_double lgamma_r(s::cl_double x, s::cl_int *signp) __NOEXC {
  return __lgamma_r(x, signp);
}
cl_half lgamma_r(s::cl_half x, s::cl_int *signp) __NOEXC {
  return __lgamma_r(x, signp);
}
MAKE_1V_2P(lgamma_r, s::cl_float, s::cl_float, s::cl_int)
MAKE_1V_2P(lgamma_r, s::cl_double, s::cl_double, s::cl_int)
MAKE_1V_2P(lgamma_r, s::cl_half, s::cl_half, s::cl_int)

// log
cl_float log(s::cl_float x) __NOEXC { return std::log(x); }
cl_double log(s::cl_double x) __NOEXC { return std::log(x); }
cl_half log(s::cl_half x) __NOEXC { return std::log(x); }
MAKE_1V(log, s::cl_float, s::cl_float)
MAKE_1V(log, s::cl_double, s::cl_double)
MAKE_1V(log, s::cl_half, s::cl_half)

// log2
cl_float log2(s::cl_float x) __NOEXC { return std::log2(x); }
cl_double log2(s::cl_double x) __NOEXC { return std::log2(x); }
cl_half log2(s::cl_half x) __NOEXC { return std::log2(x); }
MAKE_1V(log2, s::cl_float, s::cl_float)
MAKE_1V(log2, s::cl_double, s::cl_double)
MAKE_1V(log2, s::cl_half, s::cl_half)

// log10
cl_float log10(s::cl_float x) __NOEXC { return std::log10(x); }
cl_double log10(s::cl_double x) __NOEXC { return std::log10(x); }
cl_half log10(s::cl_half x) __NOEXC { return std::log10(x); }
MAKE_1V(log10, s::cl_float, s::cl_float)
MAKE_1V(log10, s::cl_double, s::cl_double)
MAKE_1V(log10, s::cl_half, s::cl_half)

// log1p
cl_float log1p(s::cl_float x) __NOEXC { return std::log1p(x); }
cl_double log1p(s::cl_double x) __NOEXC { return std::log1p(x); }
cl_half log1p(s::cl_half x) __NOEXC { return std::log1p(x); }
MAKE_1V(log1p, s::cl_float, s::cl_float)
MAKE_1V(log1p, s::cl_double, s::cl_double)
MAKE_1V(log1p, s::cl_half, s::cl_half)

// logb
cl_float logb(s::cl_float x) __NOEXC { return std::logb(x); }
cl_double logb(s::cl_double x) __NOEXC { return std::logb(x); }
cl_half logb(s::cl_half x) __NOEXC { return std::logb(x); }
MAKE_1V(logb, s::cl_float, s::cl_float)
MAKE_1V(logb, s::cl_double, s::cl_double)
MAKE_1V(logb, s::cl_half, s::cl_half)

// mad
cl_float mad(s::cl_float a, s::cl_float b, s::cl_float c) __NOEXC {
  return __mad(a, b, c);
}
cl_double mad(s::cl_double a, s::cl_double b, s::cl_double c) __NOEXC {
  return __mad(a, b, c);
}
cl_half mad(s::cl_half a, s::cl_half b, s::cl_half c) __NOEXC {
  return __mad(a, b, c);
}
MAKE_1V_2V_3V(mad, s::cl_float, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V_3V(mad, s::cl_double, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V_3V(mad, s::cl_half, s::cl_half, s::cl_half, s::cl_half)

// maxmag
cl_float maxmag(s::cl_float x, s::cl_float y) __NOEXC { return __maxmag(x, y); }
cl_double maxmag(s::cl_double x, s::cl_double y) __NOEXC {
  return __maxmag(x, y);
}
cl_half maxmag(s::cl_half x, s::cl_half y) __NOEXC { return __maxmag(x, y); }
MAKE_1V_2V(maxmag, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(maxmag, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(maxmag, s::cl_half, s::cl_half, s::cl_half)

// minmag
cl_float minmag(s::cl_float x, s::cl_float y) __NOEXC { return __minmag(x, y); }
cl_double minmag(s::cl_double x, s::cl_double y) __NOEXC {
  return __minmag(x, y);
}
cl_half minmag(s::cl_half x, s::cl_half y) __NOEXC { return __minmag(x, y); }
MAKE_1V_2V(minmag, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(minmag, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(minmag, s::cl_half, s::cl_half, s::cl_half)

// modf
cl_float modf(s::cl_float x, s::cl_float *iptr) __NOEXC {
  return std::modf(x, iptr);
}
cl_double modf(s::cl_double x, s::cl_double *iptr) __NOEXC {
  return std::modf(x, iptr);
}
cl_half modf(s::cl_half x, s::cl_half *iptr) __NOEXC {
  return std::modf(x, reinterpret_cast<s::cl_float *>(iptr));
}
MAKE_1V_2P(modf, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2P(modf, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2P(modf, s::cl_half, s::cl_half, s::cl_half)

// nan
cl_float nan(s::cl_uint nancode) __NOEXC { return d::quiet_NaN<float>(); }
cl_double nan(s::cl_ulong nancode) __NOEXC { return d::quiet_NaN<double>(); }
cl_half nan(s::cl_ushort nancode) __NOEXC {
  return s::cl_half(d::quiet_NaN<float>());
}
MAKE_1V(nan, s::cl_float, s::cl_uint)
MAKE_1V(nan, s::cl_double, s::cl_ulong)
MAKE_1V(nan, s::cl_half, s::cl_ushort)

// pow
cl_float pow(s::cl_float x, s::cl_float y) __NOEXC { return std::pow(x, y); }
cl_double pow(s::cl_double x, s::cl_double y) __NOEXC { return std::pow(x, y); }
cl_half pow(s::cl_half x, s::cl_half y) __NOEXC { return std::pow(x, y); }
MAKE_1V_2V(pow, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(pow, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(pow, s::cl_half, s::cl_half, s::cl_half)

// pown
cl_float pown(s::cl_float x, s::cl_int y) __NOEXC { return std::pow(x, y); }
cl_double pown(s::cl_double x, s::cl_int y) __NOEXC { return std::pow(x, y); }
cl_half pown(s::cl_half x, s::cl_int y) __NOEXC { return std::pow(x, y); }
MAKE_1V_2V(pown, s::cl_float, s::cl_float, s::cl_int)
MAKE_1V_2V(pown, s::cl_double, s::cl_double, s::cl_int)
MAKE_1V_2V(pown, s::cl_half, s::cl_half, s::cl_int)

// powr
cl_float powr(s::cl_float x, s::cl_float y) __NOEXC { return __powr(x, y); }
cl_double powr(s::cl_double x, s::cl_double y) __NOEXC { return __powr(x, y); }
cl_half powr(s::cl_half x, s::cl_half y) __NOEXC { return __powr(x, y); }
MAKE_1V_2V(powr, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(powr, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(powr, s::cl_half, s::cl_half, s::cl_half)

// remainder
cl_float remainder(s::cl_float x, s::cl_float y) __NOEXC {
  return std::remainder(x, y);
}
cl_double remainder(s::cl_double x, s::cl_double y) __NOEXC {
  return std::remainder(x, y);
}
cl_half remainder(s::cl_half x, s::cl_half y) __NOEXC {
  return std::remainder(x, y);
}
MAKE_1V_2V(remainder, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(remainder, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(remainder, s::cl_half, s::cl_half, s::cl_half)

// remquo
cl_float remquo(s::cl_float x, s::cl_float y, s::cl_int *quo) __NOEXC {
  return std::remquo(x, y, quo);
}
cl_double remquo(s::cl_double x, s::cl_double y, s::cl_int *quo) __NOEXC {
  return std::remquo(x, y, quo);
}
cl_half remquo(s::cl_half x, s::cl_half y, s::cl_int *quo) __NOEXC {
  return std::remquo(x, y, quo);
}
MAKE_1V_2V_3P(remquo, s::cl_float, s::cl_float, s::cl_float, s::cl_int)
MAKE_1V_2V_3P(remquo, s::cl_double, s::cl_double, s::cl_double, s::cl_int)
MAKE_1V_2V_3P(remquo, s::cl_half, s::cl_half, s::cl_half, s::cl_int)

// rint
cl_float rint(s::cl_float x) __NOEXC { return std::rint(x); }
cl_double rint(s::cl_double x) __NOEXC { return std::rint(x); }
cl_half rint(s::cl_half x) __NOEXC { return std::rint(x); }
MAKE_1V(rint, s::cl_float, s::cl_float)
MAKE_1V(rint, s::cl_double, s::cl_double)
MAKE_1V(rint, s::cl_half, s::cl_half)

// rootn
cl_float rootn(s::cl_float x, s::cl_int y) __NOEXC { return __rootn(x, y); }
cl_double rootn(s::cl_double x, s::cl_int y) __NOEXC { return __rootn(x, y); }
cl_half rootn(s::cl_half x, s::cl_int y) __NOEXC { return __rootn(x, y); }
MAKE_1V_2V(rootn, s::cl_float, s::cl_float, s::cl_int)
MAKE_1V_2V(rootn, s::cl_double, s::cl_double, s::cl_int)
MAKE_1V_2V(rootn, s::cl_half, s::cl_half, s::cl_int)

// round
cl_float round(s::cl_float x) __NOEXC { return std::round(x); }
cl_double round(s::cl_double x) __NOEXC { return std::round(x); }
cl_half round(s::cl_half x) __NOEXC { return std::round(x); }
MAKE_1V(round, s::cl_float, s::cl_float)
MAKE_1V(round, s::cl_double, s::cl_double)
MAKE_1V(round, s::cl_half, s::cl_half)

// rsqrt
cl_float rsqrt(s::cl_float x) __NOEXC { return __rsqrt(x); }
cl_double rsqrt(s::cl_double x) __NOEXC { return __rsqrt(x); }
cl_half rsqrt(s::cl_half x) __NOEXC { return __rsqrt(x); }
MAKE_1V(rsqrt, s::cl_float, s::cl_float)
MAKE_1V(rsqrt, s::cl_double, s::cl_double)
MAKE_1V(rsqrt, s::cl_half, s::cl_half)

// sin
cl_float sin(s::cl_float x) __NOEXC { return std::sin(x); }
cl_double sin(s::cl_double x) __NOEXC { return std::sin(x); }
cl_half sin(s::cl_half x) __NOEXC { return std::sin(x); }
MAKE_1V(sin, s::cl_float, s::cl_float)
MAKE_1V(sin, s::cl_double, s::cl_double)
MAKE_1V(sin, s::cl_half, s::cl_half)

// sincos
cl_float sincos(s::cl_float x, s::cl_float *cosval) __NOEXC {
  return __sincos(x, cosval);
}
cl_double sincos(s::cl_double x, s::cl_double *cosval) __NOEXC {
  return __sincos(x, cosval);
}
cl_half sincos(s::cl_half x, s::cl_half *cosval) __NOEXC {
  return __sincos(x, cosval);
}
MAKE_1V_2P(sincos, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2P(sincos, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2P(sincos, s::cl_half, s::cl_half, s::cl_half)

// sinh
cl_float sinh(s::cl_float x) __NOEXC { return std::sinh(x); }
cl_double sinh(s::cl_double x) __NOEXC { return std::sinh(x); }
cl_half sinh(s::cl_half x) __NOEXC { return std::sinh(x); }
MAKE_1V(sinh, s::cl_float, s::cl_float)
MAKE_1V(sinh, s::cl_double, s::cl_double)
MAKE_1V(sinh, s::cl_half, s::cl_half)

// sinpi
cl_float sinpi(s::cl_float x) __NOEXC { return __sinpi(x); }
cl_double sinpi(s::cl_double x) __NOEXC { return __sinpi(x); }
cl_half sinpi(s::cl_half x) __NOEXC { return __sinpi(x); }
MAKE_1V(sinpi, s::cl_float, s::cl_float)
MAKE_1V(sinpi, s::cl_double, s::cl_double)
MAKE_1V(sinpi, s::cl_half, s::cl_half)

// sqrt
cl_float sqrt(s::cl_float x) __NOEXC { return std::sqrt(x); }
cl_double sqrt(s::cl_double x) __NOEXC { return std::sqrt(x); }
cl_half sqrt(s::cl_half x) __NOEXC { return std::sqrt(x); }
MAKE_1V(sqrt, s::cl_float, s::cl_float)
MAKE_1V(sqrt, s::cl_double, s::cl_double)
MAKE_1V(sqrt, s::cl_half, s::cl_half)

// tan
cl_float tan(s::cl_float x) __NOEXC { return std::tan(x); }
cl_double tan(s::cl_double x) __NOEXC { return std::tan(x); }
cl_half tan(s::cl_half x) __NOEXC { return std::tan(x); }
MAKE_1V(tan, s::cl_float, s::cl_float)
MAKE_1V(tan, s::cl_double, s::cl_double)
MAKE_1V(tan, s::cl_half, s::cl_half)

// tanh
cl_float tanh(s::cl_float x) __NOEXC { return std::tanh(x); }
cl_double tanh(s::cl_double x) __NOEXC { return std::tanh(x); }
cl_half tanh(s::cl_half x) __NOEXC { return std::tanh(x); }
MAKE_1V(tanh, s::cl_float, s::cl_float)
MAKE_1V(tanh, s::cl_double, s::cl_double)
MAKE_1V(tanh, s::cl_half, s::cl_half)

// tanpi
cl_float tanpi(s::cl_float x) __NOEXC { return __tanpi(x); }
cl_double tanpi(s::cl_double x) __NOEXC { return __tanpi(x); }
cl_half tanpi(s::cl_half x) __NOEXC { return __tanpi(x); }
MAKE_1V(tanpi, s::cl_float, s::cl_float)
MAKE_1V(tanpi, s::cl_double, s::cl_double)
MAKE_1V(tanpi, s::cl_half, s::cl_half)

// tgamma
cl_float tgamma(s::cl_float x) __NOEXC { return std::tgamma(x); }
cl_double tgamma(s::cl_double x) __NOEXC { return std::tgamma(x); }
cl_half tgamma(s::cl_half x) __NOEXC { return std::tgamma(x); }
MAKE_1V(tgamma, s::cl_float, s::cl_float)
MAKE_1V(tgamma, s::cl_double, s::cl_double)
MAKE_1V(tgamma, s::cl_half, s::cl_half)

// trunc
cl_float trunc(s::cl_float x) __NOEXC { return std::trunc(x); }
cl_double trunc(s::cl_double x) __NOEXC { return std::trunc(x); }
cl_half trunc(s::cl_half x) __NOEXC { return std::trunc(x); }
MAKE_1V(trunc, s::cl_float, s::cl_float)
MAKE_1V(trunc, s::cl_double, s::cl_double)
MAKE_1V(trunc, s::cl_half, s::cl_half)

// --------------- 4.13.3 Native Math functions. Host implementations. ---------
// native_cos
cl_float native_cos(s::cl_float x) __NOEXC { return std::cos(x); }
MAKE_1V(native_cos, s::cl_float, s::cl_float)

// native_divide
cl_float native_divide(s::cl_float x, s::cl_float y) __NOEXC { return x / y; }
MAKE_1V_2V(native_divide, s::cl_float, s::cl_float, s::cl_float)

// native_exp
cl_float native_exp(s::cl_float x) __NOEXC { return std::exp(x); }
MAKE_1V(native_exp, s::cl_float, s::cl_float)

// native_exp2
cl_float native_exp2(s::cl_float x) __NOEXC { return std::exp2(x); }
MAKE_1V(native_exp2, s::cl_float, s::cl_float)

// native_exp10
cl_float native_exp10(s::cl_float x) __NOEXC { return std::pow(10, x); }
MAKE_1V(native_exp10, s::cl_float, s::cl_float)

// native_log
cl_float native_log(s::cl_float x) __NOEXC { return std::log(x); }
MAKE_1V(native_log, s::cl_float, s::cl_float)

// native_log2
cl_float native_log2(s::cl_float x) __NOEXC { return std::log2(x); }
MAKE_1V(native_log2, s::cl_float, s::cl_float)

// native_log10
cl_float native_log10(s::cl_float x) __NOEXC { return std::log10(x); }
MAKE_1V(native_log10, s::cl_float, s::cl_float)

// native_powr
cl_float native_powr(s::cl_float x, s::cl_float y) __NOEXC {
  return (x >= 0 ? std::pow(x, y) : x);
}
MAKE_1V_2V(native_powr, s::cl_float, s::cl_float, s::cl_float)

// native_recip
cl_float native_recip(s::cl_float x) __NOEXC { return 1.0 / x; }
MAKE_1V(native_recip, s::cl_float, s::cl_float)

// native_rsqrt
cl_float native_rsqrt(s::cl_float x) __NOEXC { return 1.0 / std::sqrt(x); }
MAKE_1V(native_rsqrt, s::cl_float, s::cl_float)

// native_sin
cl_float native_sin(s::cl_float x) __NOEXC { return std::sin(x); }
MAKE_1V(native_sin, s::cl_float, s::cl_float)

// native_sqrt
cl_float native_sqrt(s::cl_float x) __NOEXC { return std::sqrt(x); }
MAKE_1V(native_sqrt, s::cl_float, s::cl_float)

// native_tan
cl_float native_tan(s::cl_float x) __NOEXC { return std::tan(x); }
MAKE_1V(native_tan, s::cl_float, s::cl_float)

// ---------- 4.13.3 Half Precision Math functions. Host implementations. ------
// half_cos
cl_float half_cos(s::cl_float x) __NOEXC { return std::cos(x); }
MAKE_1V(half_cos, s::cl_float, s::cl_float)

// half_divide
cl_float half_divide(s::cl_float x, s::cl_float y) __NOEXC { return x / y; }
MAKE_1V_2V(half_divide, s::cl_float, s::cl_float, s::cl_float)

// half_exp
cl_float half_exp(s::cl_float x) __NOEXC { return std::exp(x); }
MAKE_1V(half_exp, s::cl_float, s::cl_float)
// half_exp2
cl_float half_exp2(s::cl_float x) __NOEXC { return std::exp2(x); }
MAKE_1V(half_exp2, s::cl_float, s::cl_float)

// half_exp10
cl_float half_exp10(s::cl_float x) __NOEXC { return std::pow(10, x); }
MAKE_1V(half_exp10, s::cl_float, s::cl_float)
// half_log
cl_float half_log(s::cl_float x) __NOEXC { return std::log(x); }
MAKE_1V(half_log, s::cl_float, s::cl_float)

// half_log2
cl_float half_log2(s::cl_float x) __NOEXC { return std::log2(x); }
MAKE_1V(half_log2, s::cl_float, s::cl_float)

// half_log10
cl_float half_log10(s::cl_float x) __NOEXC { return std::log10(x); }
MAKE_1V(half_log10, s::cl_float, s::cl_float)

// half_powr
cl_float half_powr(s::cl_float x, s::cl_float y) __NOEXC {
  return (x >= 0 ? std::pow(x, y) : x);
}
MAKE_1V_2V(half_powr, s::cl_float, s::cl_float, s::cl_float)

// half_recip
cl_float half_recip(s::cl_float x) __NOEXC { return 1.0 / x; }
MAKE_1V(half_recip, s::cl_float, s::cl_float)

// half_rsqrt
cl_float half_rsqrt(s::cl_float x) __NOEXC { return 1.0 / std::sqrt(x); }
MAKE_1V(half_rsqrt, s::cl_float, s::cl_float)

// half_sin
cl_float half_sin(s::cl_float x) __NOEXC { return std::sin(x); }
MAKE_1V(half_sin, s::cl_float, s::cl_float)

// half_sqrt
cl_float half_sqrt(s::cl_float x) __NOEXC { return std::sqrt(x); }
MAKE_1V(half_sqrt, s::cl_float, s::cl_float)

// half_tan
cl_float half_tan(s::cl_float x) __NOEXC { return std::tan(x); }
MAKE_1V(half_tan, s::cl_float, s::cl_float)

} // namespace __host_std
} // namespace cl
