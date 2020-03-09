//==--- cmath_wrapper.cpp - wrappers for C math library functions ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "device.h"
#include "device_math.h"

// All exported functions in math and complex device libraries are weak
// reference. If users provide their own math or complex functions(with
// the prototype), functions in device libraries will be ignored and
// overrided by users' version.
DEVICE_EXTERN_C
double log(double x) { return __devicelib_log(x); }

DEVICE_EXTERN_C
double exp(double x) { return __devicelib_exp(x); }

DEVICE_EXTERN_C
double frexp(double x, int *exp) { return __devicelib_frexp(x, exp); }

DEVICE_EXTERN_C
double ldexp(double x, int exp) { return __devicelib_ldexp(x, exp); }

DEVICE_EXTERN_C
double log10(double x) { return __devicelib_log10(x); }

DEVICE_EXTERN_C
double modf(double x, double *intpart) { return __devicelib_modf(x, intpart); }

DEVICE_EXTERN_C
double exp2(double x) { return __devicelib_exp2(x); }

DEVICE_EXTERN_C
double expm1(double x) { return __devicelib_expm1(x); }

DEVICE_EXTERN_C
int ilogb(double x) { return __devicelib_ilogb(x); }

DEVICE_EXTERN_C
double log1p(double x) { return __devicelib_log1p(x); }

DEVICE_EXTERN_C
double log2(double x) { return __devicelib_log2(x); }

DEVICE_EXTERN_C
double logb(double x) { return __devicelib_logb(x); }

DEVICE_EXTERN_C
double sqrt(double x) { return __devicelib_sqrt(x); }

DEVICE_EXTERN_C
double cbrt(double x) { return __devicelib_cbrt(x); }

DEVICE_EXTERN_C
double hypot(double x, double y) { return __devicelib_hypot(x, y); }

DEVICE_EXTERN_C
double erf(double x) { return __devicelib_erf(x); }

DEVICE_EXTERN_C
double erfc(double x) { return __devicelib_erfc(x); }

DEVICE_EXTERN_C
double tgamma(double x) { return __devicelib_tgamma(x); }

DEVICE_EXTERN_C
double lgamma(double x) { return __devicelib_lgamma(x); }

DEVICE_EXTERN_C
double fmod(double x, double y) { return __devicelib_fmod(x, y); }

DEVICE_EXTERN_C
double remainder(double x, double y) { return __devicelib_remainder(x, y); }

DEVICE_EXTERN_C
double remquo(double x, double y, int *q) {
  return __devicelib_remquo(x, y, q);
}

DEVICE_EXTERN_C
double nextafter(double x, double y) { return __devicelib_nextafter(x, y); }

DEVICE_EXTERN_C
double fdim(double x, double y) { return __devicelib_fdim(x, y); }

DEVICE_EXTERN_C
double fma(double x, double y, double z) { return __devicelib_fma(x, y, z); }

DEVICE_EXTERN_C
double sin(double x) { return __devicelib_sin(x); }

DEVICE_EXTERN_C
double cos(double x) { return __devicelib_cos(x); }

DEVICE_EXTERN_C
double tan(double x) { return __devicelib_tan(x); }

DEVICE_EXTERN_C
double pow(double x, double y) { return __devicelib_pow(x, y); }

DEVICE_EXTERN_C
double acos(double x) { return __devicelib_acos(x); }

DEVICE_EXTERN_C
double asin(double x) { return __devicelib_asin(x); }

DEVICE_EXTERN_C
double atan(double x) { return __devicelib_atan(x); }

DEVICE_EXTERN_C
double atan2(double x, double y) { return __devicelib_atan2(x, y); }

DEVICE_EXTERN_C
double cosh(double x) { return __devicelib_cosh(x); }

DEVICE_EXTERN_C
double sinh(double x) { return __devicelib_sinh(x); }

DEVICE_EXTERN_C
double tanh(double x) { return __devicelib_tanh(x); }

DEVICE_EXTERN_C
double acosh(double x) { return __devicelib_acosh(x); }

DEVICE_EXTERN_C
double asinh(double x) { return __devicelib_asinh(x); }

DEVICE_EXTERN_C
double atanh(double x) { return __devicelib_atanh(x); }
