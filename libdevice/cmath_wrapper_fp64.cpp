//==--- cmath_wrapper_fp64.cpp - wrappers for double precision C math library
// functions ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "device_math.h"

#if defined(__SPIR__) || defined(__SPIRV__)

// All exported functions in math and complex device libraries are weak
// reference. If users provide their own math or complex functions(with
// the prototype), functions in device libraries will be ignored and
// overrided by users' version.

DEVICE_EXTERN_C_INLINE
double fabs(double x) { return __devicelib_fabs(x); }

DEVICE_EXTERN_C_INLINE
double ceil(double x) { return __devicelib_ceil(x); }

DEVICE_EXTERN_C_INLINE
double copysign(double x, double y) { return __devicelib_copysign(x, y); }

DEVICE_EXTERN_C_INLINE
double scalbln(double x, long y) { return __devicelib_scalbln(x, y); }

DEVICE_EXTERN_C_INLINE
double cospi(double x) { return __devicelib_cospi(x); }

extern "C" SYCL_EXTERNAL double __devicelib_fmax(double, double);
DEVICE_EXTERN_C_INLINE
double fmax(double x, double y) { return __devicelib_fmax(x, y); }

extern "C" SYCL_EXTERNAL double __devicelib_fmin(double, double);
DEVICE_EXTERN_C_INLINE
double fmin(double x, double y) { return __devicelib_fmin(x, y); }

DEVICE_EXTERN_C_INLINE
double trunc(double x) { return __devicelib_trunc(x); }

DEVICE_EXTERN_C_INLINE
double sinpi(double x) { return __devicelib_sinpi(x); }

DEVICE_EXTERN_C_INLINE
double rsqrt(double x) { return __devicelib_rsqrt(x); }

DEVICE_EXTERN_C_INLINE
double exp10(double x) { return __devicelib_exp10(x); }

DEVICE_EXTERN_C_INLINE
double log(double x) { return __devicelib_log(x); }

DEVICE_EXTERN_C_INLINE
double round(double x) { return __devicelib_round(x); }

DEVICE_EXTERN_C_INLINE
double floor(double x) { return __devicelib_floor(x); }

DEVICE_EXTERN_C_INLINE
double exp(double x) { return __devicelib_exp(x); }

DEVICE_EXTERN_C_INLINE
double frexp(double x, int *exp) { return __devicelib_frexp(x, exp); }

DEVICE_EXTERN_C_INLINE
double ldexp(double x, int exp) { return __devicelib_ldexp(x, exp); }

DEVICE_EXTERN_C_INLINE
double log10(double x) { return __devicelib_log10(x); }

DEVICE_EXTERN_C_INLINE
double modf(double x, double *intpart) { return __devicelib_modf(x, intpart); }

DEVICE_EXTERN_C_INLINE
double exp2(double x) { return __devicelib_exp2(x); }

DEVICE_EXTERN_C_INLINE
double expm1(double x) { return __devicelib_expm1(x); }

DEVICE_EXTERN_C_INLINE
int ilogb(double x) { return __devicelib_ilogb(x); }

DEVICE_EXTERN_C_INLINE
double log1p(double x) { return __devicelib_log1p(x); }

DEVICE_EXTERN_C_INLINE
double log2(double x) { return __devicelib_log2(x); }

DEVICE_EXTERN_C_INLINE
double logb(double x) { return __devicelib_logb(x); }

DEVICE_EXTERN_C_INLINE
double sqrt(double x) { return __devicelib_sqrt(x); }

DEVICE_EXTERN_C_INLINE
double cbrt(double x) { return __devicelib_cbrt(x); }

DEVICE_EXTERN_C_INLINE
double hypot(double x, double y) { return __devicelib_hypot(x, y); }

DEVICE_EXTERN_C_INLINE
double erf(double x) { return __devicelib_erf(x); }

DEVICE_EXTERN_C_INLINE
double erfc(double x) { return __devicelib_erfc(x); }

DEVICE_EXTERN_C_INLINE
double tgamma(double x) { return __devicelib_tgamma(x); }

DEVICE_EXTERN_C_INLINE
double lgamma(double x) { return __devicelib_lgamma(x); }

DEVICE_EXTERN_C_INLINE
double fmod(double x, double y) { return __devicelib_fmod(x, y); }

DEVICE_EXTERN_C_INLINE
double remainder(double x, double y) { return __devicelib_remainder(x, y); }

DEVICE_EXTERN_C_INLINE
double remquo(double x, double y, int *q) {
  return __devicelib_remquo(x, y, q);
}

DEVICE_EXTERN_C_INLINE
double nextafter(double x, double y) { return __devicelib_nextafter(x, y); }

DEVICE_EXTERN_C_INLINE
double fdim(double x, double y) { return __devicelib_fdim(x, y); }

DEVICE_EXTERN_C_INLINE
double fma(double x, double y, double z) { return __devicelib_fma(x, y, z); }

DEVICE_EXTERN_C_INLINE
double sin(double x) { return __devicelib_sin(x); }

DEVICE_EXTERN_C_INLINE
double cos(double x) { return __devicelib_cos(x); }

DEVICE_EXTERN_C_INLINE
double tan(double x) { return __devicelib_tan(x); }

DEVICE_EXTERN_C_INLINE
double pow(double x, double y) { return __devicelib_pow(x, y); }

DEVICE_EXTERN_C_INLINE
double acos(double x) { return __devicelib_acos(x); }

DEVICE_EXTERN_C_INLINE
double asin(double x) { return __devicelib_asin(x); }

DEVICE_EXTERN_C_INLINE
double atan(double x) { return __devicelib_atan(x); }

DEVICE_EXTERN_C_INLINE
double atan2(double x, double y) { return __devicelib_atan2(x, y); }

DEVICE_EXTERN_C_INLINE
double cosh(double x) { return __devicelib_cosh(x); }

DEVICE_EXTERN_C_INLINE
double sinh(double x) { return __devicelib_sinh(x); }

DEVICE_EXTERN_C_INLINE
double tanh(double x) { return __devicelib_tanh(x); }

DEVICE_EXTERN_C_INLINE
double acosh(double x) { return __devicelib_acosh(x); }

DEVICE_EXTERN_C_INLINE
double asinh(double x) { return __devicelib_asinh(x); }

DEVICE_EXTERN_C_INLINE
double atanh(double x) { return __devicelib_atanh(x); }

DEVICE_EXTERN_C_INLINE
double scalbn(double x, int exp) { return __devicelib_scalbn(x, exp); }

DEVICE_EXTERN_C_INLINE
double rint(double x) { return __spirv_ocl_rint(x); }
#endif // __SPIR__ || __SPIRV__
