//==--- fallback-cmath-fp64.cpp - fallback implementation of double precision
// math functions -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "device.h"
#ifdef __LIBDEVICE_TARGET_SUPPORT

static inline double __devicelib_fabs(double x) { return x < 0 ? -x : x; }

static inline double __devicelib_ceil(double x) { return __spirv_ocl_ceil(x); }

static inline double __devicelib_copysign(double x, double y) {
  return __spirv_ocl_copysign(x, y);
}

static inline double __devicelib_cospi(double x) {
  return __spirv_ocl_cospi(x);
}

static inline double __devicelib_scalbln(double x, long int y) {
  return __spirv_ocl_ldexp(x, (int)y);
}

static inline double __devicelib_fmax(double x, double y) {
  return __spirv_ocl_fmax(x, y);
}

static inline double __devicelib_fmin(double x, double y) {
  return __spirv_ocl_fmin(x, y);
}

static inline double __devicelib_trunc(double x) {
  return __spirv_ocl_trunc(x);
}

static inline double __devicelib_sinpi(double x) {
  return __spirv_ocl_sinpi(x);
}

static inline double __devicelib_rsqrt(double x) {
  return __spirv_ocl_rsqrt(x);
}

static inline double __devicelib_exp10(double x) {
  return __spirv_ocl_exp10(x);
}

static inline double __devicelib_log(double x) { return __spirv_ocl_log(x); }

static inline double __devicelib_exp(double x) { return __spirv_ocl_exp(x); }

static inline double __devicelib_frexp(double x, int *exp) {
  return __spirv_ocl_frexp(x, exp);
}

static inline double __devicelib_ldexp(double x, int exp) {
  return __spirv_ocl_ldexp(x, exp);
}

static inline double __devicelib_log10(double x) {
  return __spirv_ocl_log10(x);
}

static inline double __devicelib_modf(double x, double *intpart) {
  return __spirv_ocl_modf(x, intpart);
}

static inline double __devicelib_round(double x) {
  return __spirv_ocl_round(x);
}

static inline double __devicelib_floor(double x) {
  return __spirv_ocl_floor(x);
}

static inline double __devicelib_exp2(double x) { return __spirv_ocl_exp2(x); }

static inline double __devicelib_expm1(double x) {
  return __spirv_ocl_expm1(x);
}

static inline int __devicelib_ilogb(double x) { return __spirv_ocl_ilogb(x); }

static inline double __devicelib_log1p(double x) {
  return __spirv_ocl_log1p(x);
}

static inline double __devicelib_log2(double x) { return __spirv_ocl_log2(x); }

static inline double __devicelib_logb(double x) { return __spirv_ocl_logb(x); }

static inline double __devicelib_sqrt(double x) { return __spirv_ocl_sqrt(x); }

static inline double __devicelib_cbrt(double x) { return __spirv_ocl_cbrt(x); }

static inline double __devicelib_hypot(double x, double y) {
  return __spirv_ocl_hypot(x, y);
}

static inline double __devicelib_erf(double x) { return __spirv_ocl_erf(x); }

static inline double __devicelib_erfc(double x) { return __spirv_ocl_erfc(x); }

static inline double __devicelib_tgamma(double x) {
  return __spirv_ocl_tgamma(x);
}

static inline double __devicelib_lgamma(double x) {
  return __spirv_ocl_lgamma(x);
}

static inline double __devicelib_fmod(double x, double y) {
  return __spirv_ocl_fmod(x, y);
}

static inline double __devicelib_remainder(double x, double y) {
  return __spirv_ocl_remainder(x, y);
}

static inline double __devicelib_remquo(double x, double y, int *q) {
  return __spirv_ocl_remquo(x, y, q);
}

static inline double __devicelib_nextafter(double x, double y) {
  return __spirv_ocl_nextafter(x, y);
}

static inline double __devicelib_fdim(double x, double y) {
  return __spirv_ocl_fdim(x, y);
}

static inline double __devicelib_fma(double x, double y, double z) {
  return __spirv_ocl_fma(x, y, z);
}

static inline double __devicelib_sin(double x) { return __spirv_ocl_sin(x); }

static inline double __devicelib_cos(double x) { return __spirv_ocl_cos(x); }

static inline double __devicelib_tan(double x) { return __spirv_ocl_tan(x); }

static inline double __devicelib_pow(double x, double y) {
  return __spirv_ocl_pow(x, y);
}

static inline double __devicelib_acos(double x) { return __spirv_ocl_acos(x); }

static inline double __devicelib_asin(double x) { return __spirv_ocl_asin(x); }

static inline double __devicelib_atan(double x) { return __spirv_ocl_atan(x); }

static inline double __devicelib_atan2(double x, double y) {
  return __spirv_ocl_atan2(x, y);
}

static inline double __devicelib_cosh(double x) { return __spirv_ocl_cosh(x); }

static inline double __devicelib_sinh(double x) { return __spirv_ocl_sinh(x); }

static inline double __devicelib_tanh(double x) { return __spirv_ocl_tanh(x); }

static inline double __devicelib_acosh(double x) {
  return __spirv_ocl_acosh(x);
}

static inline double __devicelib_asinh(double x) {
  return __spirv_ocl_asinh(x);
}

static inline double __devicelib_atanh(double x) {
  return __spirv_ocl_atanh(x);
}

static inline double __devicelib_scalbn(double x, int exp) {
  return __spirv_ocl_ldexp(x, exp);
}
#endif // __LIBDEVICE_TARGET_SUPPORT
