//==--- fallback-cmath.cpp - fallback implementation of math functions -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifdef __SYCL_DEVICE_ONLY__
#include "device_math.h"
extern "C" {
SYCL_EXTERNAL
double __devicelib_log(double x) {
  return __spirv_ocl_log(x);
}

SYCL_EXTERNAL
double __devicelib_exp(double x) {
  return __spirv_ocl_exp(x);
}

SYCL_EXTERNAL
double __devicelib_frexp(double x, int *exp) {
  return __spirv_ocl_frexp(x, exp);
}

SYCL_EXTERNAL
double __devicelib_ldexp(double x, int exp) {
  return __spirv_ocl_ldexp(x, exp);
}

SYCL_EXTERNAL
double __devicelib_log10(double x) {
  return __spirv_ocl_log10(x);
}

SYCL_EXTERNAL
double __devicelib_modf(double x, double *intpart) {
  return __spirv_ocl_modf(x, intpart);
}

SYCL_EXTERNAL
double __devicelib_exp2(double x) {
  return __spirv_ocl_exp2(x);
}

SYCL_EXTERNAL
double __devicelib_expm1(double x) {
  return __spirv_ocl_expm1(x);
}

SYCL_EXTERNAL
int __devicelib_ilogb(double x) {
  return __spirv_ocl_ilogb(x);
}

SYCL_EXTERNAL
double __devicelib_log1p(double x) {
  return __spirv_ocl_log1p(x);
}

SYCL_EXTERNAL
double __devicelib_log2(double x) {
  return __spirv_ocl_log2(x);
}

SYCL_EXTERNAL
double __devicelib_logb(double x) {
  return __spirv_ocl_logb(x);
}

SYCL_EXTERNAL
double __devicelib_sqrt(double x) {
  return __spirv_ocl_sqrt(x);
}

SYCL_EXTERNAL
double __devicelib_cbrt(double x) {
  return __spirv_ocl_cbrt(x);
}

SYCL_EXTERNAL
double __devicelib_hypot(double x, double y) {
  return __spirv_ocl_hypot(x, y);
}

SYCL_EXTERNAL
double __devicelib_erf(double x) {
  return __spirv_ocl_erf(x);
}

SYCL_EXTERNAL
double __devicelib_erfc(double x) {
  return __spirv_ocl_erfc(x);
}

SYCL_EXTERNAL
double __devicelib_tgamma(double x) {
  return __spirv_ocl_tgamma(x);
}

SYCL_EXTERNAL
double __devicelib_lgamma(double x) {
  return __spirv_ocl_lgamma(x);
}

SYCL_EXTERNAL
double __devicelib_fmod(double x, double y) {
  return __spirv_ocl_fmod(x, y);
}

SYCL_EXTERNAL
double __devicelib_remainder(double x, double y) {
  return __spirv_ocl_remainder(x, y);
}

SYCL_EXTERNAL
double __devicelib_remquo(double x, double y, int *q) {
  return __spirv_ocl_remquo(x, y, q);
}

SYCL_EXTERNAL
double __devicelib_nextafter(double x, double y) {
  return __spirv_ocl_nextafter(x, y);
}

SYCL_EXTERNAL
double __devicelib_fdim(double x, double y) {
  return __spirv_ocl_fdim(x, y);
}

SYCL_EXTERNAL
double __devicelib_fma(double x, double y, double z) {
  return __spirv_ocl_fma(x, y, z);
}

SYCL_EXTERNAL
double __devicelib_sin(double x) {
  return __spirv_ocl_sin(x);
}

SYCL_EXTERNAL
double __devicelib_cos(double x) {
  return __spirv_ocl_cos(x);
}

SYCL_EXTERNAL
double __devicelib_tan(double x) {
  return __spirv_ocl_tan(x);
}

SYCL_EXTERNAL
double __devicelib_pow(double x, double y) {
  return __spirv_ocl_pow(x, y);
}

SYCL_EXTERNAL
double __devicelib_acos(double x) {
  return __spirv_ocl_acos(x);
}

SYCL_EXTERNAL
double __devicelib_asin(double x) {
  return __spirv_ocl_asin(x);
}

SYCL_EXTERNAL
double __devicelib_atan(double x) {
  return __spirv_ocl_atan(x);
}

SYCL_EXTERNAL
double __devicelib_atan2(double x, double y) {
  return __spirv_ocl_atan2(x, y);
}

SYCL_EXTERNAL
double __devicelib_cosh(double x) {
  return __spirv_ocl_cosh(x);
}

SYCL_EXTERNAL
double __devicelib_sinh(double x) {
  return __spirv_ocl_sinh(x);
}

SYCL_EXTERNAL
double __devicelib_tanh(double x) {
  return __spirv_ocl_tanh(x);
}

SYCL_EXTERNAL
double __devicelib_acosh(double x) {
  return __spirv_ocl_acosh(x);
}

SYCL_EXTERNAL
double __devicelib_asinh(double x) {
  return __spirv_ocl_asinh(x);
}

SYCL_EXTERNAL
double __devicelib_atanh(double x) {
  return __spirv_ocl_atanh(x);
}
}
#endif
