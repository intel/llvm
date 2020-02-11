//==--- cmath_wrapper.cpp - wrappers for C math library functions ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifdef __SYCL_DEVICE_ONLY__
#include "device_math.h"
extern "C" {
// All exported functions in math and complex device libraries are weak
// reference. If users provide their own math or complex functions(with
// the prototype), functions in device libraries will be ignored and
// overrided by users' version.
SYCL_EXTERNAL
double __attribute__((weak)) log(double x) {
  return __devicelib_log(x);
}

SYCL_EXTERNAL
double __attribute__((weak)) exp(double x) {
  return __devicelib_exp(x);
}

SYCL_EXTERNAL
double __attribute__((weak)) frexp(double x, int *exp) {
  return __devicelib_frexp(x, exp);
}

SYCL_EXTERNAL
double __attribute__((weak)) ldexp(double x, int exp) {
  return __devicelib_ldexp(x, exp);
}

SYCL_EXTERNAL
double __attribute__((weak)) log10(double x) {
  return __devicelib_log10(x);
}

SYCL_EXTERNAL
double __attribute__((weak)) modf(double x, double *intpart) {
  return __devicelib_modf(x, intpart);
}

SYCL_EXTERNAL
double __attribute__((weak)) exp2(double x) {
  return __devicelib_exp2(x);
}

SYCL_EXTERNAL
double __attribute__((weak)) expm1(double x) {
  return __devicelib_expm1(x);
}

SYCL_EXTERNAL
int __attribute__((weak)) ilogb(double x) {
  return __devicelib_ilogb(x);
}

SYCL_EXTERNAL
double __attribute__((weak)) log1p(double x) {
  return __devicelib_log1p(x);
}

SYCL_EXTERNAL
double __attribute__((weak)) log2(double x) {
  return __devicelib_log2(x);
}

SYCL_EXTERNAL
double __attribute__((weak)) logb(double x) {
  return __devicelib_logb(x);
}

SYCL_EXTERNAL
double __attribute__((weak)) sqrt(double x) {
  return __devicelib_sqrt(x);
}

SYCL_EXTERNAL
double __attribute__((weak)) cbrt(double x) {
  return __devicelib_cbrt(x);
}

SYCL_EXTERNAL
double __attribute__((weak)) hypot(double x, double y) {
  return __devicelib_hypot(x, y);
}

SYCL_EXTERNAL
double __attribute__((weak)) erf(double x) {
  return __devicelib_erf(x);
}

SYCL_EXTERNAL
double __attribute__((weak)) erfc(double x) {
  return __devicelib_erfc(x);
}

SYCL_EXTERNAL
double __attribute__((weak)) tgamma(double x) {
  return __devicelib_tgamma(x);
}

SYCL_EXTERNAL
double __attribute__((weak)) lgamma(double x) {
  return __devicelib_lgamma(x);
}

SYCL_EXTERNAL
double __attribute__((weak)) fmod(double x, double y) {
  return __devicelib_fmod(x, y);
}

SYCL_EXTERNAL
double __attribute__((weak)) remainder(double x, double y) {
  return __devicelib_remainder(x, y);
}

SYCL_EXTERNAL
double __attribute__((weak)) remquo(double x, double y, int *q) {
  return __devicelib_remquo(x, y, q);
}

SYCL_EXTERNAL
double __attribute__((weak)) nextafter(double x, double y) {
  return __devicelib_nextafter(x, y);
}

SYCL_EXTERNAL
double __attribute__((weak)) fdim(double x, double y) {
  return __devicelib_fdim(x, y);
}

SYCL_EXTERNAL
double __attribute__((weak)) fma(double x, double y, double z) {
  return __devicelib_fma(x, y, z);
}

SYCL_EXTERNAL
double __attribute__((weak)) sin(double x) {
  return __devicelib_sin(x);
}

SYCL_EXTERNAL
double __attribute__((weak)) cos(double x) {
  return __devicelib_cos(x);
}

SYCL_EXTERNAL
double __attribute__((weak)) tan(double x) {
  return __devicelib_tan(x);
}

SYCL_EXTERNAL
double __attribute__((weak)) pow(double x, double y) {
  return __devicelib_pow(x, y);
}

SYCL_EXTERNAL
double __attribute__ ((weak)) acos(double x) {
  return __devicelib_acos(x);
}

SYCL_EXTERNAL
double __attribute__((weak)) asin(double x) {
  return __devicelib_asin(x);
}

SYCL_EXTERNAL
double __attribute__((weak)) atan(double x) {
  return __devicelib_atan(x);
}

SYCL_EXTERNAL
double __attribute__((weak)) atan2(double x, double y) {
  return __devicelib_atan2(x, y);
}

SYCL_EXTERNAL
double __attribute__((weak)) cosh(double x) {
  return __devicelib_cosh(x);
}

SYCL_EXTERNAL
double  __attribute__((weak)) sinh(double x) {
  return __devicelib_sinh(x);
}

SYCL_EXTERNAL
double __attribute__((weak)) tanh(double x) {
  return __devicelib_tanh(x);
}

SYCL_EXTERNAL
double __attribute__((weak)) acosh(double x) {
  return __devicelib_acosh(x);
}

SYCL_EXTERNAL
double __attribute__((weak)) asinh(double x) {
  return __devicelib_asinh(x);
}

SYCL_EXTERNAL
double __attribute__((weak)) atanh(double x) {
  return __devicelib_atanh(x);
}
}
#endif
