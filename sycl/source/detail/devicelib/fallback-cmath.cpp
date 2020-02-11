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
float __devicelib_scalbnf(float x, int n) {
  return __spirv_ocl_ldexp(x, n);
}

SYCL_EXTERNAL
float __devicelib_logf(float x) {
  return __spirv_ocl_log(x);
}

SYCL_EXTERNAL
float __devicelib_expf(float x) {
  return __spirv_ocl_exp(x);
}

SYCL_EXTERNAL
float __devicelib_frexpf(float x, int *exp) {
  return __spirv_ocl_frexp(x, exp);
}

SYCL_EXTERNAL
float __devicelib_ldexpf(float x, int exp) {
  return __spirv_ocl_ldexp(x, exp);
}

SYCL_EXTERNAL
float __devicelib_log10f(float x) {
  return __spirv_ocl_log10(x);
}

SYCL_EXTERNAL
float __devicelib_modff(float x, float *intpart) {
  return __spirv_ocl_modf(x, intpart);
}

SYCL_EXTERNAL
float __devicelib_exp2f(float x) {
  return __spirv_ocl_exp2(x);
}

SYCL_EXTERNAL
float __devicelib_expm1f(float x) {
  return __spirv_ocl_expm1(x);
}

SYCL_EXTERNAL
int __devicelib_ilogbf(float x) {
  return __spirv_ocl_ilogb(x);
}

SYCL_EXTERNAL
float __devicelib_log1pf(float x) {
  return __spirv_ocl_log1p(x);
}

SYCL_EXTERNAL
float __devicelib_log2f(float x) {
  return __spirv_ocl_log2(x);
}

SYCL_EXTERNAL
float __devicelib_logbf(float x) {
  return __spirv_ocl_logb(x);
}

SYCL_EXTERNAL
float __devicelib_sqrtf(float x) {
  return __spirv_ocl_sqrt(x);
}

SYCL_EXTERNAL
float __devicelib_cbrtf(float x) {
  return __spirv_ocl_cbrt(x);
}

SYCL_EXTERNAL
float __devicelib_hypotf(float x, float y) {
  return __spirv_ocl_hypot(x, y);
}

SYCL_EXTERNAL
float __devicelib_erff(float x) {
  return __spirv_ocl_erf(x);
}

SYCL_EXTERNAL
float __devicelib_erfcf(float x) {
  return __spirv_ocl_erfc(x);
}

SYCL_EXTERNAL
float __devicelib_tgammaf(float x) {
  return __spirv_ocl_tgamma(x);
}

SYCL_EXTERNAL
float __devicelib_lgammaf(float x) {
  return __spirv_ocl_lgamma(x);
}

SYCL_EXTERNAL
float __devicelib_fmodf(float x, float y) {
  return __spirv_ocl_fmod(x, y);
}

SYCL_EXTERNAL
float __devicelib_remainderf(float x, float y) {
  return __spirv_ocl_remainder(x, y);
}

SYCL_EXTERNAL
float __devicelib_remquof(float x, float y, int *q) {
  return __spirv_ocl_remquo(x, y, q);
}

SYCL_EXTERNAL
float __devicelib_nextafterf(float x, float y) {
  return __spirv_ocl_nextafter(x, y);
}

SYCL_EXTERNAL
float __devicelib_fdimf(float x, float y) {
  return __spirv_ocl_fdim(x, y);
}

SYCL_EXTERNAL
float __devicelib_fmaf(float x, float y, float z) {
  return __spirv_ocl_fma(x, y, z);
}

SYCL_EXTERNAL
float __devicelib_sinf(float x) {
  return __spirv_ocl_sin(x);
}

SYCL_EXTERNAL
float __devicelib_cosf(float x) {
  return __spirv_ocl_cos(x);
}

SYCL_EXTERNAL
float __devicelib_tanf(float x) {
  return __spirv_ocl_tan(x);
}

SYCL_EXTERNAL
float __devicelib_powf(float x, float y) {
  return __spirv_ocl_pow(x, y);
}

SYCL_EXTERNAL
float __devicelib_acosf(float x) {
  return __spirv_ocl_acos(x);
}

SYCL_EXTERNAL
float __devicelib_asinf(float x) {
  return __spirv_ocl_asin(x);
}

SYCL_EXTERNAL
float __devicelib_atanf(float x) {
  return __spirv_ocl_atan(x);
}

SYCL_EXTERNAL
float __devicelib_atan2f(float x, float y) {
  return __spirv_ocl_atan2(x, y);
}

SYCL_EXTERNAL
float __devicelib_coshf(float x) {
  return  __spirv_ocl_cosh(x);
}

SYCL_EXTERNAL
float __devicelib_sinhf(float x) {
  return __spirv_ocl_sinh(x);
}

SYCL_EXTERNAL
float __devicelib_tanhf(float x) {
  return __spirv_ocl_tanh(x);
}

SYCL_EXTERNAL
float __devicelib_acoshf(float x) {
  return __spirv_ocl_acosh(x);
}

SYCL_EXTERNAL
float __devicelib_asinhf(float x) {
  return __spirv_ocl_asinh(x);
}

SYCL_EXTERNAL
float __devicelib_atanhf(float x) {
  return __spirv_ocl_atanh(x);
}
}
#endif
