//==--- complex_wrapper.cpp - wrappers for C99 complex math functions ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifdef __SYCL_DEVICE_ONLY__
#include "device_complex.h"
extern "C" {
SYCL_EXTERNAL
float __attribute__((weak)) cimagf(float __complex__ z) {
  return __devicelib_cimagf(z);
}

SYCL_EXTERNAL
float __attribute__((weak)) crealf(float __complex__ z) {
  return __devicelib_crealf(z);
}

SYCL_EXTERNAL
float __attribute__((weak)) cargf(float __complex__ z) {
  return __devicelib_cargf(z);
}

SYCL_EXTERNAL
float __attribute__((weak)) cabsf(float __complex__ z) {
  return __devicelib_cabsf(z);
}

SYCL_EXTERNAL
float __complex__ __attribute__((weak)) cprojf(float __complex__ z) {
  return __devicelib_cprojf(z);
}

SYCL_EXTERNAL
float __complex__ __attribute__((weak)) cexpf(float __complex__ z) {
  return __devicelib_cexpf(z);
}

SYCL_EXTERNAL
float __complex__ __attribute__((weak)) clogf(float __complex__ z) {
  return __devicelib_clogf(z);
}

SYCL_EXTERNAL
float __complex__ __attribute__((weak)) cpowf(float __complex__ x,
                                              float __complex__ y) {
  return __devicelib_cpowf(x, y);
}

SYCL_EXTERNAL
float __complex__ __attribute__((weak)) cpolarf(float rho, float theta) {
  return __devicelib_cpolarf(rho, theta);
}

SYCL_EXTERNAL
float __complex__ __attribute__((weak)) csqrtf(float __complex__ z) {
  return __devicelib_csqrtf(z);
}

SYCL_EXTERNAL
float __complex__ __attribute__((weak)) csinhf(float __complex__ z) {
  return __devicelib_csinhf(z);
}

SYCL_EXTERNAL
float __complex__ __attribute__((weak)) ccoshf(float __complex__ z) {
  return __devicelib_ccoshf(z);
}

SYCL_EXTERNAL
float __complex__ __attribute__((weak)) ctanhf(float __complex__ z) {
  return __devicelib_ctanhf(z);
}

SYCL_EXTERNAL
float __complex__ __attribute__((weak)) csinf(float __complex__ z) {
  return __devicelib_csinf(z);
}

SYCL_EXTERNAL
float __complex__ __attribute__((weak)) ccosf(float __complex__ z) {
  return __devicelib_ccosf(z);
}

SYCL_EXTERNAL
float __complex__ __attribute__((weak)) ctanf(float __complex__ z) {
  return __devicelib_ctanf(z);
}

SYCL_EXTERNAL
float __complex__ __attribute__((weak)) cacosf(float __complex__ z) {
  return __devicelib_cacosf(z);
}

SYCL_EXTERNAL
float __complex__ __attribute__((weak)) casinhf(float __complex__ z) {
  return __devicelib_casinhf(z);
}

SYCL_EXTERNAL
float __complex__ __attribute__((weak)) casinf(float __complex__ z) {
  return __devicelib_casinf(z);
}

SYCL_EXTERNAL
float __complex__ __attribute__((weak)) cacoshf(float __complex__ z) {
  return __devicelib_cacoshf(z);
}

SYCL_EXTERNAL
float __complex__ __attribute__((weak)) catanhf(float __complex__ z) {
  return __devicelib_catanhf(z);
}

SYCL_EXTERNAL
float __complex__ __attribute__((weak)) catanf(float __complex__ z) {
  return __devicelib_catanf(z);
}

// __mulsc3
// Returns: the product of a + ib and c + id
SYCL_EXTERNAL
float __complex__ __attribute__((weak)) __mulsc3(float __a, float __b,
                                                 float __c, float __d) {
  return __devicelib___mulsc3(__a, __b, __c, __d);
}

// __divsc3
// Returns: the quotient of (a + ib) / (c + id)
SYCL_EXTERNAL
float __complex__ __attribute__((weak)) __divsc3(float __a, float __b,
                                                 float __c, float __d) {
  return __devicelib___divsc3(__a, __b, __c, __d);
}
}
#endif
