//==--- complex_wrapper.cpp - wrappers for C99 complex math functions ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "device.h"

#if IMPL_ENABLED
#include "device_complex.h"

DEVICE_EXTERN_C
float __attribute__((weak)) cimagf(float __complex__ z) {
  return __devicelib_cimagf(z);
}

DEVICE_EXTERN_C
float __attribute__((weak)) crealf(float __complex__ z) {
  return __devicelib_crealf(z);
}

DEVICE_EXTERN_C
float __attribute__((weak)) cargf(float __complex__ z) {
  return __devicelib_cargf(z);
}

DEVICE_EXTERN_C
float __attribute__((weak)) cabsf(float __complex__ z) {
  return __devicelib_cabsf(z);
}

DEVICE_EXTERN_C
float __complex__ __attribute__((weak)) cprojf(float __complex__ z) {
  return __devicelib_cprojf(z);
}

DEVICE_EXTERN_C
float __complex__ __attribute__((weak)) cexpf(float __complex__ z) {
  return __devicelib_cexpf(z);
}

DEVICE_EXTERN_C
float __complex__ __attribute__((weak)) clogf(float __complex__ z) {
  return __devicelib_clogf(z);
}

DEVICE_EXTERN_C
float __complex__ __attribute__((weak)) cpowf(float __complex__ x,
                                              float __complex__ y) {
  return __devicelib_cpowf(x, y);
}

DEVICE_EXTERN_C
float __complex__ __attribute__((weak)) cpolarf(float rho, float theta) {
  return __devicelib_cpolarf(rho, theta);
}

DEVICE_EXTERN_C
float __complex__ __attribute__((weak)) csqrtf(float __complex__ z) {
  return __devicelib_csqrtf(z);
}

DEVICE_EXTERN_C
float __complex__ __attribute__((weak)) csinhf(float __complex__ z) {
  return __devicelib_csinhf(z);
}

DEVICE_EXTERN_C
float __complex__ __attribute__((weak)) ccoshf(float __complex__ z) {
  return __devicelib_ccoshf(z);
}

DEVICE_EXTERN_C
float __complex__ __attribute__((weak)) ctanhf(float __complex__ z) {
  return __devicelib_ctanhf(z);
}

DEVICE_EXTERN_C
float __complex__ __attribute__((weak)) csinf(float __complex__ z) {
  return __devicelib_csinf(z);
}

DEVICE_EXTERN_C
float __complex__ __attribute__((weak)) ccosf(float __complex__ z) {
  return __devicelib_ccosf(z);
}

DEVICE_EXTERN_C
float __complex__ __attribute__((weak)) ctanf(float __complex__ z) {
  return __devicelib_ctanf(z);
}

DEVICE_EXTERN_C
float __complex__ __attribute__((weak)) cacosf(float __complex__ z) {
  return __devicelib_cacosf(z);
}

DEVICE_EXTERN_C
float __complex__ __attribute__((weak)) casinhf(float __complex__ z) {
  return __devicelib_casinhf(z);
}

DEVICE_EXTERN_C
float __complex__ __attribute__((weak)) casinf(float __complex__ z) {
  return __devicelib_casinf(z);
}

DEVICE_EXTERN_C
float __complex__ __attribute__((weak)) cacoshf(float __complex__ z) {
  return __devicelib_cacoshf(z);
}

DEVICE_EXTERN_C
float __complex__ __attribute__((weak)) catanhf(float __complex__ z) {
  return __devicelib_catanhf(z);
}

DEVICE_EXTERN_C
float __complex__ __attribute__((weak)) catanf(float __complex__ z) {
  return __devicelib_catanf(z);
}

// __mulsc3
// Returns: the product of a + ib and c + id
DEVICE_EXTERN_C
float __complex__ __attribute__((weak)) __mulsc3(float __a, float __b,
                                                 float __c, float __d) {
  return __devicelib___mulsc3(__a, __b, __c, __d);
}

// __divsc3
// Returns: the quotient of (a + ib) / (c + id)
DEVICE_EXTERN_C
float __complex__ __attribute__((weak)) __divsc3(float __a, float __b,
                                                 float __c, float __d) {
  return __devicelib___divsc3(__a, __b, __c, __d);
}
#endif  // IMPL_ENABLED
