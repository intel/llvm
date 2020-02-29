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
double __attribute__((weak)) cimag(double __complex__ z) {
  return __devicelib_cimag(z);
}

DEVICE_EXTERN_C
double __attribute__((weak)) creal(double __complex__ z) {
  return __devicelib_creal(z);
}

DEVICE_EXTERN_C
double __attribute__((weak)) cabs(double __complex__ z) {
  return __devicelib_cabs(z);
}

DEVICE_EXTERN_C
double __attribute__((weak)) carg(double __complex__ z) {
  return __devicelib_carg(z);
}

DEVICE_EXTERN_C
double __complex__ __attribute__((weak)) cproj(double __complex__ z) {
  return __devicelib_cproj(z);
}

DEVICE_EXTERN_C
double __complex__ __attribute__((weak)) cexp(double __complex__ z) {
  return __devicelib_cexp(z);
}

DEVICE_EXTERN_C
double __complex__ __attribute__((weak)) clog(double __complex__ z) {
  return __devicelib_clog(z);
}

DEVICE_EXTERN_C
double __complex__ __attribute__((weak)) cpow(double __complex__ x,
                                              double __complex__ y) {
  return __devicelib_cpow(x, y);
}

DEVICE_EXTERN_C
double __complex__ __attribute__((weak)) cpolar(double rho, double theta) {
  return __devicelib_cpolar(rho, theta);
}

DEVICE_EXTERN_C
double __complex__ __attribute__((weak)) csqrt(double __complex__ z) {
  return __devicelib_csqrt(z);
}

DEVICE_EXTERN_C
double __complex__ __attribute__((weak)) csinh(double __complex__ z) {
  return __devicelib_csinh(z);
}

DEVICE_EXTERN_C
double __complex__ __attribute__((weak)) ccosh(double __complex__ z) {
  return __devicelib_ccosh(z);
}

DEVICE_EXTERN_C
double __complex__ __attribute__((weak)) ctanh(double __complex__ z) {
  return __devicelib_ctanh(z);
}

DEVICE_EXTERN_C
double __complex__ __attribute__((weak)) csin(double __complex__ z) {
  return __devicelib_csin(z);
}

DEVICE_EXTERN_C
double __complex__ __attribute__((weak)) ccos(double __complex__ z) {
  return __devicelib_ccos(z);
}

DEVICE_EXTERN_C
double __complex__ __attribute__((weak)) ctan(double __complex__ z) {
  return __devicelib_ctan(z);
}

DEVICE_EXTERN_C
double __complex__ __attribute__((weak)) cacos(double __complex__ z) {
  return __devicelib_cacos(z);
}

DEVICE_EXTERN_C
double __complex__ __attribute__((weak)) casinh(double __complex__ z) {
  return __devicelib_casinh(z);
}

DEVICE_EXTERN_C
double __complex__ __attribute__((weak)) casin(double __complex__ z) {
  return __devicelib_casin(z);
}

DEVICE_EXTERN_C
double __complex__ __attribute__((weak)) cacosh(double __complex__ z) {
  return __devicelib_cacosh(z);
}

DEVICE_EXTERN_C
double __complex__ __attribute__((weak)) catanh(double __complex__ z) {
  return __devicelib_catanh(z);
}

DEVICE_EXTERN_C
double __complex__ __attribute__((weak)) catan(double __complex__ z) {
  return __devicelib_catan(z);
}

// __muldc3
// Returns: the product of a + ib and c + id
DEVICE_EXTERN_C
double __complex__ __attribute__((weak)) __muldc3(double __a, double __b,
                                                  double __c, double __d) {
  return __devicelib___muldc3(__a, __b, __c, __d);
}

// __divdc3
// Returns: the quotient of (a + ib) / (c + id)
DEVICE_EXTERN_C
double __complex__ __attribute__((weak)) __divdc3(double __a, double __b,
                                                  double __c, double __d) {
  return __devicelib___divdc3(__a, __b, __c, __d);
}
#endif  // IMPL_ENABLED
