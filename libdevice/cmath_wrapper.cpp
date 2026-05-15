//==--- cmath_wrapper.cpp - wrappers for C math library functions ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "complex_wrapper.hpp"
#include "device.h"

#if defined(__SPIR__) || defined(__SPIRV__)
DEVICE_EXTERN_C_INLINE
float erfcf(float x) { return __spirv_ocl_erfc(x); }

DEVICE_EXTERN_C_INLINE
float expm1f(float x) { return __spirv_ocl_expm1(x); }

DEVICE_EXTERN_C_INLINE
float tgammaf(float x) { return __spirv_ocl_tgamma(x); }

DEVICE_EXTERN_C_INLINE
float lgammaf(float x) { return __spirv_ocl_lgamma(x); }

DEVICE_EXTERN_C_INLINE
double cospi(double x) { return __spirv_ocl_cospi(x); }

DEVICE_EXTERN_C_INLINE
double expm1(double x) { return __spirv_ocl_expm1(x); }

DEVICE_EXTERN_C_INLINE
double sinpi(double x) { return __spirv_ocl_sinpi(x); }

DEVICE_EXTERN_C_INLINE
double rsqrt(double x) { return __spirv_ocl_rsqrt(x); }

DEVICE_EXTERN_C_INLINE
double erf(double x) { return __spirv_ocl_erf(x); }

DEVICE_EXTERN_C_INLINE
double erfc(double x) { return __spirv_ocl_erfc(x); }

DEVICE_EXTERN_C_INLINE
double log1p(double x) { return __spirv_ocl_log1p(x); }

DEVICE_EXTERN_C_INLINE
double tgamma(double x) { return __spirv_ocl_tgamma(x); }

DEVICE_EXTERN_C_INLINE
double lgamma(double x) { return __spirv_ocl_lgamma(x); }

DEVICE_EXTERN_C_INLINE
double cosh(double x) { return __spirv_ocl_cosh(x); }

DEVICE_EXTERN_C_INLINE
double sinh(double x) { return __spirv_ocl_sinh(x); }

DEVICE_EXTERN_C_INLINE
double acosh(double x) { return __spirv_ocl_acosh(x); }

DEVICE_EXTERN_C_INLINE
double asinh(double x) { return __spirv_ocl_asinh(x); }

DEVICE_EXTERN_C_INLINE
double atanh(double x) { return __spirv_ocl_atanh(x); }

DEVICE_EXTERN_C_INLINE
double hypot(double x, double y) { return __spirv_ocl_hypot(x, y); }
#endif // __SPIR__ || __SPIRV__
