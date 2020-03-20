//==------- device_math.h - math devicelib functions declarations-----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==------------------------------------------------------------------------==//

#ifndef __LIBDEVICE_DEVICE_MATH_H__
#define __LIBDEVICE_DEVICE_MATH_H__

#include "device.h"

DEVICE_EXTERNAL double __spirv_ocl_log(double);
DEVICE_EXTERNAL double __spirv_ocl_sin(double);
DEVICE_EXTERNAL double __spirv_ocl_cos(double);
DEVICE_EXTERNAL double __spirv_ocl_sinh(double);
DEVICE_EXTERNAL double __spirv_ocl_cosh(double);
DEVICE_EXTERNAL double __spirv_ocl_tanh(double);
DEVICE_EXTERNAL double __spirv_ocl_exp(double);
DEVICE_EXTERNAL double __spirv_ocl_sqrt(double);
DEVICE_EXTERNAL bool __spirv_IsInf(double);
DEVICE_EXTERNAL bool __spirv_IsFinite(double);
DEVICE_EXTERNAL bool __spirv_IsNan(double);
DEVICE_EXTERNAL bool __spirv_IsNormal(double);
DEVICE_EXTERNAL bool __spirv_SignBitSet(double);
DEVICE_EXTERNAL double __spirv_ocl_hypot(double, double);
DEVICE_EXTERNAL double __spirv_ocl_atan2(double, double);
DEVICE_EXTERNAL double __spirv_ocl_pow(double, double);
DEVICE_EXTERNAL double __spirv_ocl_ldexp(double, int);
DEVICE_EXTERNAL double __spirv_ocl_copysign(double, double);
DEVICE_EXTERNAL double __spirv_ocl_fmax(double, double);
DEVICE_EXTERNAL double __spirv_ocl_fabs(double);
DEVICE_EXTERNAL double __spirv_ocl_tan(double);
DEVICE_EXTERNAL double __spirv_ocl_acos(double);
DEVICE_EXTERNAL double __spirv_ocl_asin(double);
DEVICE_EXTERNAL double __spirv_ocl_atan(double);
DEVICE_EXTERNAL double __spirv_ocl_atan2(double, double);
DEVICE_EXTERNAL double __spirv_ocl_cosh(double);
DEVICE_EXTERNAL double __spirv_ocl_sinh(double);
DEVICE_EXTERNAL double __spirv_ocl_tanh(double);
DEVICE_EXTERNAL double __spirv_ocl_acosh(double);
DEVICE_EXTERNAL double __spirv_ocl_asinh(double);
DEVICE_EXTERNAL double __spirv_ocl_atanh(double);
DEVICE_EXTERNAL double __spirv_ocl_frexp(double, int *);
DEVICE_EXTERNAL double __spirv_ocl_log10(double);
DEVICE_EXTERNAL double __spirv_ocl_modf(double, double *);
DEVICE_EXTERNAL double __spirv_ocl_exp2(double);
DEVICE_EXTERNAL double __spirv_ocl_expm1(double);
DEVICE_EXTERNAL int __spirv_ocl_ilogb(double);
DEVICE_EXTERNAL double __spriv_ocl_log1p(double);
DEVICE_EXTERNAL double __spirv_ocl_log2(double);
DEVICE_EXTERNAL double __spirv_ocl_logb(double);
DEVICE_EXTERNAL double __spirv_ocl_sqrt(double);
DEVICE_EXTERNAL double __spirv_ocl_cbrt(double);
DEVICE_EXTERNAL double __spirv_ocl_hypot(double);
DEVICE_EXTERNAL double __spirv_ocl_erf(double);
DEVICE_EXTERNAL double __spirv_ocl_erfc(double);
DEVICE_EXTERNAL double __spirv_ocl_tgamma(double);
DEVICE_EXTERNAL double __spirv_ocl_lgamma(double);
DEVICE_EXTERNAL double __spirv_ocl_fmod(double, double);
DEVICE_EXTERNAL double __spirv_ocl_remainder(double, double);
DEVICE_EXTERNAL double __spirv_ocl_remquo(double, double, int *);
DEVICE_EXTERNAL double __spirv_ocl_nextafter(double, double);
DEVICE_EXTERNAL double __spirv_ocl_fdim(double, double);
DEVICE_EXTERNAL double __spirv_ocl_fma(double, double, double);

DEVICE_EXTERNAL float __spirv_ocl_log(float);
DEVICE_EXTERNAL float __spirv_ocl_logb(float);
DEVICE_EXTERNAL float __spirv_ocl_sin(float);
DEVICE_EXTERNAL float __spirv_ocl_cos(float);
DEVICE_EXTERNAL float __spirv_ocl_sinh(float);
DEVICE_EXTERNAL float __spirv_ocl_cosh(float);
DEVICE_EXTERNAL float __spirv_ocl_tanh(float);
DEVICE_EXTERNAL float __spirv_ocl_exp(float);
DEVICE_EXTERNAL float __spirv_ocl_sqrt(float);
DEVICE_EXTERNAL bool __spirv_IsInf(float);
DEVICE_EXTERNAL bool __spirv_IsFinite(float);
DEVICE_EXTERNAL bool __spirv_IsNan(float);
DEVICE_EXTERNAL bool __spirv_IsNormal(double);
DEVICE_EXTERNAL bool __spirv_SignBitSet(float);
DEVICE_EXTERNAL float __spirv_ocl_hypot(float, float);
DEVICE_EXTERNAL float __spirv_ocl_atan2(float, float);
DEVICE_EXTERNAL float __spirv_ocl_pow(float, float);
DEVICE_EXTERNAL float __spirv_ocl_ldexp(float, int);
DEVICE_EXTERNAL float __spirv_ocl_copysign(float, float);
DEVICE_EXTERNAL float __spirv_ocl_fmax(float, float);
DEVICE_EXTERNAL float __spirv_ocl_fabs(float);
DEVICE_EXTERNAL float __spirv_ocl_tan(float);
DEVICE_EXTERNAL float __spirv_ocl_acos(float);
DEVICE_EXTERNAL float __spirv_ocl_asin(float);
DEVICE_EXTERNAL float __spirv_ocl_atan(float);
DEVICE_EXTERNAL float __spirv_ocl_atan2(float, float);
DEVICE_EXTERNAL float __spirv_ocl_cosh(float);
DEVICE_EXTERNAL float __spirv_ocl_sinh(float);
DEVICE_EXTERNAL float __spirv_ocl_tanh(float);
DEVICE_EXTERNAL float __spirv_ocl_acosh(float);
DEVICE_EXTERNAL float __spirv_ocl_asinh(float);
DEVICE_EXTERNAL float __spirv_ocl_atanh(float);
DEVICE_EXTERNAL float __spirv_ocl_frexp(float, int *);
DEVICE_EXTERNAL float __spirv_ocl_log10(float);
DEVICE_EXTERNAL float __spirv_ocl_modf(float, float *);
DEVICE_EXTERNAL float __spirv_ocl_exp2(float);
DEVICE_EXTERNAL float __spirv_ocl_expm1(float);
DEVICE_EXTERNAL int __spirv_ocl_ilogb(float);
DEVICE_EXTERNAL float __spirv_ocl_log1p(float);
DEVICE_EXTERNAL float __spirv_ocl_log2(float);
DEVICE_EXTERNAL float __spirv_ocl_sqrt(float);
DEVICE_EXTERNAL float __spirv_ocl_cbrt(float);
DEVICE_EXTERNAL float __spirv_ocl_hypot(float);
DEVICE_EXTERNAL float __spirv_ocl_erf(float);
DEVICE_EXTERNAL float __spirv_ocl_erfc(float);
DEVICE_EXTERNAL float __spirv_ocl_tgamma(float);
DEVICE_EXTERNAL float __spirv_ocl_lgamma(float);
DEVICE_EXTERNAL float __spirv_ocl_fmod(float, float);
DEVICE_EXTERNAL float __spirv_ocl_remainder(float, float);
DEVICE_EXTERNAL float __spirv_ocl_remquo(float, float, int *);
DEVICE_EXTERNAL float __spirv_ocl_nextafter(float, float);
DEVICE_EXTERNAL float __spirv_ocl_fdim(float, float);
DEVICE_EXTERNAL float __spirv_ocl_fma(float, float, float);

DEVICE_EXTERN_C
double __devicelib_log(double x);

DEVICE_EXTERN_C
float __devicelib_logf(float x);

DEVICE_EXTERN_C
double __devicelib_sin(double x);

DEVICE_EXTERN_C
float __devicelib_sinf(float x);

DEVICE_EXTERN_C
double __devicelib_cos(double x);

DEVICE_EXTERN_C
float __devicelib_cosf(float x);

DEVICE_EXTERN_C
double __devicelib_tan(double x);

DEVICE_EXTERN_C
float __devicelib_tanf(float x);

DEVICE_EXTERN_C
double __devicelib_acos(double x);

DEVICE_EXTERN_C
float __devicelib_acosf(float x);

DEVICE_EXTERN_C
double __devicelib_pow(double x, double y);

DEVICE_EXTERN_C
float __devicelib_powf(float x, float y);

DEVICE_EXTERN_C
double __devicelib_sqrt(double x);

DEVICE_EXTERN_C
float __devicelib_sqrtf(float x);

DEVICE_EXTERN_C
double __devicelib_cbrt(double x);

DEVICE_EXTERN_C
float __devicelib_cbrtf(float x);

DEVICE_EXTERN_C
double __devicelib_hypot(double x, double y);

DEVICE_EXTERN_C
float __devicelib_hypotf(float x, float y);

DEVICE_EXTERN_C
double __devicelib_erf(double x);

DEVICE_EXTERN_C
float __devicelib_erff(float x);

DEVICE_EXTERN_C
double __devicelib_erfc(double x);

DEVICE_EXTERN_C
float __devicelib_erfcf(float x);

DEVICE_EXTERN_C
double __devicelib_tgamma(double x);

DEVICE_EXTERN_C
float __devicelib_tgammaf(float x);

DEVICE_EXTERN_C
double __devicelib_lgamma(double x);

DEVICE_EXTERN_C
float __devicelib_lgammaf(float x);

DEVICE_EXTERN_C
double __devicelib_fmod(double x, double y);

DEVICE_EXTERN_C
float __devicelib_fmodf(float x, float y);

DEVICE_EXTERN_C
double __devicelib_remainder(double x, double y);

DEVICE_EXTERN_C
float __devicelib_remainderf(float x, float y);

DEVICE_EXTERN_C
double __devicelib_remquo(double x, double y, int *q);

DEVICE_EXTERN_C
float __devicelib_remquof(float x, float y, int *q);

DEVICE_EXTERN_C
double __devicelib_nextafter(double x, double y);

DEVICE_EXTERN_C
float __devicelib_nextafterf(float x, float y);

DEVICE_EXTERN_C
double __devicelib_fdim(double x, double y);

DEVICE_EXTERN_C
float __devicelib_fdimf(float x, float y);

DEVICE_EXTERN_C
double __devicelib_fma(double x, double y, double z);

DEVICE_EXTERN_C
float __devicelib_fmaf(float x, float y, float z);

DEVICE_EXTERN_C
float __devicelib_asinf(float x);

DEVICE_EXTERN_C
double __devicelib_asin(double x);

DEVICE_EXTERN_C
float __devicelib_atanf(float x);

DEVICE_EXTERN_C
double __devicelib_atan(double x);

DEVICE_EXTERN_C
float __devicelib_atan2f(float x, float y);

DEVICE_EXTERN_C
double __devicelib_atan2(double x, double y);

DEVICE_EXTERN_C
float __devicelib_coshf(float x);

DEVICE_EXTERN_C
double __devicelib_cosh(double x);

DEVICE_EXTERN_C
float __devicelib_sinhf(float x);

DEVICE_EXTERN_C
double __devicelib_sinh(double x);

DEVICE_EXTERN_C
float __devicelib_tanhf(float x);

DEVICE_EXTERN_C
double __devicelib_tanh(double x);

DEVICE_EXTERN_C
float __devicelib_acoshf(float x);

DEVICE_EXTERN_C
double __devicelib_acosh(double x);

DEVICE_EXTERN_C
float __devicelib_asinhf(float x);

DEVICE_EXTERN_C
double __devicelib_asinh(double x);

DEVICE_EXTERN_C
float __devicelib_atanhf(float x);

DEVICE_EXTERN_C
double __devicelib_atanh(double x);

DEVICE_EXTERN_C
double __devicelib_frexp(double x, int *exp);

DEVICE_EXTERN_C
float __devicelib_frexpf(float x, int *exp);

DEVICE_EXTERN_C
double __devicelib_ldexp(double x, int exp);

DEVICE_EXTERN_C
float __devicelib_ldexpf(float x, int exp);

DEVICE_EXTERN_C
double __devicelib_log10(double x);

DEVICE_EXTERN_C
float __devicelib_log10f(float x);

DEVICE_EXTERN_C
double __devicelib_modf(double x, double *intpart);

DEVICE_EXTERN_C
float __devicelib_modff(float x, float *intpart);

DEVICE_EXTERN_C
double __devicelib_exp(double x);

DEVICE_EXTERN_C
float __devicelib_expf(float x);

DEVICE_EXTERN_C
double __devicelib_exp2(double x);

DEVICE_EXTERN_C
float __devicelib_exp2f(float x);

DEVICE_EXTERN_C
double __devicelib_expm1(double x);

DEVICE_EXTERN_C
float __devicelib_expm1f(float x);

DEVICE_EXTERN_C
int __devicelib_ilogb(double x);

DEVICE_EXTERN_C
int __devicelib_ilogbf(float x);

DEVICE_EXTERN_C
double __devicelib_log1p(double x);

DEVICE_EXTERN_C
float __devicelib_log1pf(float x);

DEVICE_EXTERN_C
double __devicelib_log2(double x);

DEVICE_EXTERN_C
float __devicelib_log2f(float x);

DEVICE_EXTERN_C
double __devicelib_logb(double x);

DEVICE_EXTERN_C
float __devicelib_logbf(float x);

DEVICE_EXTERN_C
float __devicelib_scalbnf(float x, int n);
#endif // __LIBDEVICE_DEVICE_MATH_H__
