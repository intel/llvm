//==------- device_math.h - math devicelib functions declarations-----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==------------------------------------------------------------------------==//

#ifndef __SYCL_CMATH_WRAPPER_H__
#define __SYCL_CMATH_WRAPPER_H__

double __spirv_ocl_log(double);
double __spirv_ocl_sin(double);
double __spirv_ocl_cos(double);
double __spirv_ocl_sinh(double);
double __spirv_ocl_cosh(double);
double __spirv_ocl_tanh(double);
double __spirv_ocl_exp(double);
double __spirv_ocl_sqrt(double);
bool   __spirv_IsInf(double);
bool   __spirv_IsFinite(double);
bool   __spirv_IsNan(double);
bool   __spirv_IsNormal(double);
bool   __spirv_SignBitSet(double);
double __spirv_ocl_hypot(double, double);
double __spirv_ocl_atan2(double, double);
double __spirv_ocl_pow(double, double);
double __spirv_ocl_ldexp(double, int);
double __spirv_ocl_copysign(double, double);
double __spirv_ocl_fmax(double, double);
double __spirv_ocl_fabs(double);
double __spirv_ocl_tan(double);
double __spirv_ocl_acos(double);
double __spirv_ocl_asin(double);
double __spirv_ocl_atan(double);
double __spirv_ocl_atan2(double, double);
double __spirv_ocl_cosh(double);
double __spirv_ocl_sinh(double);
double __spirv_ocl_tanh(double);
double __spirv_ocl_acosh(double);
double __spirv_ocl_asinh(double);
double __spirv_ocl_atanh(double);
double __spirv_ocl_frexp(double, int *);
double __spirv_ocl_log10(double);
double __spirv_ocl_modf(double, double *);
double __spirv_ocl_exp2(double);
double __spirv_ocl_expm1(double);
int    __spirv_ocl_ilogb(double);
double __spriv_ocl_log1p(double);
double __spirv_ocl_log2(double);
double __spirv_ocl_logb(double);
double __spirv_ocl_sqrt(double);
double __spirv_ocl_cbrt(double);
double __spirv_ocl_hypot(double);
double __spirv_ocl_erf(double);
double __spirv_ocl_erfc(double);
double __spirv_ocl_tgamma(double);
double __spirv_ocl_lgamma(double);
double __spirv_ocl_fmod(double, double);
double __spirv_ocl_remainder(double, double);
double __spirv_ocl_remquo(double, double, int*);
double __spirv_ocl_nextafter(double, double);
double __spirv_ocl_fdim(double, double);
double __spirv_ocl_fma(double, double, double);

float  __spirv_ocl_log(float);
float  __spirv_ocl_logb(float);
float  __spirv_ocl_sin(float);
float  __spirv_ocl_cos(float);
float  __spirv_ocl_sinh(float);
float  __spirv_ocl_cosh(float);
float  __spirv_ocl_tanh(float);
float  __spirv_ocl_exp(float);
float  __spirv_ocl_sqrt(float);
bool   __spirv_IsInf(float);
bool   __spirv_IsFinite(float);
bool   __spirv_IsNan(float);
bool   __spirv_IsNormal(double);
bool   __spirv_SignBitSet(float);
float  __spirv_ocl_hypot(float, float);
float  __spirv_ocl_atan2(float, float);
float  __spirv_ocl_pow(float, float);
float  __spirv_ocl_ldexp(float, int);
float  __spirv_ocl_copysign(float, float);
float  __spirv_ocl_fmax(float, float);
float  __spirv_ocl_fabs(float);
float  __spirv_ocl_tan(float);
float  __spirv_ocl_acos(float);
float  __spirv_ocl_asin(float);
float  __spirv_ocl_atan(float);
float  __spirv_ocl_atan2(float, float);
float  __spirv_ocl_cosh(float);
float  __spirv_ocl_sinh(float);
float  __spirv_ocl_tanh(float);
float  __spirv_ocl_acosh(float);
float  __spirv_ocl_asinh(float);
float  __spirv_ocl_atanh(float);
float  __spirv_ocl_frexp(float, int *);
float  __spirv_ocl_log10(float);
float  __spirv_ocl_modf(float, float *);
float  __spirv_ocl_exp2(float);
float  __spirv_ocl_expm1(float);
int    __spirv_ocl_ilogb(float);
float  __spirv_ocl_log1p(float);
float  __spirv_ocl_log2(float);
float  __spirv_ocl_sqrt(float);
float  __spirv_ocl_cbrt(float);
float  __spirv_ocl_hypot(float);
float  __spirv_ocl_erf(float);
float  __spirv_ocl_erfc(float);
float  __spirv_ocl_tgamma(float);
float  __spirv_ocl_lgamma(float);
float  __spirv_ocl_fmod(float, float);
float  __spirv_ocl_remainder(float, float);
float  __spirv_ocl_remquo(float, float, int*);
float  __spirv_ocl_nextafter(float, float);
float  __spirv_ocl_fdim(float, float);
float  __spirv_ocl_fma(float, float, float);

SYCL_EXTERNAL
extern "C" double __devicelib_log(double x);

SYCL_EXTERNAL
extern "C" float  __devicelib_logf(float x);

SYCL_EXTERNAL
extern "C" double __devicelib_sin(double x);

SYCL_EXTERNAL
extern "C" float  __devicelib_sinf(float x);

SYCL_EXTERNAL
extern "C" double __devicelib_cos(double x);

SYCL_EXTERNAL
extern "C" float  __devicelib_cosf(float x);

SYCL_EXTERNAL
extern "C" double __devicelib_tan(double x);

SYCL_EXTERNAL
extern "C" float  __devicelib_tanf(float x);

SYCL_EXTERNAL
extern "C" double __devicelib_acos(double x);

SYCL_EXTERNAL
extern "C" float  __devicelib_acosf(float x);

SYCL_EXTERNAL
extern "C" double __devicelib_pow(double x, double y);

SYCL_EXTERNAL
extern "C" float  __devicelib_powf(float x, float y);

SYCL_EXTERNAL
extern "C" double __devicelib_sqrt(double x);

SYCL_EXTERNAL
extern "C" float  __devicelib_sqrtf(float x);

SYCL_EXTERNAL
extern "C" double __devicelib_cbrt(double x);

SYCL_EXTERNAL
extern "C" float  __devicelib_cbrtf(float x);

SYCL_EXTERNAL
extern "C" double __devicelib_hypot(double x, double y);

SYCL_EXTERNAL
extern "C" float  __devicelib_hypotf(float x, float y);

SYCL_EXTERNAL
extern "C" double __devicelib_erf(double x);

SYCL_EXTERNAL
extern "C" float  __devicelib_erff(float x);

SYCL_EXTERNAL
extern "C" double __devicelib_erfc(double x);

SYCL_EXTERNAL
extern "C" float  __devicelib_erfcf(float x);

SYCL_EXTERNAL
extern "C" double __devicelib_tgamma(double x);

SYCL_EXTERNAL
extern "C" float  __devicelib_tgammaf(float x);

SYCL_EXTERNAL
extern "C" double __devicelib_lgamma(double x);

SYCL_EXTERNAL
extern "C" float  __devicelib_lgammaf(float x);

SYCL_EXTERNAL
extern "C" double __devicelib_fmod(double x, double y);

SYCL_EXTERNAL
extern "C" float  __devicelib_fmodf(float x, float y);

SYCL_EXTERNAL
extern "C" double __devicelib_remainder(double x, double y);

SYCL_EXTERNAL
extern "C" float  __devicelib_remainderf(float x, float y);

SYCL_EXTERNAL
extern "C" double __devicelib_remquo(double x, double y, int *q);

SYCL_EXTERNAL
extern "C" float  __devicelib_remquof(float x, float y, int *q);

SYCL_EXTERNAL
extern "C" double __devicelib_nextafter(double x, double y);

SYCL_EXTERNAL
extern "C" float  __devicelib_nextafterf(float x, float y);

SYCL_EXTERNAL
extern "C" double __devicelib_fdim(double x, double y);

SYCL_EXTERNAL
extern "C" float  __devicelib_fdimf(float x, float y);

SYCL_EXTERNAL
extern "C" double __devicelib_fma(double x, double y, double z);

SYCL_EXTERNAL
extern "C" float  __devicelib_fmaf(float x, float y, float z);

SYCL_EXTERNAL
extern "C" float  __devicelib_asinf(float x);

SYCL_EXTERNAL
extern "C" double __devicelib_asin(double x);

SYCL_EXTERNAL
extern "C" float  __devicelib_atanf(float x);

SYCL_EXTERNAL
extern "C" double __devicelib_atan(double x);

SYCL_EXTERNAL
extern "C" float  __devicelib_atan2f(float x, float y);

SYCL_EXTERNAL
extern "C" double __devicelib_atan2(double x, double y);

SYCL_EXTERNAL
extern "C" float  __devicelib_coshf(float x);

SYCL_EXTERNAL
extern "C" double __devicelib_cosh(double x);

SYCL_EXTERNAL
extern "C" float  __devicelib_sinhf(float x);

SYCL_EXTERNAL
extern "C" double __devicelib_sinh(double x);

SYCL_EXTERNAL
extern "C" float  __devicelib_tanhf(float x);

SYCL_EXTERNAL
extern "C" double __devicelib_tanh(double x);

SYCL_EXTERNAL
extern "C" float  __devicelib_acoshf(float x);

SYCL_EXTERNAL
extern "C" double __devicelib_acosh(double x);

SYCL_EXTERNAL
extern "C" float  __devicelib_asinhf(float x);

SYCL_EXTERNAL
extern "C" double __devicelib_asinh(double x);

SYCL_EXTERNAL
extern "C" float  __devicelib_atanhf(float x);

SYCL_EXTERNAL
extern "C" double __devicelib_atanh(double x);

SYCL_EXTERNAL
extern "C" double __devicelib_frexp(double x, int *exp);

SYCL_EXTERNAL
extern "C" float  __devicelib_frexpf(float x, int *exp);

SYCL_EXTERNAL
extern "C" double __devicelib_ldexp(double x, int exp);

SYCL_EXTERNAL
extern "C" float  __devicelib_ldexpf(float x, int exp);

SYCL_EXTERNAL
extern "C" double __devicelib_log10(double x);

SYCL_EXTERNAL
extern "C" float  __devicelib_log10f(float x);

SYCL_EXTERNAL
extern "C" double __devicelib_modf(double x, double *intpart);

SYCL_EXTERNAL
extern "C" float  __devicelib_modff(float x, float *intpart);

SYCL_EXTERNAL
extern "C" double __devicelib_exp(double x);

SYCL_EXTERNAL
extern "C" float  __devicelib_expf(float x);

SYCL_EXTERNAL
extern "C" double __devicelib_exp2(double x);

SYCL_EXTERNAL
extern "C" float  __devicelib_exp2f(float x);

SYCL_EXTERNAL
extern "C" double __devicelib_expm1(double x);

SYCL_EXTERNAL
extern "C" float  __devicelib_expm1f(float x);

SYCL_EXTERNAL
extern "C" int    __devicelib_ilogb(double x);

SYCL_EXTERNAL
extern "C" int    __devicelib_ilogbf(float x);

SYCL_EXTERNAL
extern "C" double __devicelib_log1p(double x);

SYCL_EXTERNAL
extern "C" float  __devicelib_log1pf(float x);

SYCL_EXTERNAL
extern "C" double __devicelib_log2(double x);

SYCL_EXTERNAL
extern "C" float  __devicelib_log2f(float x);

SYCL_EXTERNAL
extern "C" double __devicelib_logb(double x);

SYCL_EXTERNAL
extern "C" float  __devicelib_logbf(float x);

SYCL_EXTERNAL
extern "C" float  __devicelib_scalbnf(float x, int n);
#endif
