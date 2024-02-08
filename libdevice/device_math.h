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
#if defined(__SPIR__) || defined(__NVPTX__)
#include <cstdint>

typedef struct {
  int32_t quot;
  int32_t rem;
} __devicelib_div_t_32;

typedef struct {
  int64_t quot;
  int64_t rem;
} __devicelib_div_t_64;

typedef __devicelib_div_t_32 div_t;
#ifdef _WIN32
typedef __devicelib_div_t_32 ldiv_t;
#else
typedef __devicelib_div_t_64 ldiv_t;
#endif
typedef __devicelib_div_t_64 lldiv_t;

DEVICE_EXTERN_C
int __devicelib_abs(int x);

DEVICE_EXTERN_C
long int __devicelib_labs(long int x);

DEVICE_EXTERN_C
long long int __devicelib_llabs(long long int x);

DEVICE_EXTERN_C
float __devicelib_fabsf(float x);

DEVICE_EXTERN_C
double __devicelib_fabs(double x);

DEVICE_EXTERN_C
div_t __devicelib_div(int x, int y);

DEVICE_EXTERN_C
ldiv_t __devicelib_ldiv(long int x, long int y);

DEVICE_EXTERN_C
lldiv_t __devicelib_lldiv(long long int x, long long int y);

DEVICE_EXTERN_C
double __devicelib_round(double x);

DEVICE_EXTERN_C
float __devicelib_roundf(float x);

DEVICE_EXTERN_C
double __devicelib_floor(double x);

DEVICE_EXTERN_C
float __devicelib_floorf(float x);

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

DEVICE_EXTERN_C
double __devicelib_scalbn(double x, int exp);

#endif // __SPIR__ || __NVPTX__
#endif // __LIBDEVICE_DEVICE_MATH_H__
