//==------- device_math.h - math devicelib functions declarations-----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==------------------------------------------------------------------------==//

#ifndef __SYCL_CMATH_WRAPPER_H__
#define __SYCL_CMATH_WRAPPER_H__

__SYCL_HAS_DEFINITION__ double __spirv_ocl_log(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_sin(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_cos(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_sinh(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_cosh(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_tanh(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_exp(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_sqrt(double);
__SYCL_HAS_DEFINITION__ bool   __spirv_IsInf(double);
__SYCL_HAS_DEFINITION__ bool   __spirv_IsFinite(double);
__SYCL_HAS_DEFINITION__ bool   __spirv_IsNan(double);
__SYCL_HAS_DEFINITION__ bool   __spirv_IsNormal(double);
__SYCL_HAS_DEFINITION__ bool   __spirv_SignBitSet(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_hypot(double, double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_atan2(double, double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_pow(double, double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_ldexp(double, int);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_copysign(double, double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_fmax(double, double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_fabs(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_tan(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_acos(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_asin(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_atan(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_atan2(double, double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_cosh(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_sinh(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_tanh(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_acosh(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_asinh(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_atanh(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_frexp(double, int *);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_log10(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_modf(double, double *);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_exp2(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_expm1(double);
__SYCL_HAS_DEFINITION__ int    __spirv_ocl_ilogb(double);
__SYCL_HAS_DEFINITION__ double __spriv_ocl_log1p(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_log2(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_logb(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_sqrt(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_cbrt(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_hypot(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_erf(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_erfc(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_tgamma(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_lgamma(double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_fmod(double, double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_remainder(double, double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_remquo(double, double, int*);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_nextafter(double, double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_fdim(double, double);
__SYCL_HAS_DEFINITION__ double __spirv_ocl_fma(double, double, double);

__SYCL_HAS_DEFINITION__ float  __spirv_ocl_log(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_logb(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_sin(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_cos(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_sinh(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_cosh(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_tanh(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_exp(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_sqrt(float);
__SYCL_HAS_DEFINITION__ bool   __spirv_IsInf(float);
__SYCL_HAS_DEFINITION__ bool   __spirv_IsFinite(float);
__SYCL_HAS_DEFINITION__ bool   __spirv_IsNan(float);
__SYCL_HAS_DEFINITION__ bool   __spirv_IsNormal(double);
__SYCL_HAS_DEFINITION__ bool   __spirv_SignBitSet(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_hypot(float, float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_atan2(float, float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_pow(float, float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_ldexp(float, int);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_copysign(float, float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_fmax(float, float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_fabs(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_tan(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_acos(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_asin(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_atan(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_atan2(float, float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_cosh(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_sinh(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_tanh(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_acosh(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_asinh(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_atanh(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_frexp(float, int *);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_log10(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_modf(float, float *);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_exp2(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_expm1(float);
__SYCL_HAS_DEFINITION__ int    __spirv_ocl_ilogb(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_log1p(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_log2(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_sqrt(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_cbrt(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_hypot(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_erf(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_erfc(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_tgamma(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_lgamma(float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_fmod(float, float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_remainder(float, float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_remquo(float, float, int*);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_nextafter(float, float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_fdim(float, float);
__SYCL_HAS_DEFINITION__ float  __spirv_ocl_fma(float, float, float);

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
