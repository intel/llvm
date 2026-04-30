//==--- cmath_wrapper.cpp - wrappers for C math library functions ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "complex_wrapper.hpp"
#include "device.h"
#include "fallback-cmath-fp64.hpp"
#include "fallback-cmath.hpp"

#if defined(__SPIR__) || defined(__SPIRV__)

DEVICE_EXTERN_C_INLINE
float fabsf(float x) { return __devicelib_fabsf(x); }

DEVICE_EXTERN_C_INLINE
float ceilf(float x) { return __devicelib_ceilf(x); }

DEVICE_EXTERN_C_INLINE
float copysignf(float x, float y) { return __devicelib_copysignf(x, y); }

DEVICE_EXTERN_C_INLINE
float cospif(float x) { return __devicelib_cospif(x); }

DEVICE_EXTERN_C_INLINE
float fmaxf(float x, float y) { return __devicelib_fmaxf(x, y); }

DEVICE_EXTERN_C_INLINE
float fminf(float x, float y) { return __devicelib_fminf(x, y); }

DEVICE_EXTERN_C_INLINE
float truncf(float x) { return __devicelib_truncf(x); }

DEVICE_EXTERN_C_INLINE
float sinpif(float x) { return __devicelib_sinpif(x); }

DEVICE_EXTERN_C_INLINE
float rsqrtf(float x) { return __devicelib_rsqrtf(x); }

DEVICE_EXTERN_C_INLINE
float exp10f(float x) { return __devicelib_exp10f(x); }

DEVICE_EXTERN_C_INLINE
float roundf(float x) { return __devicelib_roundf(x); }

DEVICE_EXTERN_C_INLINE
float floorf(float x) { return __devicelib_floorf(x); }

DEVICE_EXTERN_C_INLINE
float scalbnf(float x, int n) { return __devicelib_scalbnf(x, n); }

DEVICE_EXTERN_C_INLINE
float scalblnf(float x, long int n) { return __devicelib_scalblnf(x, n); }

DEVICE_EXTERN_C_INLINE
float logf(float x) { return __devicelib_logf(x); }

DEVICE_EXTERN_C_INLINE
float expf(float x) { return __devicelib_expf(x); }

DEVICE_EXTERN_C_INLINE
float frexpf(float x, int *exp) { return __devicelib_frexpf(x, exp); }

DEVICE_EXTERN_C_INLINE
float ldexpf(float x, int exp) { return __devicelib_ldexpf(x, exp); }

DEVICE_EXTERN_C_INLINE
float log10f(float x) { return __devicelib_log10f(x); }

DEVICE_EXTERN_C_INLINE
float modff(float x, float *intpart) { return __devicelib_modff(x, intpart); }

DEVICE_EXTERN_C_INLINE
float exp2f(float x) { return __devicelib_exp2f(x); }

DEVICE_EXTERN_C_INLINE
float expm1f(float x) { return __devicelib_expm1f(x); }

DEVICE_EXTERN_C_INLINE
int ilogbf(float x) { return __devicelib_ilogbf(x); }

DEVICE_EXTERN_C_INLINE
float log1pf(float x) { return __devicelib_log1pf(x); }

DEVICE_EXTERN_C_INLINE
float log2f(float x) { return __devicelib_log2f(x); }

DEVICE_EXTERN_C_INLINE
float logbf(float x) { return __devicelib_logbf(x); }

DEVICE_EXTERN_C_INLINE
float sqrtf(float x) { return __devicelib_sqrtf(x); }

DEVICE_EXTERN_C_INLINE
float cbrtf(float x) { return __devicelib_cbrtf(x); }

DEVICE_EXTERN_C_INLINE
float hypotf(float x, float y) { return __devicelib_hypotf(x, y); }

DEVICE_EXTERN_C_INLINE
float erff(float x) { return __devicelib_erff(x); }

DEVICE_EXTERN_C_INLINE
float erfcf(float x) { return __devicelib_erfcf(x); }

DEVICE_EXTERN_C_INLINE
float tgammaf(float x) { return __devicelib_tgammaf(x); }

DEVICE_EXTERN_C_INLINE
float lgammaf(float x) { return __devicelib_lgammaf(x); }

DEVICE_EXTERN_C_INLINE
float fmodf(float x, float y) { return __devicelib_fmodf(x, y); }

DEVICE_EXTERN_C_INLINE
float remainderf(float x, float y) { return __devicelib_remainderf(x, y); }

DEVICE_EXTERN_C_INLINE
float remquof(float x, float y, int *q) { return __devicelib_remquof(x, y, q); }

DEVICE_EXTERN_C_INLINE
float nextafterf(float x, float y) { return __devicelib_nextafterf(x, y); }

DEVICE_EXTERN_C_INLINE
float fdimf(float x, float y) { return __devicelib_fdimf(x, y); }

DEVICE_EXTERN_C_INLINE
float fmaf(float x, float y, float z) { return __devicelib_fmaf(x, y, z); }

DEVICE_EXTERN_C_INLINE
float sinf(float x) { return __devicelib_sinf(x); }

DEVICE_EXTERN_C_INLINE
float cosf(float x) { return __devicelib_cosf(x); }

DEVICE_EXTERN_C_INLINE
float tanf(float x) { return __devicelib_tanf(x); }

DEVICE_EXTERN_C_INLINE
float powf(float x, float y) { return __devicelib_powf(x, y); }

DEVICE_EXTERN_C_INLINE
float acosf(float x) { return __devicelib_acosf(x); }

DEVICE_EXTERN_C_INLINE
float asinf(float x) { return __devicelib_asinf(x); }

DEVICE_EXTERN_C_INLINE
float atanf(float x) { return __devicelib_atanf(x); }

DEVICE_EXTERN_C_INLINE
float atan2f(float x, float y) { return __devicelib_atan2f(x, y); }

DEVICE_EXTERN_C_INLINE
float coshf(float x) { return __devicelib_coshf(x); }

DEVICE_EXTERN_C_INLINE
float sinhf(float x) { return __devicelib_sinhf(x); }

DEVICE_EXTERN_C_INLINE
float tanhf(float x) { return __devicelib_tanhf(x); }

DEVICE_EXTERN_C_INLINE
float acoshf(float x) { return __devicelib_acoshf(x); }

DEVICE_EXTERN_C_INLINE
float asinhf(float x) { return __devicelib_asinhf(x); }

DEVICE_EXTERN_C_INLINE
float atanhf(float x) { return __devicelib_atanhf(x); }

DEVICE_EXTERN_C_INLINE
float rintf(float x) { return __spirv_ocl_rint(x); }

DEVICE_EXTERN_C_INLINE
double fabs(double x) { return __devicelib_fabs(x); }

DEVICE_EXTERN_C_INLINE
double ceil(double x) { return __devicelib_ceil(x); }

DEVICE_EXTERN_C_INLINE
double copysign(double x, double y) { return __devicelib_copysign(x, y); }

DEVICE_EXTERN_C_INLINE
double scalbln(double x, long y) { return __devicelib_scalbln(x, y); }

DEVICE_EXTERN_C_INLINE
double cospi(double x) { return __devicelib_cospi(x); }

DEVICE_EXTERN_C_INLINE
double fmax(double x, double y) { return __devicelib_fmax(x, y); }

DEVICE_EXTERN_C_INLINE
double fmin(double x, double y) { return __devicelib_fmin(x, y); }

DEVICE_EXTERN_C_INLINE
double trunc(double x) { return __devicelib_trunc(x); }

DEVICE_EXTERN_C_INLINE
double sinpi(double x) { return __devicelib_sinpi(x); }

DEVICE_EXTERN_C_INLINE
double rsqrt(double x) { return __devicelib_rsqrt(x); }

DEVICE_EXTERN_C_INLINE
double exp10(double x) { return __devicelib_exp10(x); }

DEVICE_EXTERN_C_INLINE
double log(double x) { return __devicelib_log(x); }

DEVICE_EXTERN_C_INLINE
double round(double x) { return __devicelib_round(x); }

DEVICE_EXTERN_C_INLINE
double floor(double x) { return __devicelib_floor(x); }

DEVICE_EXTERN_C_INLINE
double exp(double x) { return __devicelib_exp(x); }

DEVICE_EXTERN_C_INLINE
double frexp(double x, int *exp) { return __devicelib_frexp(x, exp); }

DEVICE_EXTERN_C_INLINE
double ldexp(double x, int exp) { return __devicelib_ldexp(x, exp); }

DEVICE_EXTERN_C_INLINE
double log10(double x) { return __devicelib_log10(x); }

DEVICE_EXTERN_C_INLINE
double modf(double x, double *intpart) { return __devicelib_modf(x, intpart); }

DEVICE_EXTERN_C_INLINE
double exp2(double x) { return __devicelib_exp2(x); }

DEVICE_EXTERN_C_INLINE
double expm1(double x) { return __devicelib_expm1(x); }

DEVICE_EXTERN_C_INLINE
int ilogb(double x) { return __devicelib_ilogb(x); }

DEVICE_EXTERN_C_INLINE
double log1p(double x) { return __devicelib_log1p(x); }

DEVICE_EXTERN_C_INLINE
double log2(double x) { return __devicelib_log2(x); }

DEVICE_EXTERN_C_INLINE
double logb(double x) { return __devicelib_logb(x); }

DEVICE_EXTERN_C_INLINE
double sqrt(double x) { return __devicelib_sqrt(x); }

DEVICE_EXTERN_C_INLINE
double cbrt(double x) { return __devicelib_cbrt(x); }

DEVICE_EXTERN_C_INLINE
double hypot(double x, double y) { return __devicelib_hypot(x, y); }

DEVICE_EXTERN_C_INLINE
double erf(double x) { return __devicelib_erf(x); }

DEVICE_EXTERN_C_INLINE
double erfc(double x) { return __devicelib_erfc(x); }

DEVICE_EXTERN_C_INLINE
double tgamma(double x) { return __devicelib_tgamma(x); }

DEVICE_EXTERN_C_INLINE
double lgamma(double x) { return __devicelib_lgamma(x); }

DEVICE_EXTERN_C_INLINE
double fmod(double x, double y) { return __devicelib_fmod(x, y); }

DEVICE_EXTERN_C_INLINE
double remainder(double x, double y) { return __devicelib_remainder(x, y); }

DEVICE_EXTERN_C_INLINE
double remquo(double x, double y, int *q) {
  return __devicelib_remquo(x, y, q);
}

DEVICE_EXTERN_C_INLINE
double nextafter(double x, double y) { return __devicelib_nextafter(x, y); }

DEVICE_EXTERN_C_INLINE
double fdim(double x, double y) { return __devicelib_fdim(x, y); }

DEVICE_EXTERN_C_INLINE
double fma(double x, double y, double z) { return __devicelib_fma(x, y, z); }

DEVICE_EXTERN_C_INLINE
double sin(double x) { return __devicelib_sin(x); }

DEVICE_EXTERN_C_INLINE
double cos(double x) { return __devicelib_cos(x); }

DEVICE_EXTERN_C_INLINE
double tan(double x) { return __devicelib_tan(x); }

DEVICE_EXTERN_C_INLINE
double pow(double x, double y) { return __devicelib_pow(x, y); }

DEVICE_EXTERN_C_INLINE
double acos(double x) { return __devicelib_acos(x); }

DEVICE_EXTERN_C_INLINE
double asin(double x) { return __devicelib_asin(x); }

DEVICE_EXTERN_C_INLINE
double atan(double x) { return __devicelib_atan(x); }

DEVICE_EXTERN_C_INLINE
double atan2(double x, double y) { return __devicelib_atan2(x, y); }

DEVICE_EXTERN_C_INLINE
double cosh(double x) { return __devicelib_cosh(x); }

DEVICE_EXTERN_C_INLINE
double sinh(double x) { return __devicelib_sinh(x); }

DEVICE_EXTERN_C_INLINE
double tanh(double x) { return __devicelib_tanh(x); }

DEVICE_EXTERN_C_INLINE
double acosh(double x) { return __devicelib_acosh(x); }

DEVICE_EXTERN_C_INLINE
double asinh(double x) { return __devicelib_asinh(x); }

DEVICE_EXTERN_C_INLINE
double atanh(double x) { return __devicelib_atanh(x); }

DEVICE_EXTERN_C_INLINE
double scalbn(double x, int exp) { return __devicelib_scalbn(x, exp); }

DEVICE_EXTERN_C_INLINE
double rint(double x) { return __spirv_ocl_rint(x); }
#endif // __SPIR__ || __SPIRV__
