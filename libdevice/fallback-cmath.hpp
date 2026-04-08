//==--- fallback-cmath.cpp - fallback implementation of math functions -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__SPIR__) || defined(__SPIRV__)

static inline float __devicelib_fabsf(float x) { return x < 0 ? -x : x; }

static inline float __devicelib_ceilf(float x) { return __spirv_ocl_ceil(x); }

static inline float __devicelib_copysignf(float x, float y) {
  return __spirv_ocl_copysign(x, y);
}

static inline float __devicelib_cospif(float x) { return __spirv_ocl_cospi(x); }

static inline float __devicelib_scalblnf(float x, long int y) {
  return __spirv_ocl_ldexp(x, (int)y);
}

static inline float __devicelib_fmaxf(float x, float y) {
  return __spirv_ocl_fmax(x, y);
}

static inline float __devicelib_fminf(float x, float y) {
  return __spirv_ocl_fmin(x, y);
}

static inline float __devicelib_truncf(float x) { return __spirv_ocl_trunc(x); }

static inline float __devicelib_sinpif(float x) { return __spirv_ocl_sinpi(x); }

static inline float __devicelib_rsqrtf(float x) { return __spirv_ocl_rsqrt(x); }

static inline float __devicelib_exp10f(float x) { return __spirv_ocl_exp10(x); }

static inline float __devicelib_scalbnf(float x, int n) {
  return __spirv_ocl_ldexp(x, n);
}

static inline float __devicelib_roundf(float x) { return __spirv_ocl_round(x); }

static inline float __devicelib_floorf(float x) { return __spirv_ocl_floor(x); }

static inline float __devicelib_logf(float x) { return __spirv_ocl_log(x); }

static inline float __devicelib_expf(float x) { return __spirv_ocl_exp(x); }

static inline float __devicelib_frexpf(float x, int *exp) {
  return __spirv_ocl_frexp(x, exp);
}

static inline float __devicelib_ldexpf(float x, int exp) {
  return __spirv_ocl_ldexp(x, exp);
}

static inline float __devicelib_log10f(float x) { return __spirv_ocl_log10(x); }

static inline float __devicelib_modff(float x, float *intpart) {
  return __spirv_ocl_modf(x, intpart);
}

static inline float __devicelib_exp2f(float x) { return __spirv_ocl_exp2(x); }

static inline float __devicelib_expm1f(float x) { return __spirv_ocl_expm1(x); }

static inline int __devicelib_ilogbf(float x) { return __spirv_ocl_ilogb(x); }

static inline float __devicelib_log1pf(float x) { return __spirv_ocl_log1p(x); }

static inline float __devicelib_log2f(float x) { return __spirv_ocl_log2(x); }

static inline float __devicelib_logbf(float x) { return __spirv_ocl_logb(x); }

static inline float __devicelib_sqrtf(float x) { return __spirv_ocl_sqrt(x); }

static inline float __devicelib_cbrtf(float x) { return __spirv_ocl_cbrt(x); }

static inline float __devicelib_hypotf(float x, float y) {
  return __spirv_ocl_hypot(x, y);
}

static inline float __devicelib_erff(float x) { return __spirv_ocl_erf(x); }

static inline float __devicelib_erfcf(float x) { return __spirv_ocl_erfc(x); }

static inline float __devicelib_tgammaf(float x) {
  return __spirv_ocl_tgamma(x);
}

static inline float __devicelib_lgammaf(float x) {
  return __spirv_ocl_lgamma(x);
}

static inline float __devicelib_fmodf(float x, float y) {
  return __spirv_ocl_fmod(x, y);
}

static inline float __devicelib_remainderf(float x, float y) {
  return __spirv_ocl_remainder(x, y);
}

static inline float __devicelib_remquof(float x, float y, int *q) {
  return __spirv_ocl_remquo(x, y, q);
}

static inline float __devicelib_nextafterf(float x, float y) {
  return __spirv_ocl_nextafter(x, y);
}

static inline float __devicelib_fdimf(float x, float y) {
  return __spirv_ocl_fdim(x, y);
}

static inline float __devicelib_fmaf(float x, float y, float z) {
  return __spirv_ocl_fma(x, y, z);
}

#if defined(__SPIR__) || defined(__SPIRV__)
static inline float __devicelib_sinf(float x) {
  return (x == 0.0f) ? x : __spirv_ocl_sin(x);
}
#else
static inline float __devicelib_sinf(float x) { return __spirv_ocl_sin(x); }
#endif

static inline float __devicelib_cosf(float x) { return __spirv_ocl_cos(x); }

static inline float __devicelib_tanf(float x) { return __spirv_ocl_tan(x); }

static inline float __devicelib_powf(float x, float y) {
  return __spirv_ocl_pow(x, y);
}

static inline float __devicelib_acosf(float x) { return __spirv_ocl_acos(x); }

static inline float __devicelib_asinf(float x) { return __spirv_ocl_asin(x); }

static inline float __devicelib_atanf(float x) { return __spirv_ocl_atan(x); }

static inline float __devicelib_atan2f(float x, float y) {
  return __spirv_ocl_atan2(x, y);
}

static inline float __devicelib_coshf(float x) { return __spirv_ocl_cosh(x); }

static inline float __devicelib_sinhf(float x) { return __spirv_ocl_sinh(x); }

static inline float __devicelib_tanhf(float x) { return __spirv_ocl_tanh(x); }

static inline float __devicelib_acoshf(float x) { return __spirv_ocl_acosh(x); }

static inline float __devicelib_asinhf(float x) { return __spirv_ocl_asinh(x); }

static inline float __devicelib_atanhf(float x) { return __spirv_ocl_atanh(x); }

#endif // __SPIR__ || __SPIRV__
