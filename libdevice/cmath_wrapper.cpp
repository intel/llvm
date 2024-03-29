//==--- cmath_wrapper.cpp - wrappers for C math library functions ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "device_math.h"

#if defined(__SPIR__) || defined(__NVPTX__)

DEVICE_EXTERN_C_INLINE
int abs(int x) { return __devicelib_abs(x); }

DEVICE_EXTERN_C_INLINE
long int labs(long int x) { return __devicelib_labs(x); }

DEVICE_EXTERN_C_INLINE
long long int llabs(long long int x) { return __devicelib_llabs(x); }

DEVICE_EXTERN_C_INLINE
float fabsf(float x) { return __devicelib_fabsf(x); }

DEVICE_EXTERN_C_INLINE
div_t div(int x, int y) { return __devicelib_div(x, y); }

DEVICE_EXTERN_C_INLINE
ldiv_t ldiv(long x, long y) { return __devicelib_ldiv(x, y); }

DEVICE_EXTERN_C_INLINE
lldiv_t lldiv(long long x, long long y) { return __devicelib_lldiv(x, y); }

DEVICE_EXTERN_C_INLINE
float roundf(float x) { return __devicelib_roundf(x); }

DEVICE_EXTERN_C_INLINE
float floorf(float x) { return __devicelib_floorf(x); }

DEVICE_EXTERN_C_INLINE
float scalbnf(float x, int n) { return __devicelib_scalbnf(x, n); }

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

#ifdef __NVPTX__
extern "C" SYCL_EXTERNAL float __nv_nearbyintf(float);
DEVICE_EXTERN_C_INLINE
float nearbyintf(float x) { return __nv_nearbyintf(x); }

extern "C" SYCL_EXTERNAL float __nv_rintf(float);
DEVICE_EXTERN_C_INLINE
float rintf(float x) { return __nv_rintf(x); }
#endif // __NVPTX__

#endif // __SPIR__ || __NVPTX__
