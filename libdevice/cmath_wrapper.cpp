//==--- cmath_wrapper.cpp - wrappers for C math library functions ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "device.h"
#include "device_math.h"

DEVICE_EXTERN_C
float scalbnf(float x, int n) { return __devicelib_scalbnf(x, n); }

DEVICE_EXTERN_C
float logf(float x) { return __devicelib_logf(x); }

DEVICE_EXTERN_C
float expf(float x) { return __devicelib_expf(x); }

DEVICE_EXTERN_C
float frexpf(float x, int *exp) { return __devicelib_frexpf(x, exp); }

DEVICE_EXTERN_C
float ldexpf(float x, int exp) { return __devicelib_ldexpf(x, exp); }

DEVICE_EXTERN_C
float log10f(float x) { return __devicelib_log10f(x); }

DEVICE_EXTERN_C
float modff(float x, float *intpart) { return __devicelib_modff(x, intpart); }

DEVICE_EXTERN_C
float exp2f(float x) { return __devicelib_exp2f(x); }

DEVICE_EXTERN_C
float expm1f(float x) { return __devicelib_expm1f(x); }

DEVICE_EXTERN_C
int ilogbf(float x) { return __devicelib_ilogbf(x); }

DEVICE_EXTERN_C
float log1pf(float x) { return __devicelib_log1pf(x); }

DEVICE_EXTERN_C
float log2f(float x) { return __devicelib_log2f(x); }

DEVICE_EXTERN_C
float logbf(float x) { return __devicelib_logbf(x); }

DEVICE_EXTERN_C
float sqrtf(float x) { return __devicelib_sqrtf(x); }

DEVICE_EXTERN_C
float cbrtf(float x) { return __devicelib_cbrtf(x); }

DEVICE_EXTERN_C
float hypotf(float x, float y) { return __devicelib_hypotf(x, y); }

DEVICE_EXTERN_C
float erff(float x) { return __devicelib_erff(x); }

DEVICE_EXTERN_C
float erfcf(float x) { return __devicelib_erfcf(x); }

DEVICE_EXTERN_C
float tgammaf(float x) { return __devicelib_tgammaf(x); }

DEVICE_EXTERN_C
float lgammaf(float x) { return __devicelib_lgammaf(x); }

DEVICE_EXTERN_C
float fmodf(float x, float y) { return __devicelib_fmodf(x, y); }

DEVICE_EXTERN_C
float remainderf(float x, float y) { return __devicelib_remainderf(x, y); }

DEVICE_EXTERN_C
float remquof(float x, float y, int *q) { return __devicelib_remquof(x, y, q); }

DEVICE_EXTERN_C
float nextafterf(float x, float y) { return __devicelib_nextafterf(x, y); }

DEVICE_EXTERN_C
float fdimf(float x, float y) { return __devicelib_fdimf(x, y); }

DEVICE_EXTERN_C
float fmaf(float x, float y, float z) { return __devicelib_fmaf(x, y, z); }

DEVICE_EXTERN_C
float sinf(float x) { return __devicelib_sinf(x); }

DEVICE_EXTERN_C
float cosf(float x) { return __devicelib_cosf(x); }

DEVICE_EXTERN_C
float tanf(float x) { return __devicelib_tanf(x); }

DEVICE_EXTERN_C
float powf(float x, float y) { return __devicelib_powf(x, y); }

DEVICE_EXTERN_C
float acosf(float x) { return __devicelib_acosf(x); }

DEVICE_EXTERN_C
float asinf(float x) { return __devicelib_asinf(x); }

DEVICE_EXTERN_C
float atanf(float x) { return __devicelib_atanf(x); }

DEVICE_EXTERN_C
float atan2f(float x, float y) { return __devicelib_atan2f(x, y); }

DEVICE_EXTERN_C
float coshf(float x) { return __devicelib_coshf(x); }

DEVICE_EXTERN_C
float sinhf(float x) { return __devicelib_sinhf(x); }

DEVICE_EXTERN_C
float tanhf(float x) { return __devicelib_tanhf(x); }

DEVICE_EXTERN_C
float acoshf(float x) { return __devicelib_acoshf(x); }

DEVICE_EXTERN_C
float asinhf(float x) { return __devicelib_asinhf(x); }

DEVICE_EXTERN_C
float atanhf(float x) { return __devicelib_atanhf(x); }
