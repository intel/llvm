//==--- fallback-cmath.cpp - fallback implementation of math functions -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "device_math.h"

#ifdef __SPIR__

// To support fallback device libraries on-demand loading, please update the
// DeviceLibFuncMap in llvm/tools/sycl-post-link/sycl-post-link.cpp if you add
// or remove any item in this file.
// TODO: generate the DeviceLibFuncMap in sycl-post-link.cpp automatically
// during the build based on libdevice to avoid manually sync.

DEVICE_EXTERN_C
int __devicelib_abs(int x) { return x < 0 ? -x : x; }

DEVICE_EXTERN_C
long int __devicelib_labs(long int x) { return x < 0 ? -x : x; }

DEVICE_EXTERN_C
long long int __devicelib_llabs(long long int x) { return x < 0 ? -x : x; }

DEVICE_EXTERN_C
div_t __devicelib_div(int x, int y) { return {x / y, x % y}; }

DEVICE_EXTERN_C
ldiv_t __devicelib_ldiv(long x, long y) { return {x / y, x % y}; }

DEVICE_EXTERN_C
lldiv_t __devicelib_lldiv(long long x, long long y) { return {x / y, x % y}; }

DEVICE_EXTERN_C
float __devicelib_scalbnf(float x, int n) { return __spirv_ocl_ldexp(x, n); }

DEVICE_EXTERN_C
float __devicelib_logf(float x) { return __spirv_ocl_log(x); }

DEVICE_EXTERN_C
float __devicelib_expf(float x) { return __spirv_ocl_exp(x); }

DEVICE_EXTERN_C
float __devicelib_frexpf(float x, int *exp) {
  return __spirv_ocl_frexp(x, exp);
}

DEVICE_EXTERN_C
float __devicelib_ldexpf(float x, int exp) { return __spirv_ocl_ldexp(x, exp); }

DEVICE_EXTERN_C
float __devicelib_log10f(float x) { return __spirv_ocl_log10(x); }

DEVICE_EXTERN_C
float __devicelib_modff(float x, float *intpart) {
  return __spirv_ocl_modf(x, intpart);
}

DEVICE_EXTERN_C
float __devicelib_exp2f(float x) { return __spirv_ocl_exp2(x); }

DEVICE_EXTERN_C
float __devicelib_expm1f(float x) { return __spirv_ocl_expm1(x); }

DEVICE_EXTERN_C
int __devicelib_ilogbf(float x) { return __spirv_ocl_ilogb(x); }

DEVICE_EXTERN_C
float __devicelib_log1pf(float x) { return __spirv_ocl_log1p(x); }

DEVICE_EXTERN_C
float __devicelib_log2f(float x) { return __spirv_ocl_log2(x); }

DEVICE_EXTERN_C
float __devicelib_logbf(float x) { return __spirv_ocl_logb(x); }

DEVICE_EXTERN_C
float __devicelib_sqrtf(float x) { return __spirv_ocl_sqrt(x); }

DEVICE_EXTERN_C
float __devicelib_cbrtf(float x) { return __spirv_ocl_cbrt(x); }

DEVICE_EXTERN_C
float __devicelib_hypotf(float x, float y) { return __spirv_ocl_hypot(x, y); }

DEVICE_EXTERN_C
float __devicelib_erff(float x) { return __spirv_ocl_erf(x); }

DEVICE_EXTERN_C
float __devicelib_erfcf(float x) { return __spirv_ocl_erfc(x); }

DEVICE_EXTERN_C
float __devicelib_tgammaf(float x) { return __spirv_ocl_tgamma(x); }

DEVICE_EXTERN_C
float __devicelib_lgammaf(float x) { return __spirv_ocl_lgamma(x); }

DEVICE_EXTERN_C
float __devicelib_fmodf(float x, float y) { return __spirv_ocl_fmod(x, y); }

DEVICE_EXTERN_C
float __devicelib_remainderf(float x, float y) {
  return __spirv_ocl_remainder(x, y);
}

DEVICE_EXTERN_C
float __devicelib_remquof(float x, float y, int *q) {
  return __spirv_ocl_remquo(x, y, q);
}

DEVICE_EXTERN_C
float __devicelib_nextafterf(float x, float y) {
  return __spirv_ocl_nextafter(x, y);
}

DEVICE_EXTERN_C
float __devicelib_fdimf(float x, float y) { return __spirv_ocl_fdim(x, y); }

DEVICE_EXTERN_C
float __devicelib_fmaf(float x, float y, float z) {
  return __spirv_ocl_fma(x, y, z);
}

DEVICE_EXTERN_C
float __devicelib_sinf(float x) { return __spirv_ocl_sin(x); }

DEVICE_EXTERN_C
float __devicelib_cosf(float x) { return __spirv_ocl_cos(x); }

DEVICE_EXTERN_C
float __devicelib_tanf(float x) { return __spirv_ocl_tan(x); }

DEVICE_EXTERN_C
float __devicelib_powf(float x, float y) { return __spirv_ocl_pow(x, y); }

DEVICE_EXTERN_C
float __devicelib_acosf(float x) { return __spirv_ocl_acos(x); }

DEVICE_EXTERN_C
float __devicelib_asinf(float x) { return __spirv_ocl_asin(x); }

DEVICE_EXTERN_C
float __devicelib_atanf(float x) { return __spirv_ocl_atan(x); }

DEVICE_EXTERN_C
float __devicelib_atan2f(float x, float y) { return __spirv_ocl_atan2(x, y); }

DEVICE_EXTERN_C
float __devicelib_coshf(float x) { return __spirv_ocl_cosh(x); }

DEVICE_EXTERN_C
float __devicelib_sinhf(float x) { return __spirv_ocl_sinh(x); }

DEVICE_EXTERN_C
float __devicelib_tanhf(float x) { return __spirv_ocl_tanh(x); }

DEVICE_EXTERN_C
float __devicelib_acoshf(float x) { return __spirv_ocl_acosh(x); }

DEVICE_EXTERN_C
float __devicelib_asinhf(float x) { return __spirv_ocl_asinh(x); }

DEVICE_EXTERN_C
float __devicelib_atanhf(float x) { return __spirv_ocl_atanh(x); }

#endif // __SPIR__
