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
DEVICE_EXTERN_C
double __devicelib_log(double x) { return __spirv_ocl_log(x); }

DEVICE_EXTERN_C
double __devicelib_exp(double x) { return __spirv_ocl_exp(x); }

DEVICE_EXTERN_C
double __devicelib_frexp(double x, int *exp) {
  return __spirv_ocl_frexp(x, exp);
}

DEVICE_EXTERN_C
double __devicelib_ldexp(double x, int exp) {
  return __spirv_ocl_ldexp(x, exp);
}

DEVICE_EXTERN_C
double __devicelib_log10(double x) { return __spirv_ocl_log10(x); }

DEVICE_EXTERN_C
double __devicelib_modf(double x, double *intpart) {
  return __spirv_ocl_modf(x, intpart);
}

DEVICE_EXTERN_C
double __devicelib_exp2(double x) { return __spirv_ocl_exp2(x); }

DEVICE_EXTERN_C
double __devicelib_expm1(double x) { return __spirv_ocl_expm1(x); }

DEVICE_EXTERN_C
int __devicelib_ilogb(double x) { return __spirv_ocl_ilogb(x); }

DEVICE_EXTERN_C
double __devicelib_log1p(double x) { return __spirv_ocl_log1p(x); }

DEVICE_EXTERN_C
double __devicelib_log2(double x) { return __spirv_ocl_log2(x); }

DEVICE_EXTERN_C
double __devicelib_logb(double x) { return __spirv_ocl_logb(x); }

DEVICE_EXTERN_C
double __devicelib_sqrt(double x) { return __spirv_ocl_sqrt(x); }

DEVICE_EXTERN_C
double __devicelib_cbrt(double x) { return __spirv_ocl_cbrt(x); }

DEVICE_EXTERN_C
double __devicelib_hypot(double x, double y) { return __spirv_ocl_hypot(x, y); }

DEVICE_EXTERN_C
double __devicelib_erf(double x) { return __spirv_ocl_erf(x); }

DEVICE_EXTERN_C
double __devicelib_erfc(double x) { return __spirv_ocl_erfc(x); }

DEVICE_EXTERN_C
double __devicelib_tgamma(double x) { return __spirv_ocl_tgamma(x); }

DEVICE_EXTERN_C
double __devicelib_lgamma(double x) { return __spirv_ocl_lgamma(x); }

DEVICE_EXTERN_C
double __devicelib_fmod(double x, double y) { return __spirv_ocl_fmod(x, y); }

DEVICE_EXTERN_C
double __devicelib_remainder(double x, double y) {
  return __spirv_ocl_remainder(x, y);
}

DEVICE_EXTERN_C
double __devicelib_remquo(double x, double y, int *q) {
  return __spirv_ocl_remquo(x, y, q);
}

DEVICE_EXTERN_C
double __devicelib_nextafter(double x, double y) {
  return __spirv_ocl_nextafter(x, y);
}

DEVICE_EXTERN_C
double __devicelib_fdim(double x, double y) { return __spirv_ocl_fdim(x, y); }

DEVICE_EXTERN_C
double __devicelib_fma(double x, double y, double z) {
  return __spirv_ocl_fma(x, y, z);
}

DEVICE_EXTERN_C
double __devicelib_sin(double x) { return __spirv_ocl_sin(x); }

DEVICE_EXTERN_C
double __devicelib_cos(double x) { return __spirv_ocl_cos(x); }

DEVICE_EXTERN_C
double __devicelib_tan(double x) { return __spirv_ocl_tan(x); }

DEVICE_EXTERN_C
double __devicelib_pow(double x, double y) { return __spirv_ocl_pow(x, y); }

DEVICE_EXTERN_C
double __devicelib_acos(double x) { return __spirv_ocl_acos(x); }

DEVICE_EXTERN_C
double __devicelib_asin(double x) { return __spirv_ocl_asin(x); }

DEVICE_EXTERN_C
double __devicelib_atan(double x) { return __spirv_ocl_atan(x); }

DEVICE_EXTERN_C
double __devicelib_atan2(double x, double y) { return __spirv_ocl_atan2(x, y); }

DEVICE_EXTERN_C
double __devicelib_cosh(double x) { return __spirv_ocl_cosh(x); }

DEVICE_EXTERN_C
double __devicelib_sinh(double x) { return __spirv_ocl_sinh(x); }

DEVICE_EXTERN_C
double __devicelib_tanh(double x) { return __spirv_ocl_tanh(x); }

DEVICE_EXTERN_C
double __devicelib_acosh(double x) { return __spirv_ocl_acosh(x); }

DEVICE_EXTERN_C
double __devicelib_asinh(double x) { return __spirv_ocl_asinh(x); }

DEVICE_EXTERN_C
double __devicelib_atanh(double x) { return __spirv_ocl_atanh(x); }

DEVICE_EXTERN_C
double __devicelib_scalbn(double x, int exp) {
  return __spirv_ocl_ldexp(x, exp);
}
#endif // __SPIR__
