//===----------- imf_fp32_dl.cpp - fp32 functions required by DL ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// Subset of intel math functions is required by deep learning frameworks and
/// we decide to keep all functions required in a separate file and build a new
/// spirv module using this file. By this way, we can reduce unnecessary jit
/// overhead in these deep learning frameworks.
//===----------------------------------------------------------------------===//

#include "../device_imf.hpp"
#ifdef __LIBDEVICE_IMF_ENABLED__

DEVICE_EXTERN_C_INLINE int32_t __devicelib_imf_abs(int32_t x) {
  return (x >= 0) ? x : -x;
}

DEVICE_EXTERN_C_INLINE float __devicelib_imf_fabsf(float x) {
  return __fabs(x);
}

DEVICE_EXTERN_C_INLINE
int64_t __devicelib_imf_llabs(int64_t x) { return x >= 0 ? x : -x; }

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_fast_exp10f(float x) { return __fast_exp10f(x); }

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_fast_expf(float x) { return __fast_expf(x); }

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_fast_fdividef(float x, float y) {
  return __fast_fdividef(x, y);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_fast_logf(float x) { return __fast_logf(x); }

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_fast_log2f(float x) { return __fast_log2f(x); }

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_fast_log10f(float x) { return __fast_log10f(x); }

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_fast_powf(float x, float y) { return __fast_powf(x, y); }

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_max(int x, int y) { return x > y ? x : y; }

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_min(int x, int y) { return x < y ? x : y; }

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_hadd(int x, int y) { return __shadd(x, y); }
#endif /*__LIBDEVICE_IMF_ENABLED__*/
