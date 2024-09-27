//==----- imf_inline_fp64.cpp - some fp64 trivial intel math functions -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../device.h"

#ifdef __LIBDEVICE_IMF_ENABLED__

#include "../device_imf.hpp"

DEVICE_EXTERN_C_INLINE double __devicelib_imf_fma(double a, double b,
                                                  double c) {
  return __fma(a, b, c);
}

DEVICE_EXTERN_C_INLINE double __devicelib_imf_floor(double x) {
  return __floor(x);
}

DEVICE_EXTERN_C_INLINE double __devicelib_imf_ceil(double x) {
  return __ceil(x);
}

DEVICE_EXTERN_C_INLINE double __devicelib_imf_trunc(double x) {
  return __trunc(x);
}

DEVICE_EXTERN_C_INLINE double __devicelib_imf_rint(double x) {
  return __rint(x);
}

DEVICE_EXTERN_C_INLINE double __devicelib_imf_nearbyint(double x) {
  return __rint(x);
}

DEVICE_EXTERN_C_INLINE double __devicelib_imf_sqrt(double a) {
  return __sqrt(a);
}

DEVICE_EXTERN_C_INLINE double __devicelib_imf_rsqrt(double a) {
  return 1.0 / __sqrt(a);
}

DEVICE_EXTERN_C_INLINE double __devicelib_imf_inv(double a) { return 1.0 / a; }

DEVICE_EXTERN_C_INLINE double __devicelib_imf_copysign(double a, double b) {
  return __copysign(a, b);
}
#endif /*__LIBDEVICE_IMF_ENABLED__*/
