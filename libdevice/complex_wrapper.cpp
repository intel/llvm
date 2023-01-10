//==--- complex_wrapper.cpp - wrappers for C99 complex math functions ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "device_complex.h"

#ifdef __SPIR__

DEVICE_EXTERN_C_INLINE
float cimagf(float __complex__ z) { return __devicelib_cimagf(z); }

DEVICE_EXTERN_C_INLINE
float crealf(float __complex__ z) { return __devicelib_crealf(z); }

DEVICE_EXTERN_C_INLINE
float cargf(float __complex__ z) { return __devicelib_cargf(z); }

DEVICE_EXTERN_C_INLINE
float cabsf(float __complex__ z) { return __devicelib_cabsf(z); }

DEVICE_EXTERN_C_INLINE
float __complex__ cprojf(float __complex__ z) { return __devicelib_cprojf(z); }

DEVICE_EXTERN_C_INLINE
float __complex__ cexpf(float __complex__ z) { return __devicelib_cexpf(z); }

DEVICE_EXTERN_C_INLINE
float __complex__ clogf(float __complex__ z) { return __devicelib_clogf(z); }

DEVICE_EXTERN_C_INLINE
float __complex__ cpowf(float __complex__ x, float __complex__ y) {
  return __devicelib_cpowf(x, y);
}

DEVICE_EXTERN_C_INLINE
float __complex__ cpolarf(float rho, float theta) {
  return __devicelib_cpolarf(rho, theta);
}

DEVICE_EXTERN_C_INLINE
float __complex__ csqrtf(float __complex__ z) { return __devicelib_csqrtf(z); }

DEVICE_EXTERN_C_INLINE
float __complex__ csinhf(float __complex__ z) { return __devicelib_csinhf(z); }

DEVICE_EXTERN_C_INLINE
float __complex__ ccoshf(float __complex__ z) { return __devicelib_ccoshf(z); }

DEVICE_EXTERN_C_INLINE
float __complex__ ctanhf(float __complex__ z) { return __devicelib_ctanhf(z); }

DEVICE_EXTERN_C_INLINE
float __complex__ csinf(float __complex__ z) { return __devicelib_csinf(z); }

DEVICE_EXTERN_C_INLINE
float __complex__ ccosf(float __complex__ z) { return __devicelib_ccosf(z); }

DEVICE_EXTERN_C_INLINE
float __complex__ ctanf(float __complex__ z) { return __devicelib_ctanf(z); }

DEVICE_EXTERN_C_INLINE
float __complex__ cacosf(float __complex__ z) { return __devicelib_cacosf(z); }

DEVICE_EXTERN_C_INLINE
float __complex__ casinhf(float __complex__ z) {
  return __devicelib_casinhf(z);
}

DEVICE_EXTERN_C_INLINE
float __complex__ casinf(float __complex__ z) { return __devicelib_casinf(z); }

DEVICE_EXTERN_C_INLINE
float __complex__ cacoshf(float __complex__ z) {
  return __devicelib_cacoshf(z);
}

DEVICE_EXTERN_C_INLINE
float __complex__ catanhf(float __complex__ z) {
  return __devicelib_catanhf(z);
}

DEVICE_EXTERN_C_INLINE
float __complex__ catanf(float __complex__ z) { return __devicelib_catanf(z); }

// __mulsc3
// Returns: the product of a + ib and c + id
DEVICE_EXTERN_C_INLINE
float __complex__ __mulsc3(float __a, float __b, float __c, float __d) {
  return __devicelib___mulsc3(__a, __b, __c, __d);
}

// __divsc3
// Returns: the quotient of (a + ib) / (c + id)
DEVICE_EXTERN_C_INLINE
float __complex__ __divsc3(float __a, float __b, float __c, float __d) {
  return __devicelib___divsc3(__a, __b, __c, __d);
}
#endif // __SPIR__
