//==--- complex_wrapper.hpp - wrappers for C99 complex math functions ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "fallback-complex-fp64.hpp"
#include "fallback-complex.hpp"

#if defined(__SPIR__) || defined(__SPIRV__)

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

DEVICE_EXTERN_C_INLINE
double cimag(double __complex__ z) { return __devicelib_cimag(z); }

DEVICE_EXTERN_C_INLINE
double creal(double __complex__ z) { return __devicelib_creal(z); }

DEVICE_EXTERN_C_INLINE
double cabs(double __complex__ z) { return __devicelib_cabs(z); }

DEVICE_EXTERN_C_INLINE
double carg(double __complex__ z) { return __devicelib_carg(z); }

DEVICE_EXTERN_C_INLINE
double __complex__ cproj(double __complex__ z) { return __devicelib_cproj(z); }

DEVICE_EXTERN_C_INLINE
double __complex__ cexp(double __complex__ z) { return __devicelib_cexp(z); }

DEVICE_EXTERN_C_INLINE
double __complex__ clog(double __complex__ z) { return __devicelib_clog(z); }

DEVICE_EXTERN_C_INLINE
double __complex__ cpow(double __complex__ x, double __complex__ y) {
  return __devicelib_cpow(x, y);
}

DEVICE_EXTERN_C_INLINE
double __complex__ cpolar(double rho, double theta) {
  return __devicelib_cpolar(rho, theta);
}

DEVICE_EXTERN_C_INLINE
double __complex__ csqrt(double __complex__ z) { return __devicelib_csqrt(z); }

DEVICE_EXTERN_C_INLINE
double __complex__ csinh(double __complex__ z) { return __devicelib_csinh(z); }

DEVICE_EXTERN_C_INLINE
double __complex__ ccosh(double __complex__ z) { return __devicelib_ccosh(z); }

DEVICE_EXTERN_C_INLINE
double __complex__ ctanh(double __complex__ z) { return __devicelib_ctanh(z); }

DEVICE_EXTERN_C_INLINE
double __complex__ csin(double __complex__ z) { return __devicelib_csin(z); }

DEVICE_EXTERN_C_INLINE
double __complex__ ccos(double __complex__ z) { return __devicelib_ccos(z); }

DEVICE_EXTERN_C_INLINE
double __complex__ ctan(double __complex__ z) { return __devicelib_ctan(z); }

DEVICE_EXTERN_C_INLINE
double __complex__ cacos(double __complex__ z) { return __devicelib_cacos(z); }

DEVICE_EXTERN_C_INLINE
double __complex__ casinh(double __complex__ z) {
  return __devicelib_casinh(z);
}

DEVICE_EXTERN_C_INLINE
double __complex__ casin(double __complex__ z) { return __devicelib_casin(z); }

DEVICE_EXTERN_C_INLINE
double __complex__ cacosh(double __complex__ z) {
  return __devicelib_cacosh(z);
}

DEVICE_EXTERN_C_INLINE
double __complex__ catanh(double __complex__ z) {
  return __devicelib_catanh(z);
}

DEVICE_EXTERN_C_INLINE
double __complex__ catan(double __complex__ z) { return __devicelib_catan(z); }

// __muldc3
// Returns: the product of a + ib and c + id
DEVICE_EXTERN_C_INLINE
double __complex__ __muldc3(double __a, double __b, double __c, double __d) {
  return __devicelib___muldc3(__a, __b, __c, __d);
}

// __divdc3
// Returns: the quotient of (a + ib) / (c + id)
DEVICE_EXTERN_C_INLINE
double __complex__ __divdc3(double __a, double __b, double __c, double __d) {
  return __devicelib___divdc3(__a, __b, __c, __d);
}
#endif // __SPIR__ || __SPIRV__
