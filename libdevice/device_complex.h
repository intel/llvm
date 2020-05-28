//==------- device_complex.h - complex devicelib functions declarations-----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==------------------------------------------------------------------------==//
#ifndef __LIBDEVICE_DEVICE_COMPLEX_H_
#define __LIBDEVICE_DEVICE_COMPLEX_H_

#include "device.h"

#ifdef __SPIR__

// TODO: This needs to be more robust.
// clang doesn't recognize the c11 CMPLX macro, but it does have
//   its own syntax extension for initializing a complex as a struct.
#ifndef CMPLX
#define CMPLX(r, i) ((double __complex__){(double)(r), (double)(i)})
#endif
#ifndef CMPLXF
#define CMPLXF(r, i) ((float __complex__){(float)(r), (float)(i)})
#endif

DEVICE_EXTERN_C
double __devicelib_cimag(double __complex__ z);

DEVICE_EXTERN_C
float __devicelib_cimagf(float __complex__ z);

DEVICE_EXTERN_C
double __devicelib_creal(double __complex__ z);

DEVICE_EXTERN_C
float __devicelib_crealf(float __complex__ z);

DEVICE_EXTERN_C
double __devicelib_carg(double __complex__ z);

DEVICE_EXTERN_C
float __devicelib_cargf(float __complex__ z);

DEVICE_EXTERN_C
double __devicelib_cabs(double __complex__ z);

DEVICE_EXTERN_C
float __devicelib_cabsf(float __complex__ z);

DEVICE_EXTERN_C
double __complex__ __devicelib_cproj(double __complex__ z);

DEVICE_EXTERN_C
float __complex__ __devicelib_cprojf(float __complex__ z);

DEVICE_EXTERN_C
double __complex__ __devicelib_cexp(double __complex__ z);

DEVICE_EXTERN_C
float __complex__ __devicelib_cexpf(float __complex__ z);

DEVICE_EXTERN_C
double __complex__ __devicelib_clog(double __complex__ z);

DEVICE_EXTERN_C
float __complex__ __devicelib_clogf(float __complex__ z);

DEVICE_EXTERN_C
double __complex__ __devicelib_cpow(double __complex__ x, double __complex__ y);

DEVICE_EXTERN_C
float __complex__ __devicelib_cpowf(float __complex__ x, float __complex__ y);

DEVICE_EXTERN_C
double __complex__ __devicelib_cpolar(double x, double y);

DEVICE_EXTERN_C
float __complex__ __devicelib_cpolarf(float x, float y);

DEVICE_EXTERN_C
double __complex__ __devicelib_csqrt(double __complex__ z);

DEVICE_EXTERN_C
float __complex__ __devicelib_csqrtf(float __complex__ z);

DEVICE_EXTERN_C
double __complex__ __devicelib_csinh(double __complex__ z);

DEVICE_EXTERN_C
float __complex__ __devicelib_csinhf(float __complex__ z);

DEVICE_EXTERN_C
double __complex__ __devicelib_ccosh(double __complex__ z);

DEVICE_EXTERN_C
float __complex__ __devicelib_ccoshf(float __complex__ z);

DEVICE_EXTERN_C
double __complex__ __devicelib_ctanh(double __complex__ z);

DEVICE_EXTERN_C
float __complex__ __devicelib_ctanhf(float __complex__ z);

DEVICE_EXTERN_C
double __complex__ __devicelib_csin(double __complex__ z);

DEVICE_EXTERN_C
float __complex__ __devicelib_csinf(float __complex__ z);

DEVICE_EXTERN_C
double __complex__ __devicelib_ccos(double __complex__ z);

DEVICE_EXTERN_C
float __complex__ __devicelib_ccosf(float __complex__ z);

DEVICE_EXTERN_C
double __complex__ __devicelib_ctan(double __complex__ z);

DEVICE_EXTERN_C
float __complex__ __devicelib_ctanf(float __complex__ z);

DEVICE_EXTERN_C
double __complex__ __devicelib_cacos(double __complex__ z);

DEVICE_EXTERN_C
float __complex__ __devicelib_cacosf(float __complex__ z);

DEVICE_EXTERN_C
double __complex__ __devicelib_casinh(double __complex__ z);

DEVICE_EXTERN_C
float __complex__ __devicelib_casinhf(float __complex__ z);

DEVICE_EXTERN_C
double __complex__ __devicelib_casin(double __complex__ z);

DEVICE_EXTERN_C
float __complex__ __devicelib_casinf(float __complex__ z);

DEVICE_EXTERN_C
double __complex__ __devicelib_cacosh(double __complex__ z);

DEVICE_EXTERN_C
float __complex__ __devicelib_cacoshf(float __complex__ z);

DEVICE_EXTERN_C
double __complex__ __devicelib_catanh(double __complex__ z);

DEVICE_EXTERN_C
float __complex__ __devicelib_catanhf(float __complex__ z);

DEVICE_EXTERN_C
double __complex__ __devicelib_catan(double __complex__ z);

DEVICE_EXTERN_C
float __complex__ __devicelib_catanf(float __complex__ z);

DEVICE_EXTERN_C
double __complex__ __devicelib___muldc3(double a, double b, double c, double d);

DEVICE_EXTERN_C
float __complex__ __devicelib___mulsc3(float a, float b, float c, float d);

DEVICE_EXTERN_C
double __complex__ __devicelib___divdc3(double a, double b, double c, double d);

DEVICE_EXTERN_C
float __complex__ __devicelib___divsc3(float a, float b, float c, float d);
#endif // __SPIR__
#endif // __LIBDEVICE_DEVICE_COMPLEX_H_
