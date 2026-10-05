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

#if __cplusplus >= 202002L && __has_include(<version>)
#include <version> // defines __cpp_lib_math_constants
#endif
// Correctly rounded pi for the inverse trig/hyperbolic functions; the
// atan2(+0, -0) idiom from libc++ is 1 ulp low for float on Intel GPUs.
#if __cpp_lib_math_constants
#include <numbers>
#define __DEVICELIB_PI std::numbers::pi_v<double>
#else
#define __DEVICELIB_PI 3.14159265358979323846
#endif

#if defined(__SPIR__) || defined(__SPIRV__)

// TODO: This needs to be more robust.
// clang doesn't recognize the c11 CMPLX macro, but it does have
//   its own syntax extension for initializing a complex as a struct.
#ifndef CMPLX
#define CMPLX(r, i) ((double __complex__){(double)(r), (double)(i)})
#endif
#ifndef CMPLXF
#define CMPLXF(r, i) ((float __complex__){(float)(r), (float)(i)})
#endif
#endif // __SPIR__ || __SPIRV__
#endif // __LIBDEVICE_DEVICE_COMPLEX_H_
