//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PTX_NVIDIACL_LIBDEVICE_H
#define PTX_NVIDIACL_LIBDEVICE_H

#define __LIBDEVICE_UNARY_BUILTIN_F(BUILTIN) float __nv_##BUILTIN##f(float);
#define __LIBDEVICE_BINARY_BUILTIN_F(BUILTIN)                                  \
  float __nv_##BUILTIN##f(float, float);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define __LIBDEVICE_UNARY_BUILTIN_D(BUILTIN) double __nv_##BUILTIN(double);
#define __LIBDEVICE_BINARY_BUILTIN_D(BUILTIN)                                  \
  double __nv_##BUILTIN(double, double);

#else

#define __LIBDEVICE_UNARY_BUILTIN_D(BUILTIN)

#endif

#define __LIBDEVICE_UNARY_BUILTIN(BUILTIN)                                     \
  __LIBDEVICE_UNARY_BUILTIN_F(BUILTIN)                                         \
  __LIBDEVICE_UNARY_BUILTIN_D(BUILTIN)

#define __LIBDEVICE_BINARY_BUILTIN(BUILTIN)                                    \
  __LIBDEVICE_BINARY_BUILTIN_F(BUILTIN)                                        \
  __LIBDEVICE_BINARY_BUILTIN_D(BUILTIN)

__LIBDEVICE_UNARY_BUILTIN(acos)
__LIBDEVICE_UNARY_BUILTIN(acosh)
__LIBDEVICE_UNARY_BUILTIN(asin)
__LIBDEVICE_UNARY_BUILTIN(asinh)
__LIBDEVICE_UNARY_BUILTIN(atan)
__LIBDEVICE_UNARY_BUILTIN(atanh)
__LIBDEVICE_UNARY_BUILTIN(cbrt)
__LIBDEVICE_UNARY_BUILTIN(ceil)
__LIBDEVICE_BINARY_BUILTIN(copysign)
__LIBDEVICE_UNARY_BUILTIN(cos)
__LIBDEVICE_UNARY_BUILTIN(cosh)
__LIBDEVICE_UNARY_BUILTIN(cospi)
__LIBDEVICE_UNARY_BUILTIN(erf)
__LIBDEVICE_UNARY_BUILTIN(erfc)
__LIBDEVICE_UNARY_BUILTIN(exp)
__LIBDEVICE_UNARY_BUILTIN(exp2)
__LIBDEVICE_UNARY_BUILTIN(exp10)
__LIBDEVICE_UNARY_BUILTIN(expm1)
__LIBDEVICE_UNARY_BUILTIN(fabs)
__LIBDEVICE_UNARY_BUILTIN(fdim)
__LIBDEVICE_UNARY_BUILTIN(floor)
__LIBDEVICE_UNARY_BUILTIN_F(fast_exp)
__LIBDEVICE_UNARY_BUILTIN_F(fast_exp10)
__LIBDEVICE_UNARY_BUILTIN(lgamma)
__LIBDEVICE_UNARY_BUILTIN(sqrt)

__LIBDEVICE_BINARY_BUILTIN(atan2)
__LIBDEVICE_BINARY_BUILTIN(fma)
__LIBDEVICE_BINARY_BUILTIN(fmax)
__LIBDEVICE_BINARY_BUILTIN(fmin)
__LIBDEVICE_BINARY_BUILTIN(fmod)

#endif
