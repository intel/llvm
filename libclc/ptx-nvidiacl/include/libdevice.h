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

__LIBDEVICE_UNARY_BUILTIN(exp)
__LIBDEVICE_UNARY_BUILTIN(exp2)
__LIBDEVICE_UNARY_BUILTIN(exp10)
__LIBDEVICE_UNARY_BUILTIN(expm1)
__LIBDEVICE_UNARY_BUILTIN_F(fast_exp)
__LIBDEVICE_UNARY_BUILTIN_F(fast_exp10)
__LIBDEVICE_UNARY_BUILTIN(sqrt)

__LIBDEVICE_BINARY_BUILTIN(fmod)

#endif
