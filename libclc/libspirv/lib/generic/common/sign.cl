//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clcmacro.h>
#include <libspirv/spirv.h>

#define SIGN(TYPE, F)                                                          \
  _CLC_DEF _CLC_OVERLOAD TYPE __spirv_ocl_sign(TYPE x) {                       \
    if (__spirv_IsNan(x)) {                                                    \
      return 0.0F;                                                             \
    }                                                                          \
    if (x > 0.0F) {                                                            \
      return 1.0F;                                                             \
    }                                                                          \
    if (x < 0.0F) {                                                            \
      return -1.0F;                                                            \
    }                                                                          \
    return x; /* -0.0 or +0.0 */                                               \
  }

SIGN(float, f)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __spirv_ocl_sign, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

SIGN(double, )
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, __spirv_ocl_sign, double)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

SIGN(half, h)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, __spirv_ocl_sign, half)

#endif
