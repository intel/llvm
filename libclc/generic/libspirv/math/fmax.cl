//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clcmacro.h>
#include <spirv/spirv.h>

_CLC_DEFINE_BINARY_BUILTIN(float, __spirv_ocl_fmax, __builtin_fmaxf, float, float);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEFINE_BINARY_BUILTIN(double, __spirv_ocl_fmax, __builtin_fmax, double, double);

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEF _CLC_OVERLOAD half __spirv_ocl_fmax(half x, half y)
{
   if (__spirv_IsNan(x))
      return y;
   if (__spirv_IsNan(y))
      return x;
   return (x < y) ? y : x;
}
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, __spirv_ocl_fmax, half, half)

#endif

#define __CLC_BODY <fmax.inc>
#include <clc/math/gentype.inc>
