//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

#include <clc/clcmacro.h>

_CLC_DEFINE_BINARY_BUILTIN(float, __spirv_ocl_fmin, __builtin_fminf, float, float);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEFINE_BINARY_BUILTIN(double, __spirv_ocl_fmin, __builtin_fmin, double, double);

#endif
#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEF _CLC_OVERLOAD half __spirv_ocl_fmin(half x, half y)
{
   if (__spirv_IsNan(x))
      return y;
   if (__spirv_IsNan(y))
      return x;
   return (y < x) ? y : x;
}
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, __spirv_ocl_fmin, half, half)

#endif

#define __CLC_BODY <fmin.inc>
#include <clc/math/gentype.inc>
