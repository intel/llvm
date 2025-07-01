//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_fmax(float x, float y) {
  return __builtin_fmaxf(x);
}

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
_CLC_OVERLOAD _CLC_DEF double __spirv_ocl_fmax(double x, double y) {
  return __builtin_fmax(x);
}
#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
_CLC_OVERLOAD _CLC_DEF half __spirv_ocl_fmax(half x, half y) {
  if (__spirv_IsNan(x))
    return y;
  if (__spirv_IsNan(y))
    return x;
  return (x < y) ? y : x;
}
#endif

#define FUNCTION __spirv_ocl_fmax
#define __CLC_BODY <clc/shared/binary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef FUNCTION

#define __CLC_BODY <fmax.inc>
#include <clc/math/gentype.inc>
