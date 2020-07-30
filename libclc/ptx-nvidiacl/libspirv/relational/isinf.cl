//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

#include "../../include/libdevice.h"
#include <relational.h>

_CLC_DEF _CLC_OVERLOAD bool __spirv_IsInf(float x) { return __nv_isinff(x); }

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(schar, __spirv_IsInf, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEF _CLC_OVERLOAD bool __spirv_IsInf(double x) { return __nv_isinfd(x); }

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(schar, __spirv_IsInf, double)
#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEF _CLC_OVERLOAD bool __spirv_IsInf(half x) {
  float f = x;
  return __spirv_IsInf(f);
}

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(schar, __spirv_IsInf, half)
#endif
