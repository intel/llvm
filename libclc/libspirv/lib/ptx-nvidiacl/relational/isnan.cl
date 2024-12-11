//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#include <libspirv/ptx-nvidiacl/libdevice.h>
#include <relational.h>

_CLC_DEF _CLC_OVERLOAD bool __spirv_IsNan(float x) { return __nv_isnanf(x); }

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(schar, __spirv_IsNan, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEF _CLC_OVERLOAD bool __spirv_IsNan(double x) { return __nv_isnand(x); }

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(schar, __spirv_IsNan, double)
#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEF _CLC_OVERLOAD bool __spirv_IsNan(half x) {
  float f = x;
  return __spirv_IsNan(f);
}

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(schar, __spirv_IsNan, half)
#endif
