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

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEF _CLC_OVERLOAD bool __spirv_IsFinite(double x) {
  return __nv_isfinited(x);
}

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(schar, __spirv_IsFinite, double)

#endif

_CLC_DEF _CLC_OVERLOAD bool __spirv_IsFinite(float x) {
  return __nv_isfinited(x);
}

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(schar, __spirv_IsFinite, float)

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEF _CLC_OVERLOAD bool __spirv_IsFinite(half x) {
  return __nv_isfinited(x);
}

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(schar, __spirv_IsFinite, half)

#endif
