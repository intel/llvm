//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>
#include "relational.h"

_CLC_DEFINE_RELATIONAL_UNARY(int, __spirv_IsInf, __builtin_isinf, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// The scalar version of isinf(double) returns an int, but the vector versions
// return long.
_CLC_DEF _CLC_OVERLOAD int __spirv_IsInf(double x) {
  return __builtin_isinf(x);
}

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(long, __spirv_IsInf, double)
#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// The scalar version of isinf(half) returns an int, but the vector versions
// return short.
_CLC_DEF _CLC_OVERLOAD int __spirv_IsInf(half x) {
  return __builtin_isinf(x);
}

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(short, __spirv_IsInf, half)
#endif
