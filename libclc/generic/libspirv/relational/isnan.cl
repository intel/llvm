//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>
#include "relational.h"

_CLC_DEFINE_RELATIONAL_UNARY(int, __spirv_IsNan, __builtin_isnan, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// The scalar version of isnan(double) returns an int, but the vector versions
// return long.
_CLC_DEF _CLC_OVERLOAD int __spirv_IsNan(double x) {
  return __builtin_isnan(x);
}

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(long, __spirv_IsNan, double)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// The scalar version of isnan(half) returns an int, but the vector versions
// return short.
_CLC_DEF _CLC_OVERLOAD int __spirv_IsNan(half x) {
  return __builtin_isnan(x);
}

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(short, __spirv_IsNan, half)

#endif
