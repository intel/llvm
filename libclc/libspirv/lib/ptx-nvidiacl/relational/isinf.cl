//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#include <libspirv/ptx-nvidiacl/libdevice.h>
#include <libspirv/relational.h>

_CLC_DEF _CLC_OVERLOAD bool __spirv_IsInf(float x) { return __nv_isinff(x); }

// Simple implementation that doesn't require CUDA libdevice
// Check if exponent is all 1s and mantissa is 0

// _CLC_DEF _CLC_OVERLOAD bool __spirv_IsInf(float x) {
//  unsigned int bits = __builtin_bit_cast(unsigned int, x);
//  // Infinity: sign bit (any), exponent (0xFF), mantissa (0x0)
//  return ((bits & 0x7FFFFFFF) == 0x7F800000);
// }
// TODO double

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(char, __spirv_IsInf, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEF _CLC_OVERLOAD bool __spirv_IsInf(double x) { return __nv_isinfd(x); }

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(char, __spirv_IsInf, double)
#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEF _CLC_OVERLOAD bool __spirv_IsInf(half x) {
  float f = x;
  return __spirv_IsInf(f);
}

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(char, __spirv_IsInf, half)
#endif
