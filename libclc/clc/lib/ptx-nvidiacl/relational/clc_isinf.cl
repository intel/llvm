//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/relational/clc_isinf.h>

int __nv_isinff(float);
int __nv_isinfd(double);

_CLC_OVERLOAD _CLC_DEF int __clc_isinf(float x) { return __nv_isinff(x); }

// Simple implementation that doesn't require CUDA libdevice
// Check if exponent is all 1s and mantissa is 0
// _CLC_OVERLOAD _CLC_DEF int __clc_isinf(float x) {
//  unsigned int bits = __builtin_bit_cast(unsigned int, x);
//  // Infinity: sign bit (any), exponent (0xFF), mantissa (0x0)
//  return ((bits & 0x7FFFFFFF) == 0x7F800000);
//}
// TODO double

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF int __clc_isinf(double x) { return __nv_isinfd(x); }

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF int __clc_isinf(half x) { return __clc_isinf((float)x); }

#endif

#define __CLC_FUNCTION __clc_isinf
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#define __CLC_RET_TYPE __CLC_BIT_INT
#include <clc/math/gentype.inc>
