//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

#include <clcmacro.h>
#include <math/math.h>

_CLC_OVERLOAD _CLC_DEF int __spirv_ocl_ilogb(float x) {
  uint ux = as_uint(x);
  uint ax = ux & EXSIGNBIT_SP32;
  int rs = -118 - (int)__spirv_ocl_clz(ux & MANTBITS_SP32);
  int r = (int)(ax >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;
  r = ax < 0x00800000U ? rs : r;
  r = ax > EXPBITS_SP32 || ax == 0 ? 0x80000000 : r;
  r = ax == EXPBITS_SP32 ? 0x7fffffff : r;
  return r;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, int, __spirv_ocl_ilogb, float);

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF int __spirv_ocl_ilogb(double x) {
  ulong ux = as_ulong(x);
  ulong ax = ux & ~SIGNBIT_DP64;
  int r = (int)(ax >> EXPSHIFTBITS_DP64) - EXPBIAS_DP64;
  int rs = -1011 - (int)__spirv_ocl_clz(ax & MANTBITS_DP64);
  r = ax < 0x0010000000000000UL ? rs : r;
  r = ax > 0x7ff0000000000000UL || ax == 0UL ? 0x80000000 : r;
  r = ax == 0x7ff0000000000000UL ? 0x7fffffff : r;
  return r;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, int, __spirv_ocl_ilogb, double);

#endif // cl_khr_fp64

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEF _CLC_OVERLOAD int __spirv_ocl_ilogb(half x) {
  float f = x;
  return __spirv_ocl_ilogb(f);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, int, __spirv_ocl_ilogb, half)


#endif
