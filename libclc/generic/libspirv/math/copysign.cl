//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clcmacro.h>
#include <spirv/spirv.h>

_CLC_DEFINE_BINARY_BUILTIN(float, __spirv_ocl_copysign, __builtin_copysignf,
                           float, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEFINE_BINARY_BUILTIN(double, __spirv_ocl_copysign, __builtin_copysign,
                           double, double)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEF _CLC_OVERLOAD half __spirv_ocl_copysign(half x, half y) {
  ushort sign_x = as_ushort(x) & 0x8000u;
  ushort unsigned_y = as_ushort(y) & 0x7ffffu;

  return as_half((ushort)(sign_x | unsigned_y));
}
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, __spirv_ocl_copysign, half,
                      half)

#endif
