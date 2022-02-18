//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clcmacro.h>

extern int __clc_nvvm_reflect_arch();

_CLC_DEF _CLC_OVERLOAD float __spirv_ocl_fmin(float x, float y) {
  return __nvvm_fmin_f(x, y);
}
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __spirv_ocl_fmin, float,
                      float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEF _CLC_OVERLOAD double __spirv_ocl_fmin(double x, double y) {
  return __nvvm_fmin_d(x, y);
}
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, __spirv_ocl_fmin, double,
                      double)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEF _CLC_OVERLOAD half __spirv_ocl_fmin(half x, half y) {
  if (__clc_nvvm_reflect_arch() >= 800) {
    return __nvvm_fmin_f16(x, y);
  }
  return x < y ? x : y;
}
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, __spirv_ocl_fmin, half,
                      half)

#endif

_CLC_DEF _CLC_OVERLOAD ushort __spirv_ocl_fmin(ushort x, ushort y) {
  if (__clc_nvvm_reflect_arch() >= 800) {
    return __nvvm_max_rn_bf16(x, y);
  }
  __builtin_trap();
  __builtin_unreachable();
}
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ushort, __spirv_ocl_fmin, ushort,
                      ushort)

_CLC_DEF _CLC_OVERLOAD uint __spirv_ocl_fmin(uint x, uint y) {
  if (__clc_nvvm_reflect_arch() >= 800) {
    return __nvvm_max_rn_bf16x2(x, y);
  }
  __builtin_trap();
  __builtin_unreachable();
}
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uint, __spirv_ocl_fmin, uint,
                      uint)