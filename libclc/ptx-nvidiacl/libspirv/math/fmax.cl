//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clcmacro.h>
#include <spirv/spirv.h>

extern int __clc_nvvm_reflect_arch();

_CLC_DEF _CLC_OVERLOAD float __spirv_ocl_fmax(float x, float y) {
  return __nvvm_fmax_f(x, y);
}
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __spirv_ocl_fmax, float,
                      float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEF _CLC_OVERLOAD double __spirv_ocl_fmax(double x, double y) {
  return __nvvm_fmax_d(x, y);
}
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, __spirv_ocl_fmax, double,
                      double)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEF _CLC_OVERLOAD half __spirv_ocl_fmax(half x, half y) {
  if (__clc_nvvm_reflect_arch() >= 800) {
    return __nvvm_fmax_f16(x, y);
  }
  return x > y ? x : y;
}
_CLC_DEF _CLC_OVERLOAD half2 __spirv_ocl_fmax(half2 x, half2 y) {
  if (__clc_nvvm_reflect_arch() >= 800) {
    return __nvvm_fmax_f16x2(x, y);
  }
  return (half2)(__spirv_ocl_fmax(x.x, y.x), __spirv_ocl_fmax(x.y, y.y));
}
_CLC_BINARY_VECTORIZE_HAVE2(_CLC_OVERLOAD _CLC_DEF, half, __spirv_ocl_fmax,
                            half, half)

#endif

_CLC_DEF _CLC_OVERLOAD ushort __clc_fmax(ushort x, ushort y) {
  if (__clc_nvvm_reflect_arch() >= 800) {
    return __nvvm_fmax_bf16(x, y);
  }
  __builtin_trap();
  __builtin_unreachable();
}
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ushort, __clc_fmax, ushort,
                      ushort)

_CLC_DEF _CLC_OVERLOAD uint __clc_fmax(uint x, uint y) {
  if (__clc_nvvm_reflect_arch() >= 800) {
    return __nvvm_fmax_bf16x2(x, y);
  }
  __builtin_trap();
  __builtin_unreachable();
}
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uint, __clc_fmax, uint,
                      uint)