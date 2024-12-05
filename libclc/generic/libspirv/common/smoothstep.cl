//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

#include <clc/clcmacro.h>

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_smoothstep(float edge0, float edge1,
                                                    float x) {
  float t = __spirv_ocl_fclamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __spirv_ocl_smoothstep,
                       float, float, float)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double __spirv_ocl_smoothstep(double edge0, double edge1,
                                                     double x) {
  double t = __spirv_ocl_fclamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, __spirv_ocl_smoothstep,
                       double, double, double)

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF half __spirv_ocl_smoothstep(half edge0, half edge1,
                                                   half x) {
  half t = __spirv_ocl_fclamp((x - edge0) / (edge1 - edge0), 0.0h, 1.0h);
  return t * t * (3.0h - 2.0h * t);
}

_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, __spirv_ocl_smoothstep,
                       half, half, half);

#endif
