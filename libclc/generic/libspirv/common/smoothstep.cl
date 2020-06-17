//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

#include <clcmacro.h>

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_smoothstep(float edge0, float edge1,
                                                    float x) {
  float t = __spirv_ocl_fclamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __spirv_ocl_smoothstep, float, float, float);

_CLC_V_S_S_V_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __spirv_ocl_smoothstep, float, float, float);

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define SMOOTH_STEP_DEF(edge_type, x_type, impl)                               \
  _CLC_OVERLOAD _CLC_DEF x_type __spirv_ocl_smoothstep(                        \
      edge_type edge0, edge_type edge1, x_type x) {                            \
    double t = __spirv_ocl_fclamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);    \
    return t * t * (3.0 - 2.0 * t);                                            \
  }

SMOOTH_STEP_DEF(double, double, SMOOTH_STEP_IMPL_D);

_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, __spirv_ocl_smoothstep, double, double, double);

SMOOTH_STEP_DEF(float, double, SMOOTH_STEP_IMPL_D);
SMOOTH_STEP_DEF(double, float, SMOOTH_STEP_IMPL_D);

_CLC_V_S_S_V_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, __spirv_ocl_smoothstep, float, float, double);
_CLC_V_S_S_V_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __spirv_ocl_smoothstep, double, double, float);

#endif
