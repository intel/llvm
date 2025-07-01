//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_step(float edge, float x) {
  return x < edge ? 0.0f : 1.0f;
}

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double __spirv_ocl_step(double edge, double x) {
  return x < edge ? 0.0 : 1.0;
}

#endif // cl_khr_fp64

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF half __spirv_ocl_step(half edge, half x) {
  return x < edge ? 0.0h : 1.0h;
}

#endif // cl_khr_fp16

#define FUNCTION __spirv_ocl_step
#define __CLC_BODY <clc/shared/binary_def_scalarize.inc>
#include <clc/math/gentype.inc>
