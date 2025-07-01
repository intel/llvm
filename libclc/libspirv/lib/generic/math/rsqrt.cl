//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_rsqrt(float x) {
  return 1.0f / __spirv_ocl_sqrt(x);
}

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double __spirv_ocl_rsqrt(double x) {
  return 1.0 / __spirv_ocl_sqrt(x);
}

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF half __spirv_ocl_rsqrt(half x) {
  return 1.0f / __spirv_ocl_sqrt((float)x);
}

#endif

#define FUNCTION __spirv_ocl_rsqrt
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
