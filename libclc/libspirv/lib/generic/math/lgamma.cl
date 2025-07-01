//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_lgamma(float x) {
  int s;
  return __spirv_ocl_lgamma_r(x, &s);
}

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double __spirv_ocl_lgamma(double x) {
  int s;
  return __spirv_ocl_lgamma_r(x, &s);
}

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF half __spirv_ocl_lgamma(half x) {
  int s;
  return __spirv_ocl_lgamma_r(x, &s);
}

#endif

#define FUNCTION __spirv_ocl_lgamma
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
