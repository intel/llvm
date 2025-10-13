//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

int __ocml_ilogb_f32(float);
_CLC_OVERLOAD _CLC_DEF int __spirv_ocl_ilogb(float x) {
  return __ocml_ilogb_f32(x);
}

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
int __ocml_ilogb_f64(double);
_CLC_OVERLOAD _CLC_DEF int __spirv_ocl_ilogb(double x) {
  return __ocml_ilogb_f64(x);
}
#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
int __ocml_ilogb_f16(half);
_CLC_OVERLOAD _CLC_DEF int __spirv_ocl_ilogb(half x) {
  return __ocml_ilogb_f16(x);
}
#endif

#define __CLC_FUNCTION __spirv_ocl_ilogb
#define __CLC_RET_TYPE int
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
