//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/math/math.h>
#include <clc/utils.h>
#include <libspirv/ptx-nvidiacl/libdevice.h>
#include <libspirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF int __spirv_ocl_ilogb(float x) { return __nv_ilogbf(x); }

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
_CLC_OVERLOAD _CLC_DEF int __spirv_ocl_ilogb(double x) { return __nv_ilogb(x); }
#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
_CLC_OVERLOAD _CLC_DEF int __spirv_ocl_ilogb(half x) {
  float t = x;
  return __spirv_ocl_ilogb(t);
}
#endif

#define __CLC_FUNCTION __spirv_ocl_ilogb
#define __CLC_RET_TYPE int
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
