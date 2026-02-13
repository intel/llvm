//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/utils.h>
#include <libspirv/ptx-nvidiacl/libdevice.h>
#include <libspirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_nextafter(float x, float y) {
  return __nv_nextafterf(x, y);
}

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
_CLC_OVERLOAD _CLC_DEF double __spirv_ocl_nextafter(double x, double y) {
  return __nv_nextafter(x, y);
}
#endif

#include "half_nextafter.inc"

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
_CLC_OVERLOAD _CLC_DEF half __spirv_ocl_nextafter(half x, half y) {
  return half_nextafter(x, y);
}
#endif

#define __CLC_FUNCTION __spirv_ocl_nextafter
#define __CLC_BODY <clc/shared/binary_def_scalarize.inc>
#include <clc/math/gentype.inc>
