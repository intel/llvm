//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>
#include <libspirv/ptx-nvidiacl/libdevice.h>

#define __CLC_FUNCTION __spirv_ocl_fmin

extern int __clc_nvvm_reflect_arch();

_CLC_DEF _CLC_OVERLOAD float __spirv_ocl_fmin(float x, float y) {
  return __nvvm_fmin_f(x, y);
}

#define __CLC_FLOAT_ONLY
#define __CLC_BODY <clc/shared/binary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __CLC_FLOAT_ONLY

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEF _CLC_OVERLOAD double __spirv_ocl_fmin(double x, double y) {
  return __nvvm_fmin_d(x, y);
}

#define __CLC_DOUBLE_ONLY
#define __CLC_BODY <clc/shared/binary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __CLC_DOUBLE_ONLY

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEF _CLC_OVERLOAD half __spirv_ocl_fmin(half x, half y) {
  if (__clc_nvvm_reflect_arch() >= 800) {
    return __nvvm_fmin_f16(x, y);
  }
  return __nvvm_fmin_f(x,y);
}
_CLC_DEF _CLC_OVERLOAD half2 __spirv_ocl_fmin(half2 x, half2 y) {
  if (__clc_nvvm_reflect_arch() >= 800) {
    return __nvvm_fmin_f16x2(x, y);
  }
  return (half2)(__spirv_ocl_fmin(x.x, y.x), __spirv_ocl_fmin(x.y, y.y));
}

#undef __CLC_MIN_VECSIZE
#define __CLC_MIN_VECSIZE 3
#define __CLC_HALF_ONLY
#define __CLC_BODY <clc/shared/binary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __CLC_HALF_ONLY
#undef __CLC_MIN_VECSIZE

#endif

#undef __CLC_FUNCTION
#define __CLC_FUNCTION __clc_fmin
#define __CLC_SCALAR

// Requires at least sm_80
_CLC_DEF _CLC_OVERLOAD ushort __clc_fmin(ushort x, ushort y) {
    ushort res;
    __asm__("min.bf16 %0, %1, %2;" : "=h"(res) : "h"(x), "h"(y));
    return res;
}

#define __CLC_GENTYPE ushort
#include <clc/shared/binary_def_scalarize.inc>
#undef __CLC_GENTYPE

// Requires at least sm_80
_CLC_DEF _CLC_OVERLOAD uint __clc_fmin(uint x, uint y) {
    uint res;
    __asm__("min.bf16x2 %0, %1, %2;" : "=r"(res) : "r"(x), "r"(y));
    return res;
}

#define __CLC_GENTYPE uint
#include <clc/shared/binary_def_scalarize.inc>
#undef __CLC_GENTYPE

#undef __CLC_SCALAR
#undef __CLC_FUNCTION
