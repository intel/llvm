//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#include <libspirv/ptx-nvidiacl/libdevice.h>

#define __CLC_FUNCTION __spirv_ocl_fma

extern int __clc_nvvm_reflect_arch();

#define __CLC_FLOAT_ONLY
#define __CLC_MIN_VECSIZE 1
#define __CLC_IMPL_FUNCTION __nv_fmaf
#define __CLC_BODY <clc/shared/ternary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __CLC_IMPL_FUNCTION
#undef __CLC_MIN_VECSIZE
#undef __CLC_FLOAT_ONLY

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define __CLC_DOUBLE_ONLY
#define __CLC_MIN_VECSIZE 1
#define __CLC_IMPL_FUNCTION __nv_fma
#define __CLC_BODY <clc/shared/ternary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __CLC_IMPL_FUNCTION
#undef __CLC_MIN_VECSIZE
#undef __CLC_DOUBLE_ONLY

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEF _CLC_OVERLOAD half __spirv_ocl_fma(half x, half y, half z) {
  if (__clc_nvvm_reflect_arch() >= 530) {
    return __nvvm_fma_rn_f16(x, y, z);
  }
  return __nv_fmaf(x, y, z);
}

_CLC_DEF _CLC_OVERLOAD half2 __spirv_ocl_fma(half2 x, half2 y, half2 z) {
  if (__clc_nvvm_reflect_arch() >= 530) {
    return __nvvm_fma_rn_f16x2(x, y, z);
  }
  return (half2)(__spirv_ocl_fma(x.x, y.x, z.x),
                 __spirv_ocl_fma(x.y, y.y, z.y));
}

#define __CLC_HALF_ONLY
#define __CLC_MIN_VECSIZE 3
#define __CLC_BODY <clc/shared/ternary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __CLC_IMPL_FUNCTION
#undef __CLC_MIN_VECSIZE
#undef __CLC_HALF_ONLY

#endif

#undef __CLC_FUNCTION
#define __CLC_FUNCTION __clc_fma

// Requires at least sm_80
_CLC_DEF _CLC_OVERLOAD ushort __clc_fma(ushort x, ushort y, ushort z) {
    ushort res;
    __asm__("fma.rn.bf16 %0, %1, %2, %3;" : "=h"(res) : "h"(x), "h"(y), "h"(z));
    return res;
}

#define __CLC_SCALAR

#define __CLC_GENTYPE ushort
#include <clc/shared/ternary_def_scalarize.inc>
#undef __CLC_GENTYPE

// Requires at least sm_80
_CLC_DEF _CLC_OVERLOAD uint __clc_fma(uint x, uint y, uint z) {
    uint res;
    __asm__("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(res) : "r"(x), "r"(y), "r"(z));
    return res;
}

#define __CLC_GENTYPE uint
#include <clc/shared/ternary_def_scalarize.inc>
#undef __CLC_GENTYPE

#undef __CLC_SCALAR

#undef FUNCTION
