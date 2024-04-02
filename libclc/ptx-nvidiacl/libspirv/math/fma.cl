//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

#include "../../include/libdevice.h"
#include <clcmacro.h>

extern int __clc_nvvm_reflect_arch();

_CLC_DEFINE_TERNARY_BUILTIN(float, __spirv_ocl_fma, __nv_fmaf, float, float,
                            float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEFINE_TERNARY_BUILTIN(double, __spirv_ocl_fma, __nv_fma, double, double,
                            double)

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
_CLC_TERNARY_VECTORIZE_HAVE2(_CLC_OVERLOAD _CLC_DEF, half, __spirv_ocl_fma,
                             half, half, half)

#endif

// Requires at least sm_80
_CLC_DEF _CLC_OVERLOAD ushort __clc_fma(ushort x, ushort y, ushort z) {
    ushort res;
    __asm__("fma.rn.bf16 %0, %1, %2, %3;" : "=h"(res) : "h"(x), "h"(y), "h"(z));
    return res;
}
_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ushort, __clc_fma, ushort,
                       ushort, ushort)

// Requires at least sm_80
_CLC_DEF _CLC_OVERLOAD uint __clc_fma(uint x, uint y, uint z) {
    uint res;
    __asm__("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(res) : "r"(x), "r"(y), "r"(z));
    return res;
}
_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uint, __clc_fma, uint,
                       uint, uint)

#undef __CLC_BUILTIN
#undef __CLC_BUILTIN_F
#undef __CLC_FUNCTION
