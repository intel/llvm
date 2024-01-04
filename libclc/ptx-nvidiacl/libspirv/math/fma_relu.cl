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

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEF _CLC_OVERLOAD half __clc_fma_relu(half x, half y, half z) {
  if (__clc_nvvm_reflect_arch() >= 800) {
    return __nvvm_fma_rn_relu_f16(x, y, z);
  }
  __builtin_trap();
  __builtin_unreachable();
}

_CLC_DEF _CLC_OVERLOAD half2 __clc_fma_relu(half2 x, half2 y, half2 z) {
  if (__clc_nvvm_reflect_arch() >= 800) {
    return __nvvm_fma_rn_relu_f16x2(x, y, z);
  }
  return (half2)(__clc_fma_relu(x.x, y.x, z.x),
                 __clc_fma_relu(x.y, y.y, z.y));
}
_CLC_TERNARY_VECTORIZE_HAVE2(_CLC_OVERLOAD _CLC_DEF, half, __clc_fma_relu,
                             half, half, half)

#endif

_CLC_DEF _CLC_OVERLOAD ushort __clc_fma_relu(ushort x, ushort y,
                                                   ushort z) {
  if (__clc_nvvm_reflect_arch() >= 800) {
    ushort res;
    __asm__("fma.rn.relu.bf16 %0, %1, %2, %3;"
        : "=h"(res)
        : "h"(x), "h"(y), "h"(z));
    return res;
  }
  __builtin_trap();
  __builtin_unreachable();
}
_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ushort, __clc_fma_relu,
                       ushort, ushort, ushort)

_CLC_DEF _CLC_OVERLOAD uint __clc_fma_relu(uint x, uint y, uint z) {
  if (__clc_nvvm_reflect_arch() >= 800) {
    uint res;
    __asm__("fma.rn.relu.bf16x2 %0, %1, %2, %3;"
        : "=r"(res)
        : "r"(x), "r"(y), "r"(z));
    return res;
  }
  __builtin_trap();
  __builtin_unreachable();
}
_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uint, __clc_fma_relu, uint,
                       uint, uint)
