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

#define __CLC_FUNCTION __spirv_ocl_exp2
#define __CLC_BUILTIN __nv_exp2
#define __CLC_BUILTIN_F __CLC_XCONCAT(__CLC_BUILTIN, f)
#include <math/unary_builtin.inc>

_CLC_DEF _CLC_OVERLOAD ushort __spirv_ocl_exp2(ushort x) {
  if (__clc_nvvm_reflect_arch() >= 800) {
    return __nvvm_ex2_approx_bf16(x);
  }
  __builtin_trap();
  __builtin_unreachable();
}
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ushort, __spirv_ocl_exp2, ushort)

_CLC_DEF _CLC_OVERLOAD uint __spirv_ocl_exp2(uint x) {
  if (__clc_nvvm_reflect_arch() >= 800) {
    return __nvvm_ex2_approx_bf16x2(x);
  }
  __builtin_trap();
  __builtin_unreachable();
}
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uint, __spirv_ocl_exp2, uint)
