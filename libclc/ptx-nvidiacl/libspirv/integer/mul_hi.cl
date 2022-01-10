//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clcmacro.h>
#include <core/clc_core.h>
#include <spirv/spirv.h>

#include "../../include/intrinsics.h"

_CLC_OVERLOAD _CLC_DEF int __spirv_ocl_s_mul_hi(int x, int y) {
  return __nvvm_mulhi_i(x, y);
}

_CLC_OVERLOAD _CLC_DEF uint __spirv_ocl_u_mul_hi(uint x, uint y) {
  return __nvvm_mulhi_ui(x, y);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_ocl_s_mul_hi(long x, long y) {
  return __clc_nvvm_mulhi(x, y);
}

_CLC_OVERLOAD _CLC_DEF ulong __spirv_ocl_u_mul_hi(ulong x, ulong y) {
  return __clc_nvvm_mulhi(x, y);
}

#define __CLC_MUL_HI_IMPL(BGENTYPE, SPV_MUL_HI, GENTYPE, GENSIZE)              \
  _CLC_OVERLOAD _CLC_DEF GENTYPE SPV_MUL_HI(GENTYPE x, GENTYPE y) {            \
    return (GENTYPE)SPV_MUL_HI((BGENTYPE)(((BGENTYPE)x) << GENSIZE),           \
                               (BGENTYPE)y);                                   \
  }

__CLC_MUL_HI_IMPL(short, __spirv_ocl_s_mul_hi, char, 8)
__CLC_MUL_HI_IMPL(short, __spirv_ocl_s_mul_hi, schar, 8)
__CLC_MUL_HI_IMPL(ushort, __spirv_ocl_u_mul_hi, uchar, 8)
__CLC_MUL_HI_IMPL(int, __spirv_ocl_s_mul_hi, short, 16)
__CLC_MUL_HI_IMPL(uint, __spirv_ocl_u_mul_hi, ushort, 16)

_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, char, __spirv_ocl_s_mul_hi, char,
                      char)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, schar, __spirv_ocl_s_mul_hi,
                      schar, schar)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, short, __spirv_ocl_s_mul_hi,
                      short, short)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, int, __spirv_ocl_s_mul_hi, int,
                      int)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, long, __spirv_ocl_s_mul_hi, long,
                      long)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uchar, __spirv_ocl_u_mul_hi,
                      uchar, uchar)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ushort, __spirv_ocl_u_mul_hi,
                      ushort, ushort)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uint, __spirv_ocl_u_mul_hi, uint,
                      uint)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ulong, __spirv_ocl_u_mul_hi,
                      ulong, ulong)
