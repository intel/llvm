//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/internal/clc.h>
#include <libspirv/ptx-nvidiacl/intrinsics.h>
#include <libspirv/spirv.h>

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
__CLC_MUL_HI_IMPL(ushort, __spirv_ocl_u_mul_hi, uchar, 8)
__CLC_MUL_HI_IMPL(int, __spirv_ocl_s_mul_hi, short, 16)
__CLC_MUL_HI_IMPL(uint, __spirv_ocl_u_mul_hi, ushort, 16)

#define __CLC_SCALAR

#define __CLC_FUNCTION __spirv_ocl_s_mul_hi
#define __CLC_GENTYPE char
#include <clc/shared/binary_def_scalarize.inc>
#undef __CLC_GENTYPE
#define __CLC_GENTYPE short
#include <clc/shared/binary_def_scalarize.inc>
#undef __CLC_GENTYPE
#define __CLC_GENTYPE int
#include <clc/shared/binary_def_scalarize.inc>
#undef __CLC_GENTYPE
#define __CLC_GENTYPE long
#include <clc/shared/binary_def_scalarize.inc>
#undef __CLC_GENTYPE
#undef __CLC_FUNCTION

#define __CLC_FUNCTION __spirv_ocl_u_mul_hi
#define __CLC_GENTYPE uchar
#include <clc/shared/binary_def_scalarize.inc>
#undef __CLC_GENTYPE
#define __CLC_GENTYPE ushort
#include <clc/shared/binary_def_scalarize.inc>
#undef __CLC_GENTYPE
#define __CLC_GENTYPE uint
#include <clc/shared/binary_def_scalarize.inc>
#undef __CLC_GENTYPE
#define __CLC_GENTYPE ulong
#include <clc/shared/binary_def_scalarize.inc>
#undef __CLC_GENTYPE
#undef __CLC_FUNCTION

#undef __CLC_SCALAR
