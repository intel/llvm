//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/ptx-nvidiacl/libdevice.h>
#include <libspirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF int __spirv_ocl_s_mul24(int x, int y) {
  return __nv_mul24(x, y);
}

_CLC_OVERLOAD _CLC_DEF uint __spirv_ocl_u_mul24(uint x, uint y) {
  return __nv_umul24(x, y);
}

#define __CLC_SCALAR

#define __CLC_FUNCTION __spirv_ocl_s_mul24
#define __CLC_GENTYPE int
#include <clc/shared/binary_def_scalarize.inc>
#undef __CLC_GENTYPE
#undef __CLC_FUNCTION

#define __CLC_FUNCTION __spirv_ocl_u_mul24
#define __CLC_GENTYPE uint
#include <clc/shared/binary_def_scalarize.inc>
#undef __CLC_GENTYPE
#undef __CLC_FUNCTION

#undef __CLC_SCALAR
