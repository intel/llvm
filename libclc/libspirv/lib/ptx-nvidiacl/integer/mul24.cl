//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clcmacro.h>
#include <core/clc_core.h>
#include <libspirv/spirv.h>
#include <libspirv/ptx-nvidiacl/libdevice.h>

_CLC_OVERLOAD _CLC_DEF int __spirv_ocl_s_mul24(int x, int y) {
  return __nv_mul24(x, y);
}

_CLC_OVERLOAD _CLC_DEF uint __spirv_ocl_u_mul24(uint x, uint y) {
  return __nv_umul24(x, y);
}

_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, int, __spirv_ocl_s_mul24, int,
                      int)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uint, __spirv_ocl_u_mul24, uint,
                      uint)
