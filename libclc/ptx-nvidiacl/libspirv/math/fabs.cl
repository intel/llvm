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

#define __CLC_FUNCTION __spirv_ocl_fabs
#define __CLC_BUILTIN __nv_fabs
#define __CLC_BUILTIN_F __CLC_XCONCAT(__CLC_BUILTIN, f)
#include <math/unary_builtin.inc>

// Requires at least sm_80
_CLC_DEF _CLC_OVERLOAD ushort __clc_fabs(ushort x) {
    return __nvvm_abs_bf16(x);
}
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ushort, __clc_fabs, ushort)

// Requires at least sm_80
_CLC_DEF _CLC_OVERLOAD uint __clc_fabs(uint x) {
    return __nvvm_abs_bf16x2(x);
}
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uint, __clc_fabs, uint)
