//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/ptx-nvidiacl/libdevice.h>
#include <libspirv/spirv.h>

// Expose "bfloat16 versions" of fabs.

// FIXME: __clc symbols are internal and should not be made publicly available.
// The correct thing to do would be to expose bfloat16/bfloat16x2 versions of
// these builtins as proper __spirv_ocl_fabs builtins. An LLVM demangling bug is
// currently preventing us from using these types natively in libclc (the
// libclc-demangler fails)

// Requires at least sm_80
_CLC_DEF _CLC_OVERLOAD ushort __clc_fabs(ushort x) {
    ushort res;
    __asm__("abs.bf16 %0, %1;" : "=h"(res) : "h"(x));
    return res;
}

// Requires at least sm_80
_CLC_DEF _CLC_OVERLOAD uint __clc_fabs(uint x) {
    uint res;
    __asm__("abs.bf16x2 %0, %1;" : "=r"(res) : "r"(x));
    return res;
}

#define __CLC_FUNCTION __clc_fabs
#define __CLC_SCALAR

#define __CLC_GENTYPE ushort
#include <clc/shared/unary_def_scalarize.inc>
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uint
#include <clc/shared/unary_def_scalarize.inc>
#undef __CLC_GENTYPE

#undef __CLC_SCALAR
#undef FUNCTION
