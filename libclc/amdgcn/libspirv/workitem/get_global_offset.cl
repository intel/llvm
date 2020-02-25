//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

#if __clang_major__ >= 8
#define CONST_AS __constant
#elif __clang_major__ >= 7
#define CONST_AS __attribute__((address_space(4)))
#else
#define CONST_AS __attribute__((address_space(2)))
#endif

_CLC_DEF _CLC_OVERLOAD size_t __spirv_GlobalOffset_x() {
    CONST_AS uint * ptr =
        (CONST_AS uint *) __builtin_amdgcn_implicitarg_ptr();
    return ptr[1];
}

_CLC_DEF _CLC_OVERLOAD size_t __spirv_GlobalOffset_y() {
    CONST_AS uint * ptr =
        (CONST_AS uint *) __builtin_amdgcn_implicitarg_ptr();
    return ptr[2];
}

_CLC_DEF _CLC_OVERLOAD size_t __spirv_GlobalOffset_z() {
    CONST_AS uint * ptr =
        (CONST_AS uint *) __builtin_amdgcn_implicitarg_ptr();
    return ptr[3];
}
