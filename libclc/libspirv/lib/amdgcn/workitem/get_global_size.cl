//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

uint __clc_amdgcn_get_global_size_x(void) __asm("llvm.r600.read.global.size.x");
uint __clc_amdgcn_get_global_size_y(void) __asm("llvm.r600.read.global.size.y");
uint __clc_amdgcn_get_global_size_z(void) __asm("llvm.r600.read.global.size.z");

_CLC_DEF _CLC_OVERLOAD size_t __spirv_GlobalSize_x() {
    return __clc_amdgcn_get_global_size_x();
}

_CLC_DEF _CLC_OVERLOAD size_t __spirv_GlobalSize_y() {
    return __clc_amdgcn_get_global_size_y();
}

_CLC_DEF _CLC_OVERLOAD size_t __spirv_GlobalSize_z() {
    return __clc_amdgcn_get_global_size_z();
}
