//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

uint __clc_amdgcn_get_local_size_x(void) __asm("llvm.r600.read.local.size.x");
uint __clc_amdgcn_get_local_size_y(void) __asm("llvm.r600.read.local.size.y");
uint __clc_amdgcn_get_local_size_z(void) __asm("llvm.r600.read.local.size.z");

_CLC_DEF _CLC_OVERLOAD size_t __spirv_WorkgroupSize_x() {
    return __clc_amdgcn_get_local_size_x();
}

_CLC_DEF _CLC_OVERLOAD size_t __spirv_WorkgroupSize_y() {
    return __clc_amdgcn_get_local_size_y();
}

_CLC_DEF _CLC_OVERLOAD size_t __spirv_WorkgroupSize_z() {
    return __clc_amdgcn_get_local_size_z();
}
