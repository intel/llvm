//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

uint __clc_r600_get_num_groups_x(void) __asm("llvm.r600.read.ngroups.x");
uint __clc_r600_get_num_groups_y(void) __asm("llvm.r600.read.ngroups.y");
uint __clc_r600_get_num_groups_z(void) __asm("llvm.r600.read.ngroups.z");

_CLC_DEF _CLC_OVERLOAD size_t __spirv_NumWorkgroups_x() {
    return __clc_r600_get_num_groups_x();
}

_CLC_DEF _CLC_OVERLOAD size_t __spirv_NumWorkgroups_y() {
    return __clc_r600_get_num_groups_y();
}

_CLC_DEF _CLC_OVERLOAD size_t __spirv_NumWorkgroups_z() {
    return __clc_r600_get_num_groups_z();
}
