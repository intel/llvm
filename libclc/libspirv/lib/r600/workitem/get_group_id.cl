//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

_CLC_DEF _CLC_OVERLOAD uint __spirv_WorkgroupId_x() {
    return __builtin_r600_read_tgid_x();
}

_CLC_DEF _CLC_OVERLOAD uint __spirv_WorkgroupId_y() {
    return __builtin_r600_read_tgid_y();
}

_CLC_DEF _CLC_OVERLOAD uint __spirv_WorkgroupId_z() {
    return __builtin_r600_read_tgid_z();
}
