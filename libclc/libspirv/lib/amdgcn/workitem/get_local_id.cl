//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

_CLC_DEF _CLC_OVERLOAD size_t __spirv_LocalInvocationId_x() {
    return __builtin_amdgcn_workitem_id_x();
}

_CLC_DEF _CLC_OVERLOAD size_t __spirv_LocalInvocationId_y() {
    return __builtin_amdgcn_workitem_id_y();
}

_CLC_DEF _CLC_OVERLOAD size_t __spirv_LocalInvocationId_z() {
    return __builtin_amdgcn_workitem_id_z();
}
