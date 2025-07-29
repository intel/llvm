//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

_CLC_DEF _CLC_OVERLOAD size_t __spirv_GlobalSize_x() {
  return __spirv_NumWorkgroups_x() * __spirv_WorkgroupSize_x();
}

_CLC_DEF _CLC_OVERLOAD size_t __spirv_GlobalSize_y() {
  return __spirv_NumWorkgroups_y() * __spirv_WorkgroupSize_y();
}

_CLC_DEF _CLC_OVERLOAD size_t __spirv_GlobalSize_z() {
  return __spirv_NumWorkgroups_z() * __spirv_WorkgroupSize_z();
}
