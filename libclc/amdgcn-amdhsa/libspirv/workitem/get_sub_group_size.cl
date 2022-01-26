//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

_CLC_DEF _CLC_OVERLOAD uint __spirv_SubgroupSize() {
  if (__spirv_SubgroupId() != __spirv_NumSubgroups() - 1) {
    return __spirv_SubgroupMaxSize();
  }
  size_t size_x = __spirv_WorkgroupSize_x();
  size_t size_y = __spirv_WorkgroupSize_y();
  size_t size_z = __spirv_WorkgroupSize_z();
  size_t linear_size = size_z * size_y * size_x;
  size_t uniform_groups = __spirv_NumSubgroups() - 1;
  size_t uniform_size = __spirv_SubgroupMaxSize() * uniform_groups;
  return linear_size - uniform_size;
}
