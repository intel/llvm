//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

_CLC_DEF _CLC_OVERLOAD uint __spirv_SubgroupId() {
  size_t id_x = __spirv_LocalInvocationId_x();
  size_t id_y = __spirv_LocalInvocationId_y();
  size_t id_z = __spirv_LocalInvocationId_z();
  size_t size_x = __spirv_WorkgroupSize_x();
  size_t size_y = __spirv_WorkgroupSize_y();
  size_t size_z = __spirv_WorkgroupSize_z();
  uint sg_size = __spirv_SubgroupMaxSize();
  return (id_z * size_y * size_x + id_y * size_x + id_x) / sg_size;
}
