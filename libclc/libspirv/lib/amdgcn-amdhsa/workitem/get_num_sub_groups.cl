//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

_CLC_DEF _CLC_OVERLOAD uint __spirv_BuiltInNumSubgroups() {
  size_t size_x = __spirv_BuiltInWorkgroupSize(0);
  size_t size_y = __spirv_BuiltInWorkgroupSize(1);
  size_t size_z = __spirv_BuiltInWorkgroupSize(2);
  uint sg_size = __spirv_BuiltInSubgroupMaxSize();
  size_t linear_size = size_z * size_y * size_x;
  return (uint)((linear_size + sg_size - 1) / sg_size);
}
