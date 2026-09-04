//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

_CLC_DEF _CLC_OVERLOAD uint __spirv_BuiltInSubgroupId() {
  size_t id_x = __spirv_BuiltInLocalInvocationId(0);
  size_t id_y = __spirv_BuiltInLocalInvocationId(1);
  size_t id_z = __spirv_BuiltInLocalInvocationId(2);
  size_t size_x = __spirv_BuiltInWorkgroupSize(0);
  size_t size_y = __spirv_BuiltInWorkgroupSize(1);
  uint sg_size = __spirv_BuiltInSubgroupMaxSize();
  return (id_z * size_y * size_x + id_y * size_x + id_x) / sg_size;
}
