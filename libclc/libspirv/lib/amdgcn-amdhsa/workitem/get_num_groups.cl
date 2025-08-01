//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

_CLC_DEF _CLC_OVERLOAD size_t __spirv_BuiltInNumWorkgroups(int dim) {
  size_t global_size = __spirv_BuiltInGlobalSize(dim);
  size_t local_size = __spirv_BuiltInWorkgroupSize(dim);
  size_t num_groups = global_size / local_size;
  if (global_size % local_size != 0)
    num_groups++;
  return num_groups;
}
