//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

_CLC_DEF _CLC_OVERLOAD size_t __spirv_NumWorkgroups_x() {
  size_t global_size = __spirv_GlobalSize_x();
  size_t local_size = __spirv_WorkgroupSize_x();
  size_t num_groups = global_size / local_size;
  if (global_size % local_size != 0) {
    num_groups++;
  }
  return num_groups;
}

_CLC_DEF _CLC_OVERLOAD size_t __spirv_NumWorkgroups_y() {
  size_t global_size = __spirv_GlobalSize_y();
  size_t local_size = __spirv_WorkgroupSize_y();
  size_t num_groups = global_size / local_size;
  if (global_size % local_size != 0) {
    num_groups++;
  }
  return num_groups;
}

_CLC_DEF _CLC_OVERLOAD size_t __spirv_NumWorkgroups_z() {
  size_t global_size = __spirv_GlobalSize_z();
  size_t local_size = __spirv_WorkgroupSize_z();
  size_t num_groups = global_size / local_size;
  if (global_size % local_size != 0) {
    num_groups++;
  }
  return num_groups;
}
