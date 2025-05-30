//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>
#include <libspirv/spirv.h>


_CLC_DEF _CLC_OVERLOAD size_t get_global_id(uint dim) {
switch (dim) {
    case 0:  return __spirv_GlobalInvocationId_x();
    case 1:  return __spirv_GlobalInvocationId_y();
    case 2:  return __spirv_GlobalInvocationId_z();
    default: return 0;
  }
}
