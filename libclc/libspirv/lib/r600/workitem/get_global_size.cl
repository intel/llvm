//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

uint __clc_r600_get_global_size_x(void) __asm("llvm.r600.read.global.size.x");
uint __clc_r600_get_global_size_y(void) __asm("llvm.r600.read.global.size.y");
uint __clc_r600_get_global_size_z(void) __asm("llvm.r600.read.global.size.z");

_CLC_DEF _CLC_OVERLOAD size_t __spirv_BuiltInGlobalSize(int dim) {
  switch (dim) {
  case 0:
    return __clc_r600_get_global_size_x();
  case 1:
    return __clc_r600_get_global_size_y();
  case 2:
    return __clc_r600_get_global_size_z();
  default:
    return 1;
  }
}
