//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

_CLC_DEF _CLC_OVERLOAD uint __spirv_BuiltInWorkgroupId(int dim) {
  switch (dim) {
  case 0:
    return __builtin_r600_read_tgid_x();
  case 1:
    return __builtin_r600_read_tgid_y();
  case 2:
    return __builtin_r600_read_tgid_z();
  default:
    return 0;
  }
}
