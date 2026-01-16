//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF size_t __spirv_BuiltInGlobalOffset(int dim) {
  constant uint *ptr = __builtin_amdgcn_implicit_offset();
  switch (dim) {
  case 0:
    return ptr[0];
  case 1:
    return ptr[1];
  case 2:
    return ptr[2];
  default:
    return 0;
  }
}
