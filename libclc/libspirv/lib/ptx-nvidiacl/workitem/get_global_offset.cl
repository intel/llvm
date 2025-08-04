//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

// Compiler support is required to provide global offset on NVPTX.

_CLC_DEF _CLC_OVERLOAD size_t __spirv_BuiltInGlobalOffset(int dim) {
  switch (dim) {
  case 0:
    return __builtin_ptx_implicit_offset()[0];
  case 1:
    return __builtin_ptx_implicit_offset()[1];
  case 2:
    return __builtin_ptx_implicit_offset()[2];
  default:
    return 0;
  }
}
