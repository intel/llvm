//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

// Compiler support is required to provide global offset on NVPTX.

_CLC_DEF _CLC_OVERLOAD size_t __spirv_GlobalOffset_x() {
  return __builtin_ptx_implicit_offset()[0];
}

_CLC_DEF _CLC_OVERLOAD size_t __spirv_GlobalOffset_y() {
  return __builtin_ptx_implicit_offset()[1];
}

_CLC_DEF _CLC_OVERLOAD size_t __spirv_GlobalOffset_z() {
  return __builtin_ptx_implicit_offset()[2];
}
