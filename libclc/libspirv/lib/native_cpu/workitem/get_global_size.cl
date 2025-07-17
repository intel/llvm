//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

ulong __mux_get_global_size(int);

_CLC_DEF _CLC_OVERLOAD size_t __spirv_GlobalSize_x() {
  return __mux_get_global_size(0);
}

_CLC_DEF _CLC_OVERLOAD size_t __spirv_GlobalSize_y() {
  return __mux_get_global_size(1);
}

_CLC_DEF _CLC_OVERLOAD size_t __spirv_GlobalSize_z() {
  return __mux_get_global_size(2);
}
