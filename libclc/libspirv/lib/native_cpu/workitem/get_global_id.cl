//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

ulong __mux_get_global_id(int);

_CLC_DEF _CLC_OVERLOAD size_t __spirv_GlobalInvocationId_x() {
  return __mux_get_global_id(0);
}

_CLC_DEF _CLC_OVERLOAD size_t __spirv_GlobalInvocationId_y() {
  return __mux_get_global_id(1);
}

_CLC_DEF _CLC_OVERLOAD size_t __spirv_GlobalInvocationId_z() {
  return __mux_get_global_id(2);
}
