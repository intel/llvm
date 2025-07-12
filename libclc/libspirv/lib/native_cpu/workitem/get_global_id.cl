//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

ulong __mux_get_global_id(int);

_CLC_DEF _CLC_OVERLOAD size_t __spirv_BuiltInGlobalInvocationId(int dim) {
  switch (dim) {
  case 0:
    return __mux_get_global_id(0);
  case 1:
    return __mux_get_global_id(1);
  case 2:
    return __mux_get_global_id(2);
  default:
    return 0;
  }
}
