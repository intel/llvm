//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/workitem/clc_get_local_id.h"

ulong __mux_get_local_id(int);

_CLC_OVERLOAD _CLC_DEF size_t __clc_get_local_id(uint dim) {
  return __mux_get_local_id(dim);
}
