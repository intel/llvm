//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

uint __mux_get_sub_group_size();

_CLC_DEF _CLC_OVERLOAD uint __spirv_SubgroupSize() {
  return __mux_get_sub_group_size();
}
