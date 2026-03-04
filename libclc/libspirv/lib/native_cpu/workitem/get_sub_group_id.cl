//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/workitem/clc_get_sub_group_id.h>
#include <libspirv/spirv.h>

uint __mux_get_sub_group_id();

_CLC_OVERLOAD _CLC_DEF uint __spirv_BuiltInSubgroupId() {
  return __mux_get_sub_group_id();
}
