//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/workitem/clc_get_global_offset.h>
#include <libspirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF size_t __spirv_BuiltInGlobalOffset(int dim) {
  return __clc_get_global_offset(dim);
}
