//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <clc/utils.h>

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNC(__CLC_GENTYPE x,
                                                __CLC_GENTYPE y) {
  return __CLC_SW_FUNC(x, y);
}
