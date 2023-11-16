//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_abs_diff(__CLC_GENTYPE x,
                                                    __CLC_GENTYPE y) {
  if ((x < 0) == (y < 0))
    return (x > y) ? (x - y) : (y - x);
  __CLC_GENTYPE ax = x < 0 ? -x : x;
  __CLC_GENTYPE ay = y < 0 ? -y : y;
  return ax + ay;
}
