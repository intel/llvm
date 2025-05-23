//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// We need to use shifts here in order to mantain the sign bit for signed
// integers.  The compiler should optimize this to (x & 0x00FFFFFF) for
// unsigned integers.
#define CONVERT_TO_24BIT(x) (((x) << 8) >> 8)

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_mul24(__CLC_GENTYPE x,
                                                 __CLC_GENTYPE y) {
  return CONVERT_TO_24BIT(x) * CONVERT_TO_24BIT(y);
}
