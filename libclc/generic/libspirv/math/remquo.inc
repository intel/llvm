//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __spirv_ocl_remquo(
    __CLC_GENTYPE x, __CLC_GENTYPE y, __CLC_ADDRESS_SPACE __CLC_INTN *q) {
  private __CLC_INTN private_q;
  __CLC_GENTYPE ret = __clc_remquo(x, y, &private_q);
  *q = private_q;
  return ret;
}
