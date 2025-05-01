//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/utils.h>

#ifndef __CLC_SW_FUNC
#define __CLC_SW_FUNC __CLC_XCONCAT(__clc_, __CLC_FUNC)
#endif

#if __CLC_FPSIZE > 16
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNC(__CLC_GENTYPE x) {
  return __CLC_SW_FUNC(x);
}
#elif __CLC_FPSIZE == 16
#ifdef __CLC_SCALAR
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNC(__CLC_GENTYPE x) {
  return convert_half(__CLC_SW_FUNC(convert_float(x)));
}
#else
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNC(__CLC_GENTYPE x) {
  return __CLC_XCONCAT(convert_half, __CLC_VECSIZE)(
      __CLC_SW_FUNC(__CLC_XCONCAT(convert_float, __CLC_VECSIZE)(x)));
}
#endif
#endif
