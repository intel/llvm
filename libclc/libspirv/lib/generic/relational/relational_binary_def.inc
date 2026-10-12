//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/utils.h"

#if __CLC_VECSIZE_OR_1 == 1
#define __CLC_RETTYPE bool
#else
#define __CLC_RETTYPE __CLC_CHARN
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_RETTYPE __CLC_FUNCTION(__CLC_GENTYPE x,
                                                    __CLC_GENTYPE y) {
  return __CLC_CONVERT_CHARN(__CLC_IMPL_FUNCTION(__CLC_FUNCTION)(x, y));
}

#undef __CLC_RETTYPE
