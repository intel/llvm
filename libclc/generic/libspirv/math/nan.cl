//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>
#include <clc/utils.h>

#define __CLC_AS_GENTYPE __CLC_XCONCAT(as_, __CLC_GENTYPE)
#define __CLC_AS_UNSIGNED(TYPE)                                                \
  __CLC_XCONCAT(as_, __CLC_XCONCAT(TYPE, __CLC_VECSIZE))
#define __CLC_BODY <nan.inc>
#include <clc/math/gentype.inc>
