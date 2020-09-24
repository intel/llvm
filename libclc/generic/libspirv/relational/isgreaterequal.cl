//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <relational.h>
#include <spirv/spirv.h>

#define _CLC_SPIRV_BUILTIN __spirv_FOrdGreaterThanEqual
#define _CLC_BUILTIN_IMPL __builtin_isgreaterequal
#include "genbinrelational.inc"
#undef _CLC_SPIRV_BUILTIN
#undef _CLC_BUILTIN_IMPL

#define _CLC_SPIRV_BUILTIN __spirv_FUnordGreaterThanEqual
#define _CLC_BUILTIN_IMPL(X, Y) X >= Y
#include "genbinrelational.inc"
