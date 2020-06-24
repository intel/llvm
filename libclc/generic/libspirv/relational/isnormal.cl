//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <relational.h>
#include <spirv/spirv.h>

#define _CLC_SPIRV_BUILTIN __spirv_IsNormal
#define _CLC_BUILTIN_IMPL __builtin_isnormal
#include "genunary.inc"
