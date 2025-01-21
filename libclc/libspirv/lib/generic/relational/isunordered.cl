//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <relational.h>
#include <libspirv/spirv.h>

#define _CLC_SPIRV_BUILTIN __spirv_Unordered
#define _CLC_BUILTIN_IMPL __builtin_isunordered
#include "genbinrelational.inc"
