//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __CLC_FUNCTION __spirv_GroupBitwiseOrKHR
#define __CLC_NON_UNIFORM_FUNCTION __spirv_GroupNonUniformBitwiseOr
#define __CLC_BODY <libspirv/group/group_decl.inc>

#include <clc/integer/gentype.inc>

#undef __CLC_FUNCTION
#undef __CLC_NON_UNIFORM_FUNCTION
