//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#undef __spirv_IsNormal

#define __SPIRV_FUNCTION __spirv_IsNormal
#define __SPIRV_BODY <spirv/relational/unary_decl.inc>

#include <spirv/relational/floatn.inc>

#undef __SPIRV_BODY
#undef __SPIRV_FUNCTION
