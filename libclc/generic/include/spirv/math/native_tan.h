//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __SPIRV_BODY <spirv/math/unary_decl.inc>
#define __SPIRV_FUNCTION __spirv_ocl_native_tan
#define __FLOAT_ONLY

#include <spirv/math/gentype.inc>

#undef __FLOAT_ONLY
#undef __SPIRV_BODY
#undef __SPIRV_FUNCTION
