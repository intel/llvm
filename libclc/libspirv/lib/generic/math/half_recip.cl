//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#define recip(x) (1.0f / x)

#define __CLC_BUILTIN recip
#define __CLC_FUNCTION __spirv_ocl_half_recip
#include <clc/math/unary_builtin_scalarize.inc>
