//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/math/clc_floor.h>

#undef __CLC_FUNCTION
#define __CLC_BUILTIN __clc_floor
#define __CLC_FUNCTION __spirv_ocl_floor
#include <clc/math/unary_builtin.inc>
