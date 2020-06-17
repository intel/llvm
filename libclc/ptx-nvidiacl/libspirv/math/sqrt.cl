//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../../../generic/lib/clcmacro.h"
#include "../../include/libdevice.h"
#include <spirv/spirv.h>

#define __CLC_FUNCTION __spirv_ocl_sqrt
#define __CLC_BUILTIN __nv_sqrt
#include "unary_builtin.inc"