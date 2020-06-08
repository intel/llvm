//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>
#include "../../include/libdevice.h"
#include "../../../generic/lib/clcmacro.h"

#define __CLC_FUNCTION __spirv_ocl_cos
#define __CLC_BUILTIN __nv_cos
#include "unary_builtin.inc"

#define __CLC_FUNCTION __spirv_ocl_cosh
#define __CLC_BUILTIN __nv_cosh
#include "unary_builtin.inc"

/*
Linking globals named '_Z17__spirv_ocl_cospif': symbol multiply defined!
#define __CLC_FUNCTION __spirv_ocl_cospi
#define __CLC_BUILTIN __nv_cospi
#include "unary_builtin.inc"
*/