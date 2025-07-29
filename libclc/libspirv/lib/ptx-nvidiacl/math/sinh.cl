//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clcmacro.h>
#include <libspirv/ptx-nvidiacl/libdevice.h>
#include <libspirv/spirv.h>

#define __CLC_FUNCTION __spirv_ocl_sinh
#define __CLC_BUILTIN __nv_sinh
#define __CLC_BUILTIN_F __CLC_XCONCAT(__CLC_BUILTIN, f)

#include <clc/math/unary_builtin_scalarize.inc>
