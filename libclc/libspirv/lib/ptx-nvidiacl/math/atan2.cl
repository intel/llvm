//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#include <libspirv/ptx-nvidiacl/libdevice.h>
#include <clc/clcmacro.h>

#define __CLC_FUNCTION __spirv_ocl_atan2
#define __CLC_BUILTIN __nv_atan2
#define __CLC_BUILTIN_F __CLC_XCONCAT(__CLC_BUILTIN, f)
#include <clc/math/binary_builtin.inc>
