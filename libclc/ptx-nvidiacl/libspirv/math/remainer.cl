//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

#include "../../include/libdevice.h"

#define __CLC_FUNCTION __spirv_ocl_remainder
#define __CLC_BUILTIN __nv_remainder
#define __CLC_BUILTIN_F __CLC_XCONCAT(__CLC_BUILTIN, f)
#include <math/binary_builtin.inc>
