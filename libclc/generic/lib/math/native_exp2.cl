//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <libspirv/spirv.h>

#define __CLC_BUILTIN __spirv_ocl_native_exp2
#define __CLC_FUNCTION native_exp2
#define __CLC_BODY <native_builtin.inc>
#define __FLOAT_ONLY
#include <clc/math/gentype.inc>
