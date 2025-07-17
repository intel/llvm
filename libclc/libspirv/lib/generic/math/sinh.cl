//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>
#include <clc/math/clc_sinh.h>

#define FUNCTION __spirv_ocl_sinh
#define __CLC_FUNCTION(x) __clc_sinh
#define __CLC_BODY <clc/shared/unary_def.inc>
#include <clc/math/gentype.inc>
