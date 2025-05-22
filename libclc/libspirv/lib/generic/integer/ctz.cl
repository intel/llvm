//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/integer/clc_ctz.h>
#include <libspirv/spirv.h>

#define FUNCTION __spirv_ocl_ctz
#define __CLC_FUNCTION(x) __clc_ctz
#define __CLC_BODY <clc/shared/unary_def.inc>

#include <clc/integer/gentype.inc>
