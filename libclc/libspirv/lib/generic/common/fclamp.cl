//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/shared/clc_clamp.h>
#include <libspirv/spirv.h>

#define FUNCTION __spirv_ocl_fclamp
#define __CLC_FUNCTION(x) __clc_clamp

#define __CLC_BODY <clc/shared/ternary_def.inc>
#include <clc/math/gentype.inc>
