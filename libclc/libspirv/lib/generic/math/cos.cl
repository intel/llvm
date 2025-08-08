//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/math/clc_cos.h>
#include <libspirv/spirv.h>

#define FUNCTION __spirv_ocl_cos
#define __IMPL_FUNCTION(x) __clc_cos
#define __CLC_BODY <clc/shared/unary_def.inc>
#include <clc/math/gentype.inc>
