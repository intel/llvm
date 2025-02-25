//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>
#include <clc/common/clc_degrees.h>
#include <libspirv/spirv.h>

#define FUNCTION __spirv_ocl_degrees
#define __CLC_FUNCTION(x) __clc_degrees
#define __CLC_BODY <clc/common/unary_def.inc>
#include <clc/math/gentype.inc>
