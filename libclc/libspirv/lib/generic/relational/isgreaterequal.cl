//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/clc_convert.h"
#include "clc/relational/clc_isgreaterequal.h"
#include "clc/relational/clc_isunordered.h"

#define __CLC_FUNCTION __spirv_FOrdGreaterThanEqual
#define __CLC_IMPL_FUNCTION(x) __clc_isgreaterequal
#define __CLC_BODY "relational_binary_def.inc"
#include "clc/math/gentype.inc"

#undef __CLC_FUNCTION
#undef __CLC_IMPL_FUNCTION

#define __CLC_FUNCTION __spirv_FUnordGreaterThanEqual
#define __CLC_IMPL_FUNCTION(x) __CLC_REL_OP_BODY
#define __CLC_REL_OP_BODY(x, y)                                                \
  ((__clc_isunordered(x, y)) || (__clc_isgreaterequal(x, y)))
#define __CLC_BODY "relational_binary_def.inc"
#include "clc/math/gentype.inc"
