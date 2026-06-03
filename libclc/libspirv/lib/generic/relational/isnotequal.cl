//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/clc_convert.h"
#include "clc/relational/clc_isnotequal.h"
#include "clc/relational/clc_isordered.h"

#define __CLC_FUNCTION __spirv_FUnordNotEqual
#define __CLC_IMPL_FUNCTION(x) __clc_isnotequal
#define __CLC_BODY "relational_binary_def.inc"
#include "clc/math/gentype.inc"

#undef __CLC_FUNCTION
#undef __CLC_IMPL_FUNCTION

#define __CLC_FUNCTION __spirv_FOrdNotEqual
#define __CLC_IMPL_FUNCTION(x) __CLC_REL_OP_BODY
#define __CLC_REL_OP_BODY(x, y)                                                \
  ((__clc_isordered(x, y)) && (__clc_isnotequal(x, y)))
#define __CLC_BODY "relational_binary_def.inc"
#include "clc/math/gentype.inc"
