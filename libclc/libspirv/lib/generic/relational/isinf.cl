//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/clc_convert.h"
#include "clc/relational/clc_isinf.h"

#define __CLC_FUNCTION __spirv_IsInf
#define __CLC_IMPL_FUNCTION(x) __clc_isinf
#define __CLC_BODY "relational_unary_def.inc"
#include "clc/math/gentype.inc"
