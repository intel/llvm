//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/math/clc_ilogb.h>
#include <libspirv/spirv.h>

#define __CLC_FUNCTION __spirv_ocl_ilogb
#define __CLC_IMPL_FUNCTION(x) __clc_ilogb
#define __CLC_BODY <clc/math/unary_def_with_int_return.inc>
#include <clc/math/gentype.inc>
