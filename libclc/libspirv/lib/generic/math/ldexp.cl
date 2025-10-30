//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#include <clc/math/clc_subnormal_config.h>
#include <clc/math/math.h>
#include <clc/math/clc_ldexp.h>

#define __CLC_FUNCTION __spirv_ocl_ldexp
#define __CLC_IMPL_FUNCTION __clc_ldexp

#define __CLC_ARG2_TYPE int
#define __CLC_BODY <clc/shared/binary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __CLC_ARG2_TYPE

#define __CLC_ARG2_TYPE uint
#define __CLC_BODY <clc/shared/binary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __CLC_ARG2_TYPE
