//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc_as_type.h>
#include <clc/math/clc_ldexp.h>
#include <libspirv/spirv.h>

#define __CLC_FUNCTION __spirv_ocl_ldexp
#define __CLC_IMPL_FUNCTION(x) __clc_ldexp

#define __CLC_BODY <clc/shared/binary_def_with_int_second_arg.inc>
#include <clc/math/gentype.inc>

#define __CLC_BODY <binary_def_with_uint_second_arg.inc>
#include <clc/math/gentype.inc>
