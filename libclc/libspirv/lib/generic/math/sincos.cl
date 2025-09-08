//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/math/clc_sincos.h>
#include <libspirv/spirv.h>

#define FUNCTION __spirv_ocl_sincos
#define __IMPL_FUNCTION(x) __clc_sincos
#define __CLC_BODY <clc/math/unary_def_with_ptr.inc>
#include <clc/math/gentype.inc>
