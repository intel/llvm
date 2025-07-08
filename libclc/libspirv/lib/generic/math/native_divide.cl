//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/math/clc_native_divide.h>
#include <libspirv/spirv.h>

#define __FLOAT_ONLY
#define FUNCTION __spirv_ocl_native_divide
#define __CLC_FUNCTION(x) __clc_native_divide
#define __CLC_BODY <clc/shared/binary_def.inc>
#include <clc/math/gentype.inc>
