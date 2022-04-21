//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

#include <math/clc_tanpi.h>

#define __CLC_FUNC __spirv_ocl_tanpi
#define __CLC_SW_FUNC __clc_tanpi
#define __CLC_BODY <clc_sw_unary.inc>
#include <clc/math/gentype.inc>
#undef __CLC_SW_FUNC
