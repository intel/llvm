//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>
#include "config.h"
#include "math/clc_ldexp.h"
#include "../../lib/clcmacro.h"
#include "../../lib/math/math.h"

_CLC_DEFINE_BINARY_BUILTIN(float, __spirv_ocl_ldexp, __clc_ldexp, float, int)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEFINE_BINARY_BUILTIN(double, __spirv_ocl_ldexp, __clc_ldexp, double, int)
#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEFINE_BINARY_BUILTIN(half, __spirv_ocl_ldexp, __clc_ldexp, half, int)
#endif

// This defines all the ldexp(GENTYPE, int) variants
#define __CLC_BODY <ldexp.inc>
#include <clc/math/gentype.inc>
