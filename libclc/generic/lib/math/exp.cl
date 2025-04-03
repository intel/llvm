//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>
<<<<<<< HEAD
#include <clc/clcmacro.h>
#include <clc/math/math.h>
#include <libspirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF float exp(float x) {
    return __spirv_ocl_exp(x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, exp, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double exp(double x) {
    return __spirv_ocl_exp(x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, exp, double)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEFINE_UNARY_BUILTIN_FP16(exp)

#endif
=======
#include <clc/math/clc_exp.h>

#define FUNCTION exp
#define __CLC_BODY <clc/shared/unary_def.inc>
#include <clc/math/gentype.inc>
>>>>>>> f14ff59da7f98a405999bcc8481b20446de0d0cd
