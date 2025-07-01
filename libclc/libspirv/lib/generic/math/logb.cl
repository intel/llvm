//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#include <clc/math/math.h>

#define FUNCTION __spirv_ocl_logb

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_logb(float x) {
    int ax = __clc_as_int(x) & EXSIGNBIT_SP32;
    float s = -118 - __spirv_ocl_clz(ax);
    float r = (ax >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;
    r = ax >= PINFBITPATT_SP32 ? __clc_as_float(ax) : r;
    r = ax < 0x00800000 ? s : r;
    r = ax == 0 ? __clc_as_float(NINFBITPATT_SP32) : r;
    return r;
}

#define __FLOAT_ONLY
#define __IMPL_FUNCTION __spirv_ocl_logb
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __IMPL_FUNCTION
#undef __FLOAT_ONLY

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double __spirv_ocl_logb(double x) {
    long ax = __clc_as_long(x) & EXSIGNBIT_DP64;
    double s = -1011L - __spirv_ocl_clz(ax);
    double r = (int) (ax >> EXPSHIFTBITS_DP64) - EXPBIAS_DP64;
    r = ax >= PINFBITPATT_DP64 ? __clc_as_double(ax) : r;
    r = ax < 0x0010000000000000L ? s : r;
    r = ax == 0L ? __clc_as_double(NINFBITPATT_DP64) : r;
    return r;
}

#define __DOUBLE_ONLY
#define __IMPL_FUNCTION __spirv_ocl_logb
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __IMPL_FUNCTION
#undef __DOUBLE_ONLY

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define __CLC_MIN_VECSIZE 1
#define __HALF_ONLY
#define __IMPL_FUNCTION __builtin_logbf
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __IMPL_FUNCTION
#undef __HALF_ONLY
#undef __CLC_MIN_VECSIZE

#endif

#undef FUNCTION
