//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#include <clc/clcmacro.h>
#include <clc/math/math.h>

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_logb(float x) {
    int ax = __clc_as_int(x) & EXSIGNBIT_SP32;
    float s = -118 - __spirv_ocl_clz(ax);
    float r = (ax >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;
    r = ax >= PINFBITPATT_SP32 ? __clc_as_float(ax) : r;
    r = ax < 0x00800000 ? s : r;
    r = ax == 0 ? __clc_as_float(NINFBITPATT_SP32) : r;
    return r;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __spirv_ocl_logb, float);

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

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, __spirv_ocl_logb, double)
#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEFINE_UNARY_BUILTIN_SCALARIZE(half, __spirv_ocl_logb, __builtin_logbf, half)

#endif
