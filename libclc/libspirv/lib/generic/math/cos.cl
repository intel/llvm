//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#include "sincos_helpers.h"
#include <clc/clcmacro.h>
#include <clc/math/clc_sincos_helpers.h>
#include <clc/math/math.h>

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_cos(float x)
{
    int ix = __clc_as_int(x);
    int ax = ix & 0x7fffffff;
    float dx = __clc_as_float(ax);

    float r0, r1;
    int regn = __clc_argReductionS(&r0, &r1, dx);

    float ss = -__clc_sinf_piby4(r0, r1);
    float cc =  __clc_cosf_piby4(r0, r1);

    float c =  (regn & 1) != 0 ? ss : cc;
    c = __clc_as_float(__clc_as_int(c) ^ ((regn > 1) << 31));

    c = ax >= PINFBITPATT_SP32 ? __clc_as_float(QNANBITPATT_SP32) : c;

    return c;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __spirv_ocl_cos, float);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double __spirv_ocl_cos(double x) {
    x = __spirv_ocl_fabs(x);

    double r, rr;
    int regn;

    if (x < 0x1.0p+47)
        __clc_remainder_piby2_medium(x, &r, &rr, &regn);
    else
        __clc_remainder_piby2_large(x, &r, &rr, &regn);

    double2 sc = __clc_sincos_piby4(r, rr);
    sc.lo = -sc.lo;

    int2 c = __clc_as_int2(regn & 1 ? sc.lo : sc.hi);
    c.hi ^= (regn > 1) << 31;

    return __spirv_IsNan(x) || __spirv_IsInf(x) ? __clc_as_double(QNANBITPATT_DP64)
                                                : __clc_as_double(c);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, __spirv_ocl_cos, double);

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEFINE_UNARY_BUILTIN_SCALARIZE(half, __spirv_ocl_cos, __builtin_cosf16, half)

#endif
