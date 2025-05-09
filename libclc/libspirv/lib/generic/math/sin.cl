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

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_sin(float x)
{
    int ix = __clc_as_int(x);
    int ax = ix & 0x7fffffff;
    float dx = __clc_as_float(ax);

    float r0, r1;
    int regn = __clc_argReductionS(&r0, &r1, dx);

    float ss = __clc_sinf_piby4(r0, r1);
    float cc = __clc_cosf_piby4(r0, r1);

    float s = (regn & 1) != 0 ? cc : ss;
    s = __clc_as_float(__clc_as_int(s) ^ ((regn > 1) << 31) ^ (ix ^ ax));

    s = ax >= PINFBITPATT_SP32 ? __clc_as_float(QNANBITPATT_SP32) : s;

    //Subnormals
    s = x == 0.0f ? x : s;

    return s;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __spirv_ocl_sin, float);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double __spirv_ocl_sin(double x) {
    double y = __spirv_ocl_fabs(x);

    double r, rr;
    int regn;

    if (y < 0x1.0p+47)
        __clc_remainder_piby2_medium(y, &r, &rr, &regn);
    else
        __clc_remainder_piby2_large(y, &r, &rr, &regn);

    double2 sc = __clc_sincos_piby4(r, rr);

    int2 s = __clc_as_int2(regn & 1 ? sc.hi : sc.lo);
    s.hi ^= ((regn > 1) << 31) ^ ((x < 0.0) << 31);

    return __spirv_IsInf(x) || __spirv_IsNan(x) ? __clc_as_double(QNANBITPATT_DP64)
                                                : __clc_as_double(s);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, __spirv_ocl_sin, double);

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEFINE_UNARY_BUILTIN_SCALARIZE(half, __spirv_ocl_sin, __builtin_sinf16, half)

#endif
