//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#define _USE_MATH_DEFINES

#include <clcmacro.h>
#include <math/math.h>
#include <spirv/spirv.h>

int __ocml_scalbln_f64(double, int);
int __ocml_scalbln_f32(float, int);

//_CLC_DEFINE_BINARY_BUILTIN(int, __spirv_ocl_scalbln, __ocml_scalbln_f32, float, int)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEFINE_BINARY_BUILTIN(int, __spirv_ocl_scalbln, __ocml_scalbln_f64, double, int)
#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF int __spirv_ocl_scalbln(float x, int y) {
    union {
        float f;
        unsigned int i;
    } u;
    u.f = x;

    int exponent = ((u.i >> 23) & 0xFF) - 127;

    if (exponent == -127 && (u.i & 0x7FFFFF) == 0) {
        return 0;
    } else if (exponent == 128) {
        return (int)x;
    }

    exponent += y;

    if (exponent > 127) {
        return (int)(u.i & 0x80000000 ? -INFINITY : INFINITY);
    } else if (exponent < -126) {
        return 0;
    }

    u.i = (u.i & 0x807FFFFF) | ((exponent + 127) << 23);

    return (int)u.f;
}

_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, int, __spirv_ocl_scalbln, float, int);
#endif

#undef __CLC_BUILTIN
#undef __CLC_BUILTIN_F
#undef __CLC_FUNCTION