//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>
#include <clcmacro.h>
#include <math/math.h>

// Define the scalbln function for float type
_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_scalbln(float x, long n) {
    return x * __spirv_ocl_exp2((float)n);
}

_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __spirv_ocl_scalbln, float, long)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double __spirv_ocl_scalbln(double x, long n) {
    return x * __spirv_ocl_exp2((double)n);
}

_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, __spirv_ocl_scalbln, double, long)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF half __spirv_ocl_scalbln(half x, long n) {
    return x * __spirv_ocl_exp2((half)n);
}

_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, __spirv_ocl_scalbln, half, long)

#endif