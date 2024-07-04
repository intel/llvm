//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clcmacro.h>
#include <spirv/spirv.h>

#include "../../include/libdevice.h"
#include "utils.h"
#include <math/math.h>

/// Define lround for float using __spirv_ocl_rint and casting the result to long
_CLC_OVERLOAD _CLC_DEF long __spirv_ocl_lround(float x) {
    return (long)__spirv_ocl_rint(x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, long, __spirv_ocl_lround, float);

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Define lround for double using __spirv_ocl_rint and casting the result to long
_CLC_OVERLOAD _CLC_DEF long __spirv_ocl_lround(double x) {
    return (long)__spirv_ocl_rint(x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, long, __spirv_ocl_lround, double);
#endif // cl_khr_fp64

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Define lround for half using __spirv_ocl_rint and casting the result to long
_CLC_OVERLOAD _CLC_DEF long __spirv_ocl_lround(half x) {
    float t = x;
    return (long)__spirv_ocl_rint(t);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, long, __spirv_ocl_lround, half);
#endif // cl_khr_fp16

#undef __CLC_BUILTIN
#undef __CLC_BUILTIN_F
#undef __CLC_FUNCTION