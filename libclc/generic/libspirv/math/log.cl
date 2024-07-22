//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clcmacro.h>
#include <spirv/spirv.h>

/*
 *log(x) = log2(x) * (1/log2(e))
 */

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_log(float x)
{
    return __spirv_ocl_log2(x) * (1.0f / M_LOG2E_F);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __spirv_ocl_log, float);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double __spirv_ocl_log(double x)
{
    return __spirv_ocl_log2(x) * (1.0 / M_LOG2E);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, __spirv_ocl_log, double);

#endif // cl_khr_fp64

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEFINE_UNARY_BUILTIN(half, __spirv_ocl_log, __builtin_log, half)

#endif
