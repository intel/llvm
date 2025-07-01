//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#define FUNCTION __spirv_ocl_log

/*
 *log(x) = log2(x) * (1/log2(e))
 */

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_log(float x)
{
    return __spirv_ocl_log2(x) * (1.0f / M_LOG2E_F);
}

#define __FLOAT_ONLY
#define __IMPL_FUNCTION __spirv_ocl_log
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __IMPL_FUNCTION
#undef __FLOAT_ONLY

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double __spirv_ocl_log(double x)
{
    return __spirv_ocl_log2(x) * (1.0 / M_LOG2E);
}

#define __DOUBLE_ONLY
#define __IMPL_FUNCTION __spirv_ocl_log
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __IMPL_FUNCTION
#undef __DOUBLE_ONLY

#endif // cl_khr_fp64

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define __CLC_MIN_VECSIZE 1
#define __HALF_ONLY
#define __IMPL_FUNCTION __builtin_logf16
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __IMPL_FUNCTION
#undef __HALF_ONLY
#undef __CLC_MIN_VECSIZE

#endif

#undef FUNCTION
