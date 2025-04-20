//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <clc/math/math.h>
#include <libspirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF float exp2(float x) {
    return __spirv_ocl_exp2(x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, exp2, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double exp2(double x) {
    return __spirv_ocl_exp2(x);
}


_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, exp2, double)

#endif
