//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clcmacro.h>
#include <spirv/spirv.h>

#define __CLC_FUNCTION __spirv_ocl_log10
#define __CLC_BUILTIN __ocml_log10

float __ocml_log10_f32(float);
#define __CLC_BUILTIN_F __CLC_XCONCAT(__CLC_BUILTIN, _f32)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
double __ocml_log10_f64(double);
#define __CLC_BUILTIN_D __CLC_XCONCAT(__CLC_BUILTIN, _f64)
#endif // cl_khr_fp64

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
half __ocml_log10_f16(half);
#define __CLC_BUILTIN_H __CLC_XCONCAT(__CLC_BUILTIN, _f16)
#endif // cl_khr_fp16

#include <math/unary_builtin.inc>
