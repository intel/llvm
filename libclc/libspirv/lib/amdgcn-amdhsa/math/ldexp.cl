//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clcmacro.h>
#include <clc/math/clc_subnormal_config.h>
#include <clc/math/math.h>
#include <libspirv/spirv.h>

float __ocml_ldexp_f32(float, int);
_CLC_DEFINE_BINARY_BUILTIN(float, __spirv_ocl_ldexp, __ocml_ldexp_f32, float,
                           int)
_CLC_DEFINE_BINARY_BUILTIN(float, __spirv_ocl_ldexp, __ocml_ldexp_f32, float,
                           uint)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
double __ocml_ldexp_f64(double, int);
_CLC_DEFINE_BINARY_BUILTIN(double, __spirv_ocl_ldexp, __ocml_ldexp_f64, double,
                           int)
_CLC_DEFINE_BINARY_BUILTIN(double, __spirv_ocl_ldexp, __ocml_ldexp_f64, double,
                           uint)
#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
half __ocml_ldexp_f16(half, int);
_CLC_DEFINE_BINARY_BUILTIN(half, __spirv_ocl_ldexp, __ocml_ldexp_f16, half, int)
_CLC_DEFINE_BINARY_BUILTIN(half, __spirv_ocl_ldexp, __ocml_ldexp_f16, half,
                           uint)
#endif
