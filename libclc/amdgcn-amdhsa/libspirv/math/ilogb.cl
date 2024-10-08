//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clcmacro.h>
#include <spirv/spirv.h>

int __ocml_ilogb_f32(float);
_CLC_DEFINE_UNARY_BUILTIN(int, __spirv_ocl_ilogb, __ocml_ilogb_f32, float)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
int __ocml_ilogb_f64(double);
_CLC_DEFINE_UNARY_BUILTIN(int, __spirv_ocl_ilogb, __ocml_ilogb_f64, double)
#endif // cl_khr_fp64

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
int __ocml_ilogb_f16(half);
_CLC_DEFINE_UNARY_BUILTIN(int, __spirv_ocl_ilogb, __ocml_ilogb_f16, half)
#endif // cl_khr_fp16

#undef __CLC_BUILTIN
#undef __CLC_BUILTIN_F
#undef __CLC_FUNCTION
