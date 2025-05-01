//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clcmacro.h>
#include <clc/math/clc_mad.h>
#include <libspirv/spirv.h>

_CLC_DEFINE_TERNARY_BUILTIN(float, __spirv_ocl_mad, __clc_mad, float, float, float)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEFINE_TERNARY_BUILTIN(double, __spirv_ocl_mad, __clc_mad, double, double, double)

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEFINE_TERNARY_BUILTIN(half, __spirv_ocl_mad, __clc_mad, half, half, half)

#endif
