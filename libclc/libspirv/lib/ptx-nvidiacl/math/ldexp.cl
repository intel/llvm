//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#include <libspirv/ptx-nvidiacl/libdevice.h>
#include <clc/clcmacro.h>
#include <clc/math/clc_subnormal_config.h>
#include <clc/math/math.h>

_CLC_DEFINE_BINARY_BUILTIN(float, __spirv_ocl_ldexp, __nv_ldexpf, float, int)
_CLC_DEFINE_BINARY_BUILTIN(float, __spirv_ocl_ldexp, __nv_ldexpf, float, uint)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
_CLC_DEFINE_BINARY_BUILTIN(double, __spirv_ocl_ldexp, __nv_ldexp, double, int)
_CLC_DEFINE_BINARY_BUILTIN(double, __spirv_ocl_ldexp, __nv_ldexp, double, uint)
#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
_CLC_DEFINE_BINARY_BUILTIN(half, __spirv_ocl_ldexp, __nv_ldexpf, half, int)
_CLC_DEFINE_BINARY_BUILTIN(half, __spirv_ocl_ldexp, __nv_ldexpf, half, uint)
#endif
