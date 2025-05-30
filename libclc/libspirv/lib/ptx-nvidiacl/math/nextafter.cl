//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/ptx-nvidiacl/libdevice.h>
#include <clc/clcmacro.h>
#include <clc/utils.h>
#include <libspirv/spirv.h>

#define __SPIRV_FUNCTION __spirv_ocl_nextafter
#define __CLC_BUILTIN __nv_nextafter
#define __CLC_BUILTIN_F __CLC_XCONCAT(__CLC_BUILTIN, f)
#define __CLC_BUILTIN_D __CLC_BUILTIN

_CLC_DEFINE_BINARY_BUILTIN(float, __SPIRV_FUNCTION, __CLC_BUILTIN_F, float,
                           float)

#ifndef __FLOAT_ONLY

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEFINE_BINARY_BUILTIN(double, __SPIRV_FUNCTION, __CLC_BUILTIN_D, double,
                           double)

#endif

#include <libspirv/generic/math/half_nextafter.inc>

#endif
