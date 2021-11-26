//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clcmacro.h>
#include <spirv/spirv.h>

_CLC_DEFINE_BINARY_BUILTIN(float, __spirv_ocl_nextafter, __builtin_nextafterf,
                           float, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEFINE_BINARY_BUILTIN(double, __spirv_ocl_nextafter, __builtin_nextafter,
                           double, double)

#endif

#include "half_nextafter.inc"
