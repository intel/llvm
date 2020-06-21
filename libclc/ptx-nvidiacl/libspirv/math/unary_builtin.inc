//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../../../generic/lib/clcmacro.h"
#include "utils.h"

#ifndef __CLC_BUILTIN_F
#define __CLC_BUILTIN_F __CLC_XCONCAT(__CLC_BUILTIN, f)
#endif

_CLC_DEFINE_UNARY_BUILTIN(float, __CLC_FUNCTION, __CLC_BUILTIN_F, float)

#ifndef __FLOAT_ONLY

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEFINE_UNARY_BUILTIN(double, __CLC_FUNCTION, __CLC_BUILTIN, double)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEFINE_UNARY_BUILTIN(half, __CLC_FUNCTION, __CLC_BUILTIN, half)

#endif

#endif

#undef __CLC_BUILTIN
#undef __CLC_BUILTIN_F
#undef __CLC_FUNCTION
