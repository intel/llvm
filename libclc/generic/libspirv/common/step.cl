//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

#include <clc/clcmacro.h>

#define STEP_DEF(TYPE, TYPOSTFIX)                                              \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_ocl_step(TYPE edge, TYPE x) {            \
    return x < edge ? 0.0##TYPOSTFIX : 1.0##TYPOSTFIX;                         \
  }                                                                            \
  _CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, TYPE, __spirv_ocl_step, TYPE,  \
                        TYPE)

STEP_DEF(float, f)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

STEP_DEF(double, )

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

STEP_DEF(half, h)

#endif
