//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clcmacro.h>
#include <spirv/spirv.h>

#define _CLC_GEN_VECTORTIMESSCALAR_IMPL(DECLSPEC, TYPE, VECLEN)                \
  DECLSPEC TYPE##VECLEN __spirv_VectorTimesScalar(TYPE##VECLEN x, TYPE y) {    \
    return x * (TYPE##VECLEN)y;                                                \
  }

#define _CLC_GEN_VECTORTIMESSCALAR(DECLSPEC, TYPE)                             \
  _CLC_GEN_VECTORTIMESSCALAR_IMPL(DECLSPEC, TYPE, 2)                           \
  _CLC_GEN_VECTORTIMESSCALAR_IMPL(DECLSPEC, TYPE, 3)                           \
  _CLC_GEN_VECTORTIMESSCALAR_IMPL(DECLSPEC, TYPE, 4)                           \
  _CLC_GEN_VECTORTIMESSCALAR_IMPL(DECLSPEC, TYPE, 8)                           \
  _CLC_GEN_VECTORTIMESSCALAR_IMPL(DECLSPEC, TYPE, 16)

_CLC_GEN_VECTORTIMESSCALAR(_CLC_OVERLOAD _CLC_DEF, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_GEN_VECTORTIMESSCALAR(_CLC_OVERLOAD _CLC_DEF, double)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_GEN_VECTORTIMESSCALAR(_CLC_OVERLOAD _CLC_DEF, half)

#endif
