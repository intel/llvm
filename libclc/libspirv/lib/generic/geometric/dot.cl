//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clcmacro.h>
#include <libspirv/spirv.h>

#define _CLC_GEN_DOT(DECLSPEC, TYPE)                                           \
  DECLSPEC TYPE __spirv_Dot(TYPE x, TYPE y) { return x * y; }                  \
  DECLSPEC TYPE __spirv_Dot(TYPE##2 x, TYPE##2 y) {                            \
    return __spirv_Dot(x.x, y.x) + __spirv_Dot(x.y, y.y);                      \
  }                                                                            \
                                                                               \
  DECLSPEC TYPE __spirv_Dot(TYPE##3 x, TYPE##3 y) {                            \
    return __spirv_Dot(x.x, y.x) + __spirv_Dot(x.y, y.y) +                     \
           __spirv_Dot(x.z, y.z);                                              \
  }                                                                            \
                                                                               \
  DECLSPEC TYPE __spirv_Dot(TYPE##4 x, TYPE##4 y) {                            \
    return __spirv_Dot(x.lo, y.lo) + __spirv_Dot(x.hi, y.hi);                  \
  }                                                                            \
                                                                               \
  DECLSPEC TYPE __spirv_Dot(TYPE##8 x, TYPE##8 y) {                            \
    return __spirv_Dot(x.lo, y.lo) + __spirv_Dot(x.hi, y.hi);                  \
  }                                                                            \
                                                                               \
  DECLSPEC TYPE __spirv_Dot(TYPE##16 x, TYPE##16 y) {                          \
    return __spirv_Dot(x.lo, y.lo) + __spirv_Dot(x.hi, y.hi);                  \
  }

_CLC_GEN_DOT(_CLC_OVERLOAD _CLC_DEF, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_GEN_DOT(_CLC_OVERLOAD _CLC_DEF, double)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_GEN_DOT(_CLC_OVERLOAD _CLC_DEF, half)

#endif
