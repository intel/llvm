//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clcmacro.h>
#include <clc/common/clc_smoothstep.h>
#include <libspirv/spirv.h>

#define SMOOTHSTEP_SINGLE_DEF(X_TYPE)                                          \
  _CLC_OVERLOAD _CLC_DEF X_TYPE __spirv_ocl_smoothstep(                        \
      X_TYPE edge0, X_TYPE edge1, X_TYPE x) {                                  \
    return __clc_smoothstep(edge0, edge1, x);                                  \
  }

#define SMOOTHSTEP_DEF(type)                                                   \
  SMOOTHSTEP_SINGLE_DEF(type)                                                  \
  SMOOTHSTEP_SINGLE_DEF(type##2)                                               \
  SMOOTHSTEP_SINGLE_DEF(type##3)                                               \
  SMOOTHSTEP_SINGLE_DEF(type##4)                                               \
  SMOOTHSTEP_SINGLE_DEF(type##8)                                               \
  SMOOTHSTEP_SINGLE_DEF(type##16)

SMOOTHSTEP_DEF(float)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

SMOOTHSTEP_DEF(double);

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

SMOOTHSTEP_DEF(half);

#endif
