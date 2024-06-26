//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>
#include <utils.h>
#include <clcmacro.h>

#define __CLC_BODY <frexp.inc>
#define __CLC_ADDRESS_SPACE private
#include <clc/math/gentype.inc>
#undef __CLC_ADDRESS_SPACE

#define __CLC_BODY <frexp.inc>
#define __CLC_ADDRESS_SPACE global
#include <clc/math/gentype.inc>
#undef __CLC_ADDRESS_SPACE

#define __CLC_BODY <frexp.inc>
#define __CLC_ADDRESS_SPACE local
#include <clc/math/gentype.inc>
#undef __CLC_ADDRESS_SPACE

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
#define __CLC_BODY <frexp.inc>
#define __CLC_ADDRESS_SPACE generic
#include <clc/math/gentype.inc>
#undef __CLC_ADDRESS_SPACE
#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define _CLC_DEFINE_NO_VEC(RET_TYPE, FUNCTION, BUILTIN, ARG1_TYPE,     \
                                   ARG2_TYPE)                                  \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE FUNCTION(ARG1_TYPE x, ARG2_TYPE y) {         \
    return BUILTIN(x, y);                                                      \
  }

_CLC_DEFINE_NO_VEC(half, __spirv_ocl_frexp, __builtin_frexp, half, global int *)
_CLC_V_V_VP_VECTORIZE(_CLC_DEF _CLC_OVERLOAD, half, __spirv_ocl_frexp, half, global, int)
_CLC_DEFINE_NO_VEC(half, __spirv_ocl_frexp, __builtin_frexp, half, local int *)
_CLC_V_V_VP_VECTORIZE(_CLC_DEF _CLC_OVERLOAD, half, __spirv_ocl_frexp, half, local, int)
_CLC_DEFINE_NO_VEC(half, __spirv_ocl_frexp, __builtin_frexp, half, int *)
_CLC_V_V_VP_VECTORIZE(_CLC_DEF _CLC_OVERLOAD, half, __spirv_ocl_frexp, half, , int)

#endif
