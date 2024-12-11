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

#define __CLC_MODF_IMPL(ADDRSPACE, BUILTIN, FP_TYPE, ARG_TYPE)                 \
  _CLC_OVERLOAD _CLC_DEF ARG_TYPE __spirv_ocl_modf(ARG_TYPE x,                 \
                                                   ADDRSPACE ARG_TYPE *iptr) { \
    FP_TYPE stack_iptr;                                                        \
    ARG_TYPE ret = BUILTIN(x, &stack_iptr);                                    \
    *iptr = stack_iptr;                                                        \
    return ret;                                                                \
  }

#define __CLC_MODF(BUILTIN, FP_TYPE, ARG_TYPE)                                 \
  __CLC_MODF_IMPL(private, BUILTIN, FP_TYPE, ARG_TYPE)                         \
  __CLC_MODF_IMPL(local, BUILTIN, FP_TYPE, ARG_TYPE)                           \
  __CLC_MODF_IMPL(global, BUILTIN, FP_TYPE, ARG_TYPE)

__CLC_MODF(__nv_modff, float, float)

_CLC_V_V_VP_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __spirv_ocl_modf, float,
                      private, float)
_CLC_V_V_VP_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __spirv_ocl_modf, float,
                      local, float)
_CLC_V_V_VP_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __spirv_ocl_modf, float,
                      global, float)

#ifdef cl_khr_fp64
__CLC_MODF(__nv_modf, double, double)

_CLC_V_V_VP_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, __spirv_ocl_modf, double,
                      private, double)
_CLC_V_V_VP_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, __spirv_ocl_modf, double,
                      local, double)
_CLC_V_V_VP_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, __spirv_ocl_modf, double,
                      global, double)
#endif

#ifdef cl_khr_fp16
__CLC_MODF(__nv_modff, float, half)

_CLC_V_V_VP_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, __spirv_ocl_modf, half,
                      private, half)
_CLC_V_V_VP_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, __spirv_ocl_modf, half,
                      local, half)
_CLC_V_V_VP_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, __spirv_ocl_modf, half,
                      global, half)
#endif
