//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#include <libspirv/ptx-nvidiacl/libdevice.h>

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

#ifdef cl_khr_fp64
__CLC_MODF(__nv_modf, double, double)
#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__CLC_MODF(__nv_modff, float, half)
#endif

#define __CLC_FUNCTION __spirv_ocl_modf

#define __CLC_ADDRSPACE private
#define __CLC_BODY <clc/shared/unary_def_with_ptr_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __CLC_ADDRSPACE

#define __CLC_ADDRSPACE local
#define __CLC_BODY <clc/shared/unary_def_with_ptr_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __CLC_ADDRSPACE

#define __CLC_ADDRSPACE global
#define __CLC_BODY <clc/shared/unary_def_with_ptr_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __CLC_ADDRSPACE
