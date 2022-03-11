//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

#include "../../include/libdevice.h"
#include <clcmacro.h>

#define __CLC_FUNCTION __spirv_ocl_native_exp2
#define __CLC_BUILTIN __nv_exp2
#define __CLC_BUILTIN_F __CLC_XCONCAT(__CLC_BUILTIN, f)

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

int __clc_nvvm_reflect_arch();
#define __USE_HALF_EXP2_APPROX (__clc_nvvm_reflect_arch() >= 750)

_CLC_DEF _CLC_OVERLOAD half __clc_native_exp2(half x) {
  return (__USE_HALF_EXP2_APPROX) ? __nvvm_ex2_approx_f16(x)
                                  : __spirv_ocl_native_exp2((float)x);
}

_CLC_DEF _CLC_OVERLOAD half2 __clc_native_exp2(half2 x) {
  return (__USE_HALF_EXP2_APPROX)
             ? __nvvm_ex2_approx_f16x2(x)
             : (half2)(__spirv_ocl_native_exp2((float)x.x),
                       __spirv_ocl_native_exp2((float)x.y));
}

_CLC_UNARY_VECTORIZE_HAVE2(_CLC_OVERLOAD _CLC_DEF, half, __clc_native_exp2,
                           half)

#undef __USE_HALF_EXP2_APPROX

#endif // cl_khr_fp16

// Undef halfs before uncluding unary builtins, as they are handled above.
#ifdef cl_khr_fp16
#undef cl_khr_fp16
#endif // cl_khr_fp16
#include <math/unary_builtin.inc>
