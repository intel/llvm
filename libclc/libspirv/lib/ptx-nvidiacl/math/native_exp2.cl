//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#define __CLC_FUNCTION __spirv_ocl_native_exp2

extern int __clc_nvvm_reflect_ftz();

_CLC_DEF _CLC_OVERLOAD float __spirv_ocl_native_exp2(float x) {
  return (__clc_nvvm_reflect_ftz()) ? __nvvm_ex2_approx_ftz_f(x)
                                    : __nvvm_ex2_approx_f(x);
}

#define __CLC_FLOAT_ONLY
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __CLC_FLOAT_ONLY

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

#undef __CLC_FUNCTION
#define __CLC_FUNCTION __clc_native_exp2

#undef __CLC_MIN_VECSIZE
#define __CLC_MIN_VECSIZE 3
#define __CLC_HALF_ONLY
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __CLC_HALF_ONLY
#undef __CLC_MIN_VECSIZE

#undef __USE_HALF_EXP2_APPROX

#endif // cl_khr_fp16

#undef FUNCTION
