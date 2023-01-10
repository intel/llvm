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

extern int __clc_nvvm_reflect_arch();

#define __USE_TANH_APPROX (__clc_nvvm_reflect_arch() >= 750)

_CLC_DEF _CLC_OVERLOAD float __clc_native_tanh(float x) {
  return (__USE_TANH_APPROX) ? __nvvm_tanh_approx_f(x) : __nv_tanhf(x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __clc_native_tanh, float)

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEF _CLC_OVERLOAD half __clc_native_tanh(half x) {
  return (__USE_TANH_APPROX) ? __nvvm_tanh_approx_f16(x) : __nv_tanhf(x);
}

_CLC_DEF _CLC_OVERLOAD half2 __clc_native_tanh(half2 x) {
  return (__USE_TANH_APPROX) ? __nvvm_tanh_approx_f16x2(x)
                             : (half2)(__nv_tanhf(x.x), __nv_tanhf(x.y));
}

_CLC_UNARY_VECTORIZE_HAVE2(_CLC_OVERLOAD _CLC_DEF, half, __clc_native_tanh, half)

#endif

#undef __USE_TANH_APPROX

