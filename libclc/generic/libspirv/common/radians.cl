//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

#include <clc/clcmacro.h>

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_radians(float degrees) {
  // pi/180 = ~0.01745329251994329577 or 0x1.1df46a2529d39p-6 or 0x1.1df46ap-6F
  return 0x1.1df46ap-6F * degrees;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __spirv_ocl_radians, float);


#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double __spirv_ocl_radians(double degrees) {
  // pi/180 = ~0.01745329251994329577 or 0x1.1df46a2529d39p-6 or 0x1.1df46ap-6F
  return 0x1.1df46a2529d39p-6 * degrees;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, __spirv_ocl_radians,
                     double);

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF half __spirv_ocl_radians(half degrees) {
  // pi/180 = ~0.01745329251994329577 or 0x1.1df46a2529d39p-6 or 0x1.1df46ap-6F
  return M_PI_OVER_180_H * degrees;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, __spirv_ocl_radians, half);

#endif
