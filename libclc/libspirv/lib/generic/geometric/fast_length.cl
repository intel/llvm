//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_fast_length(float p) {
  return __spirv_ocl_fabs(p);
}

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_fast_length(float2 p) {
  return __spirv_ocl_half_sqrt(__spirv_Dot(p, p));
}

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_fast_length(float3 p) {
  return __spirv_ocl_half_sqrt(__spirv_Dot(p, p));
}

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_fast_length(float4 p) {
  return __spirv_ocl_half_sqrt(__spirv_Dot(p, p));
}
