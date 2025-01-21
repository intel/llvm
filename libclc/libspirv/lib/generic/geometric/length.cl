//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_length(float p) {
  return __spirv_ocl_fabs(p);
}

#define V_FLENGTH(p)                                                           \
  float l2 = __spirv_Dot(p, p);                                                \
                                                                               \
  if (l2 < FLT_MIN) {                                                          \
    p *= 0x1.0p+86F;                                                           \
    return __spirv_ocl_sqrt(__spirv_Dot(p, p)) * 0x1.0p-86F;                   \
  } else if (l2 == INFINITY) {                                                 \
    p *= 0x1.0p-65F;                                                           \
    return __spirv_ocl_sqrt(__spirv_Dot(p, p)) * 0x1.0p+65F;                   \
  }                                                                            \
                                                                               \
  return __spirv_ocl_sqrt(l2);

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_length(float2 p) { V_FLENGTH(p); }

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_length(float3 p) { V_FLENGTH(p); }

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_length(float4 p) { V_FLENGTH(p); }

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double __spirv_ocl_length(double p) {
  return __spirv_ocl_fabs(p);
}

#define V_DLENGTH(p)                                                           \
  double l2 = __spirv_Dot(p, p);                                               \
                                                                               \
  if (l2 < DBL_MIN) {                                                          \
    p *= 0x1.0p+563;                                                           \
    return __spirv_ocl_sqrt(__spirv_Dot(p, p)) * 0x1.0p-563;                   \
  } else if (l2 == INFINITY) {                                                 \
    p *= 0x1.0p-513;                                                           \
    return __spirv_ocl_sqrt(__spirv_Dot(p, p)) * 0x1.0p+513;                   \
  }                                                                            \
                                                                               \
  return __spirv_ocl_sqrt(l2);

_CLC_OVERLOAD _CLC_DEF double __spirv_ocl_length(double2 p) { V_DLENGTH(p); }

_CLC_OVERLOAD _CLC_DEF double __spirv_ocl_length(double3 p) { V_DLENGTH(p); }

_CLC_OVERLOAD _CLC_DEF double __spirv_ocl_length(double4 p) { V_DLENGTH(p); }

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF half __spirv_ocl_length(half p) {
  return __spirv_ocl_fabs(p);
}

// Only available in CLC1.2
#ifndef HALF_MIN
#define HALF_MIN 0x1.0p-14h
#endif

#define V_HLENGTH(p)                                                           \
  half l2 = __spirv_Dot(p, p);                                                 \
                                                                               \
  if (l2 < HALF_MIN) {                                                         \
    p *= 0x1.0p+12h;                                                           \
    return __spirv_ocl_sqrt(__spirv_Dot(p, p)) * 0x1.0p-12h;                   \
  } else if (l2 == INFINITY) {                                                 \
    p *= 0x1.0p-7h;                                                            \
    return __spirv_ocl_sqrt(__spirv_Dot(p, p)) * 0x1.0p+7h;                    \
  }                                                                            \
                                                                               \
  return __spirv_ocl_sqrt(l2);

_CLC_OVERLOAD _CLC_DEF half __spirv_ocl_length(half2 p) { V_HLENGTH(p); }

_CLC_OVERLOAD _CLC_DEF half __spirv_ocl_length(half3 p) { V_HLENGTH(p); }

_CLC_OVERLOAD _CLC_DEF half __spirv_ocl_length(half4 p) { V_HLENGTH(p); }

#endif
