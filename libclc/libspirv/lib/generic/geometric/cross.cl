//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF float3 __spirv_ocl_cross(float3 p0, float3 p1) {
  return (float3)(p0.y * p1.z - p0.z * p1.y, p0.z * p1.x - p0.x * p1.z,
                  p0.x * p1.y - p0.y * p1.x);
}

_CLC_OVERLOAD _CLC_DEF float4 __spirv_ocl_cross(float4 p0, float4 p1) {
  return (float4)(p0.y * p1.z - p0.z * p1.y, p0.z * p1.x - p0.x * p1.z,
                  p0.x * p1.y - p0.y * p1.x, 0.f);
}

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double3 __spirv_ocl_cross(double3 p0, double3 p1) {
  return (double3)(p0.y * p1.z - p0.z * p1.y, p0.z * p1.x - p0.x * p1.z,
                   p0.x * p1.y - p0.y * p1.x);
}

_CLC_OVERLOAD _CLC_DEF double4 __spirv_ocl_cross(double4 p0, double4 p1) {
  return (double4)(p0.y * p1.z - p0.z * p1.y, p0.z * p1.x - p0.x * p1.z,
                   p0.x * p1.y - p0.y * p1.x, 0.f);
}
#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF half3 __spirv_ocl_cross(half3 p0, half3 p1) {
  return (half3)(p0.y * p1.z - p0.z * p1.y, p0.z * p1.x - p0.x * p1.z,
                 p0.x * p1.y - p0.y * p1.x);
}

_CLC_OVERLOAD _CLC_DEF half4 __spirv_ocl_cross(half4 p0, half4 p1) {
  return (half4)(p0.y * p1.z - p0.z * p1.y, p0.z * p1.x - p0.x * p1.z,
                 p0.x * p1.y - p0.y * p1.x, 0.f);
}
#endif
