//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DECL float3 __spirv_ocl_cross(float3 p0, float3 p1);
_CLC_OVERLOAD _CLC_DECL float4 __spirv_ocl_cross(float4 p0, float4 p1);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL double3 __spirv_ocl_cross(double3 p0, double3 p1);
_CLC_OVERLOAD _CLC_DECL double4 __spirv_ocl_cross(double4 p0, double4 p1);
#endif
