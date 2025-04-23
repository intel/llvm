//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#ifdef cl_khr_fp16
#ifdef __CLC_HAS_FLOAT16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int8_t
__spirv_ConvertFToS_Rchar(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t
__spirv_ConvertFToS_Rchar16(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar16(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t
__spirv_ConvertFToS_Rchar16_rte(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar16_rte(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t
__spirv_ConvertFToS_Rchar16_rtn(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar16_rtn(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t
__spirv_ConvertFToS_Rchar16_rtp(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar16_rtp(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t
__spirv_ConvertFToS_Rchar16_rtz(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar16_rtz(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t
__spirv_ConvertFToS_Rchar16_sat(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar16_sat(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t
__spirv_ConvertFToS_Rchar16_sat_rte(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar16_sat_rte(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t
__spirv_ConvertFToS_Rchar16_sat_rtn(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar16_sat_rtn(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t
__spirv_ConvertFToS_Rchar16_sat_rtp(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar16_sat_rtp(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t
__spirv_ConvertFToS_Rchar16_sat_rtz(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar16_sat_rtz(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t
__spirv_ConvertFToS_Rchar2(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar2(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t
__spirv_ConvertFToS_Rchar2_rte(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar2_rte(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t
__spirv_ConvertFToS_Rchar2_rtn(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar2_rtn(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t
__spirv_ConvertFToS_Rchar2_rtp(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar2_rtp(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t
__spirv_ConvertFToS_Rchar2_rtz(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar2_rtz(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t
__spirv_ConvertFToS_Rchar2_sat(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar2_sat(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t
__spirv_ConvertFToS_Rchar2_sat_rte(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar2_sat_rte(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t
__spirv_ConvertFToS_Rchar2_sat_rtn(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar2_sat_rtn(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t
__spirv_ConvertFToS_Rchar2_sat_rtp(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar2_sat_rtp(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t
__spirv_ConvertFToS_Rchar2_sat_rtz(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar2_sat_rtz(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t
__spirv_ConvertFToS_Rchar3(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar3(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t
__spirv_ConvertFToS_Rchar3_rte(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar3_rte(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t
__spirv_ConvertFToS_Rchar3_rtn(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar3_rtn(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t
__spirv_ConvertFToS_Rchar3_rtp(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar3_rtp(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t
__spirv_ConvertFToS_Rchar3_rtz(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar3_rtz(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t
__spirv_ConvertFToS_Rchar3_sat(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar3_sat(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t
__spirv_ConvertFToS_Rchar3_sat_rte(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar3_sat_rte(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t
__spirv_ConvertFToS_Rchar3_sat_rtn(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar3_sat_rtn(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t
__spirv_ConvertFToS_Rchar3_sat_rtp(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar3_sat_rtp(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t
__spirv_ConvertFToS_Rchar3_sat_rtz(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar3_sat_rtz(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t
__spirv_ConvertFToS_Rchar4(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar4(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t
__spirv_ConvertFToS_Rchar4_rte(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar4_rte(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t
__spirv_ConvertFToS_Rchar4_rtn(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar4_rtn(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t
__spirv_ConvertFToS_Rchar4_rtp(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar4_rtp(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t
__spirv_ConvertFToS_Rchar4_rtz(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar4_rtz(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t
__spirv_ConvertFToS_Rchar4_sat(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar4_sat(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t
__spirv_ConvertFToS_Rchar4_sat_rte(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar4_sat_rte(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t
__spirv_ConvertFToS_Rchar4_sat_rtn(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar4_sat_rtn(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t
__spirv_ConvertFToS_Rchar4_sat_rtp(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar4_sat_rtp(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t
__spirv_ConvertFToS_Rchar4_sat_rtz(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar4_sat_rtz(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t
__spirv_ConvertFToS_Rchar8(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar8(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t
__spirv_ConvertFToS_Rchar8_rte(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar8_rte(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t
__spirv_ConvertFToS_Rchar8_rtn(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar8_rtn(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t
__spirv_ConvertFToS_Rchar8_rtp(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar8_rtp(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t
__spirv_ConvertFToS_Rchar8_rtz(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar8_rtz(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t
__spirv_ConvertFToS_Rchar8_sat(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar8_sat(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t
__spirv_ConvertFToS_Rchar8_sat_rte(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar8_sat_rte(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t
__spirv_ConvertFToS_Rchar8_sat_rtn(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar8_sat_rtn(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t
__spirv_ConvertFToS_Rchar8_sat_rtp(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar8_sat_rtp(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t
__spirv_ConvertFToS_Rchar8_sat_rtz(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar8_sat_rtz(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int8_t
__spirv_ConvertFToS_Rchar_rte(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar_rte(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int8_t
__spirv_ConvertFToS_Rchar_rtn(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar_rtn(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int8_t
__spirv_ConvertFToS_Rchar_rtp(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar_rtp(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int8_t
__spirv_ConvertFToS_Rchar_rtz(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar_rtz(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int8_t
__spirv_ConvertFToS_Rchar_sat(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar_sat(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int8_t
__spirv_ConvertFToS_Rchar_sat_rte(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar_sat_rte(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int8_t
__spirv_ConvertFToS_Rchar_sat_rtn(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar_sat_rtn(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int8_t
__spirv_ConvertFToS_Rchar_sat_rtp(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar_sat_rtp(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int8_t
__spirv_ConvertFToS_Rchar_sat_rtz(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rchar_sat_rtz(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int32_t
__spirv_ConvertFToS_Rint(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rint(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int32_t
__spirv_ConvertFToS_Rint16(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rint16(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int32_t
__spirv_ConvertFToS_Rint16_rte(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rint16_rte(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int32_t
__spirv_ConvertFToS_Rint16_rtn(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rint16_rtn(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int32_t
__spirv_ConvertFToS_Rint16_rtp(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rint16_rtp(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int32_t
__spirv_ConvertFToS_Rint16_rtz(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rint16_rtz(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int32_t
__spirv_ConvertFToS_Rint16_sat(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rint16_sat(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int32_t
__spirv_ConvertFToS_Rint16_sat_rte(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rint16_sat_rte(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int32_t
__spirv_ConvertFToS_Rint16_sat_rtn(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rint16_sat_rtn(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int32_t
__spirv_ConvertFToS_Rint16_sat_rtp(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rint16_sat_rtp(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int32_t
__spirv_ConvertFToS_Rint16_sat_rtz(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rint16_sat_rtz(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int32_t
__spirv_ConvertFToS_Rint2(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rint2(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int32_t
__spirv_ConvertFToS_Rint2_rte(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rint2_rte(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int32_t
__spirv_ConvertFToS_Rint2_rtn(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rint2_rtn(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int32_t
__spirv_ConvertFToS_Rint2_rtp(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rint2_rtp(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int32_t
__spirv_ConvertFToS_Rint2_rtz(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rint2_rtz(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int32_t
__spirv_ConvertFToS_Rint2_sat(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rint2_sat(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int32_t
__spirv_ConvertFToS_Rint2_sat_rte(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rint2_sat_rte(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int32_t
__spirv_ConvertFToS_Rint2_sat_rtn(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rint2_sat_rtn(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int32_t
__spirv_ConvertFToS_Rint2_sat_rtp(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rint2_sat_rtp(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int32_t
__spirv_ConvertFToS_Rint2_sat_rtz(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rint2_sat_rtz(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int32_t
__spirv_ConvertFToS_Rint3(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rint3(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int32_t
__spirv_ConvertFToS_Rint3_rte(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rint3_rte(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int32_t
__spirv_ConvertFToS_Rint3_rtn(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rint3_rtn(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int32_t
__spirv_ConvertFToS_Rint3_rtp(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rint3_rtp(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int32_t
__spirv_ConvertFToS_Rint3_rtz(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rint3_rtz(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int32_t
__spirv_ConvertFToS_Rint3_sat(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rint3_sat(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int32_t
__spirv_ConvertFToS_Rint3_sat_rte(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rint3_sat_rte(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int32_t
__spirv_ConvertFToS_Rint3_sat_rtn(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rint3_sat_rtn(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int32_t
__spirv_ConvertFToS_Rint3_sat_rtp(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rint3_sat_rtp(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int32_t
__spirv_ConvertFToS_Rint3_sat_rtz(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rint3_sat_rtz(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int32_t
__spirv_ConvertFToS_Rint4(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rint4(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int32_t
__spirv_ConvertFToS_Rint4_rte(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rint4_rte(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int32_t
__spirv_ConvertFToS_Rint4_rtn(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rint4_rtn(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int32_t
__spirv_ConvertFToS_Rint4_rtp(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rint4_rtp(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int32_t
__spirv_ConvertFToS_Rint4_rtz(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rint4_rtz(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int32_t
__spirv_ConvertFToS_Rint4_sat(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rint4_sat(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int32_t
__spirv_ConvertFToS_Rint4_sat_rte(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rint4_sat_rte(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int32_t
__spirv_ConvertFToS_Rint4_sat_rtn(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rint4_sat_rtn(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int32_t
__spirv_ConvertFToS_Rint4_sat_rtp(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rint4_sat_rtp(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int32_t
__spirv_ConvertFToS_Rint4_sat_rtz(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rint4_sat_rtz(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int32_t
__spirv_ConvertFToS_Rint8(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rint8(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int32_t
__spirv_ConvertFToS_Rint8_rte(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rint8_rte(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int32_t
__spirv_ConvertFToS_Rint8_rtn(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rint8_rtn(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int32_t
__spirv_ConvertFToS_Rint8_rtp(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rint8_rtp(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int32_t
__spirv_ConvertFToS_Rint8_rtz(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rint8_rtz(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int32_t
__spirv_ConvertFToS_Rint8_sat(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rint8_sat(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int32_t
__spirv_ConvertFToS_Rint8_sat_rte(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rint8_sat_rte(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int32_t
__spirv_ConvertFToS_Rint8_sat_rtn(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rint8_sat_rtn(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int32_t
__spirv_ConvertFToS_Rint8_sat_rtp(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rint8_sat_rtp(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int32_t
__spirv_ConvertFToS_Rint8_sat_rtz(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rint8_sat_rtz(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int32_t
__spirv_ConvertFToS_Rint_rte(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rint_rte(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int32_t
__spirv_ConvertFToS_Rint_rtn(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rint_rtn(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int32_t
__spirv_ConvertFToS_Rint_rtp(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rint_rtp(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int32_t
__spirv_ConvertFToS_Rint_rtz(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rint_rtz(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int32_t
__spirv_ConvertFToS_Rint_sat(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rint_sat(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int32_t
__spirv_ConvertFToS_Rint_sat_rte(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rint_sat_rte(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int32_t
__spirv_ConvertFToS_Rint_sat_rtn(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rint_sat_rtn(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int32_t
__spirv_ConvertFToS_Rint_sat_rtp(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rint_sat_rtp(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int32_t
__spirv_ConvertFToS_Rint_sat_rtz(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rint_sat_rtz(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int64_t
__spirv_ConvertFToS_Rlong(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int64_t
__spirv_ConvertFToS_Rlong16(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong16(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int64_t
__spirv_ConvertFToS_Rlong16_rte(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong16_rte(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int64_t
__spirv_ConvertFToS_Rlong16_rtn(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong16_rtn(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int64_t
__spirv_ConvertFToS_Rlong16_rtp(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong16_rtp(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int64_t
__spirv_ConvertFToS_Rlong16_rtz(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong16_rtz(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int64_t
__spirv_ConvertFToS_Rlong16_sat(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong16_sat(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int64_t
__spirv_ConvertFToS_Rlong16_sat_rte(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong16_sat_rte(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int64_t
__spirv_ConvertFToS_Rlong16_sat_rtn(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong16_sat_rtn(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int64_t
__spirv_ConvertFToS_Rlong16_sat_rtp(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong16_sat_rtp(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int64_t
__spirv_ConvertFToS_Rlong16_sat_rtz(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong16_sat_rtz(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int64_t
__spirv_ConvertFToS_Rlong2(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong2(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int64_t
__spirv_ConvertFToS_Rlong2_rte(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong2_rte(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int64_t
__spirv_ConvertFToS_Rlong2_rtn(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong2_rtn(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int64_t
__spirv_ConvertFToS_Rlong2_rtp(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong2_rtp(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int64_t
__spirv_ConvertFToS_Rlong2_rtz(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong2_rtz(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int64_t
__spirv_ConvertFToS_Rlong2_sat(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong2_sat(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int64_t
__spirv_ConvertFToS_Rlong2_sat_rte(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong2_sat_rte(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int64_t
__spirv_ConvertFToS_Rlong2_sat_rtn(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong2_sat_rtn(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int64_t
__spirv_ConvertFToS_Rlong2_sat_rtp(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong2_sat_rtp(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int64_t
__spirv_ConvertFToS_Rlong2_sat_rtz(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong2_sat_rtz(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int64_t
__spirv_ConvertFToS_Rlong3(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong3(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int64_t
__spirv_ConvertFToS_Rlong3_rte(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong3_rte(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int64_t
__spirv_ConvertFToS_Rlong3_rtn(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong3_rtn(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int64_t
__spirv_ConvertFToS_Rlong3_rtp(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong3_rtp(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int64_t
__spirv_ConvertFToS_Rlong3_rtz(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong3_rtz(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int64_t
__spirv_ConvertFToS_Rlong3_sat(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong3_sat(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int64_t
__spirv_ConvertFToS_Rlong3_sat_rte(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong3_sat_rte(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int64_t
__spirv_ConvertFToS_Rlong3_sat_rtn(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong3_sat_rtn(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int64_t
__spirv_ConvertFToS_Rlong3_sat_rtp(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong3_sat_rtp(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int64_t
__spirv_ConvertFToS_Rlong3_sat_rtz(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong3_sat_rtz(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int64_t
__spirv_ConvertFToS_Rlong4(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong4(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int64_t
__spirv_ConvertFToS_Rlong4_rte(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong4_rte(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int64_t
__spirv_ConvertFToS_Rlong4_rtn(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong4_rtn(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int64_t
__spirv_ConvertFToS_Rlong4_rtp(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong4_rtp(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int64_t
__spirv_ConvertFToS_Rlong4_rtz(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong4_rtz(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int64_t
__spirv_ConvertFToS_Rlong4_sat(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong4_sat(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int64_t
__spirv_ConvertFToS_Rlong4_sat_rte(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong4_sat_rte(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int64_t
__spirv_ConvertFToS_Rlong4_sat_rtn(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong4_sat_rtn(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int64_t
__spirv_ConvertFToS_Rlong4_sat_rtp(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong4_sat_rtp(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int64_t
__spirv_ConvertFToS_Rlong4_sat_rtz(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong4_sat_rtz(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int64_t
__spirv_ConvertFToS_Rlong8(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong8(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int64_t
__spirv_ConvertFToS_Rlong8_rte(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong8_rte(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int64_t
__spirv_ConvertFToS_Rlong8_rtn(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong8_rtn(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int64_t
__spirv_ConvertFToS_Rlong8_rtp(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong8_rtp(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int64_t
__spirv_ConvertFToS_Rlong8_rtz(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong8_rtz(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int64_t
__spirv_ConvertFToS_Rlong8_sat(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong8_sat(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int64_t
__spirv_ConvertFToS_Rlong8_sat_rte(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong8_sat_rte(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int64_t
__spirv_ConvertFToS_Rlong8_sat_rtn(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong8_sat_rtn(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int64_t
__spirv_ConvertFToS_Rlong8_sat_rtp(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong8_sat_rtp(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int64_t
__spirv_ConvertFToS_Rlong8_sat_rtz(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong8_sat_rtz(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int64_t
__spirv_ConvertFToS_Rlong_rte(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong_rte(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int64_t
__spirv_ConvertFToS_Rlong_rtn(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong_rtn(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int64_t
__spirv_ConvertFToS_Rlong_rtp(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong_rtp(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int64_t
__spirv_ConvertFToS_Rlong_rtz(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong_rtz(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int64_t
__spirv_ConvertFToS_Rlong_sat(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong_sat(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int64_t
__spirv_ConvertFToS_Rlong_sat_rte(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong_sat_rte(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int64_t
__spirv_ConvertFToS_Rlong_sat_rtn(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong_sat_rtn(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int64_t
__spirv_ConvertFToS_Rlong_sat_rtp(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong_sat_rtp(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int64_t
__spirv_ConvertFToS_Rlong_sat_rtz(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rlong_sat_rtz(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int16_t
__spirv_ConvertFToS_Rshort(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int16_t
__spirv_ConvertFToS_Rshort16(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort16(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int16_t
__spirv_ConvertFToS_Rshort16_rte(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort16_rte(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int16_t
__spirv_ConvertFToS_Rshort16_rtn(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort16_rtn(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int16_t
__spirv_ConvertFToS_Rshort16_rtp(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort16_rtp(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int16_t
__spirv_ConvertFToS_Rshort16_rtz(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort16_rtz(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int16_t
__spirv_ConvertFToS_Rshort16_sat(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort16_sat(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int16_t
__spirv_ConvertFToS_Rshort16_sat_rte(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort16_sat_rte(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int16_t
__spirv_ConvertFToS_Rshort16_sat_rtn(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort16_sat_rtn(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int16_t
__spirv_ConvertFToS_Rshort16_sat_rtp(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort16_sat_rtp(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int16_t
__spirv_ConvertFToS_Rshort16_sat_rtz(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort16_sat_rtz(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int16_t
__spirv_ConvertFToS_Rshort2(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort2(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int16_t
__spirv_ConvertFToS_Rshort2_rte(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort2_rte(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int16_t
__spirv_ConvertFToS_Rshort2_rtn(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort2_rtn(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int16_t
__spirv_ConvertFToS_Rshort2_rtp(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort2_rtp(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int16_t
__spirv_ConvertFToS_Rshort2_rtz(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort2_rtz(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int16_t
__spirv_ConvertFToS_Rshort2_sat(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort2_sat(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int16_t
__spirv_ConvertFToS_Rshort2_sat_rte(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort2_sat_rte(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int16_t
__spirv_ConvertFToS_Rshort2_sat_rtn(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort2_sat_rtn(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int16_t
__spirv_ConvertFToS_Rshort2_sat_rtp(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort2_sat_rtp(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int16_t
__spirv_ConvertFToS_Rshort2_sat_rtz(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort2_sat_rtz(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int16_t
__spirv_ConvertFToS_Rshort3(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort3(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int16_t
__spirv_ConvertFToS_Rshort3_rte(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort3_rte(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int16_t
__spirv_ConvertFToS_Rshort3_rtn(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort3_rtn(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int16_t
__spirv_ConvertFToS_Rshort3_rtp(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort3_rtp(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int16_t
__spirv_ConvertFToS_Rshort3_rtz(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort3_rtz(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int16_t
__spirv_ConvertFToS_Rshort3_sat(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort3_sat(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int16_t
__spirv_ConvertFToS_Rshort3_sat_rte(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort3_sat_rte(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int16_t
__spirv_ConvertFToS_Rshort3_sat_rtn(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort3_sat_rtn(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int16_t
__spirv_ConvertFToS_Rshort3_sat_rtp(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort3_sat_rtp(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int16_t
__spirv_ConvertFToS_Rshort3_sat_rtz(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort3_sat_rtz(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int16_t
__spirv_ConvertFToS_Rshort4(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort4(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int16_t
__spirv_ConvertFToS_Rshort4_rte(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort4_rte(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int16_t
__spirv_ConvertFToS_Rshort4_rtn(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort4_rtn(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int16_t
__spirv_ConvertFToS_Rshort4_rtp(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort4_rtp(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int16_t
__spirv_ConvertFToS_Rshort4_rtz(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort4_rtz(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int16_t
__spirv_ConvertFToS_Rshort4_sat(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort4_sat(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int16_t
__spirv_ConvertFToS_Rshort4_sat_rte(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort4_sat_rte(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int16_t
__spirv_ConvertFToS_Rshort4_sat_rtn(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort4_sat_rtn(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int16_t
__spirv_ConvertFToS_Rshort4_sat_rtp(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort4_sat_rtp(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int16_t
__spirv_ConvertFToS_Rshort4_sat_rtz(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort4_sat_rtz(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int16_t
__spirv_ConvertFToS_Rshort8(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort8(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int16_t
__spirv_ConvertFToS_Rshort8_rte(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort8_rte(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int16_t
__spirv_ConvertFToS_Rshort8_rtn(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort8_rtn(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int16_t
__spirv_ConvertFToS_Rshort8_rtp(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort8_rtp(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int16_t
__spirv_ConvertFToS_Rshort8_rtz(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort8_rtz(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int16_t
__spirv_ConvertFToS_Rshort8_sat(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort8_sat(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int16_t
__spirv_ConvertFToS_Rshort8_sat_rte(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort8_sat_rte(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int16_t
__spirv_ConvertFToS_Rshort8_sat_rtn(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort8_sat_rtn(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int16_t
__spirv_ConvertFToS_Rshort8_sat_rtp(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort8_sat_rtp(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int16_t
__spirv_ConvertFToS_Rshort8_sat_rtz(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort8_sat_rtz(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int16_t
__spirv_ConvertFToS_Rshort_rte(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort_rte(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int16_t
__spirv_ConvertFToS_Rshort_rtn(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort_rtn(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int16_t
__spirv_ConvertFToS_Rshort_rtp(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort_rtp(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int16_t
__spirv_ConvertFToS_Rshort_rtz(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort_rtz(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int16_t
__spirv_ConvertFToS_Rshort_sat(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort_sat(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int16_t
__spirv_ConvertFToS_Rshort_sat_rte(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort_sat_rte(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int16_t
__spirv_ConvertFToS_Rshort_sat_rtn(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort_sat_rtn(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int16_t
__spirv_ConvertFToS_Rshort_sat_rtp(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort_sat_rtp(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int16_t
__spirv_ConvertFToS_Rshort_sat_rtz(__clc_float16_t args_0) {
  return __spirv_ConvertFToS_Rshort_sat_rtz(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint8_t
__spirv_ConvertFToU_Ruchar(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint8_t
__spirv_ConvertFToU_Ruchar16(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar16(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint8_t
__spirv_ConvertFToU_Ruchar16_rte(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar16_rte(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint8_t
__spirv_ConvertFToU_Ruchar16_rtn(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar16_rtn(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint8_t
__spirv_ConvertFToU_Ruchar16_rtp(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar16_rtp(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint8_t
__spirv_ConvertFToU_Ruchar16_rtz(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar16_rtz(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint8_t
__spirv_ConvertFToU_Ruchar16_sat(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar16_sat(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint8_t
__spirv_ConvertFToU_Ruchar16_sat_rte(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar16_sat_rte(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint8_t
__spirv_ConvertFToU_Ruchar16_sat_rtn(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar16_sat_rtn(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint8_t
__spirv_ConvertFToU_Ruchar16_sat_rtp(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar16_sat_rtp(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint8_t
__spirv_ConvertFToU_Ruchar16_sat_rtz(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar16_sat_rtz(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint8_t
__spirv_ConvertFToU_Ruchar2(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar2(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint8_t
__spirv_ConvertFToU_Ruchar2_rte(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar2_rte(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint8_t
__spirv_ConvertFToU_Ruchar2_rtn(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar2_rtn(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint8_t
__spirv_ConvertFToU_Ruchar2_rtp(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar2_rtp(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint8_t
__spirv_ConvertFToU_Ruchar2_rtz(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar2_rtz(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint8_t
__spirv_ConvertFToU_Ruchar2_sat(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar2_sat(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint8_t
__spirv_ConvertFToU_Ruchar2_sat_rte(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar2_sat_rte(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint8_t
__spirv_ConvertFToU_Ruchar2_sat_rtn(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar2_sat_rtn(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint8_t
__spirv_ConvertFToU_Ruchar2_sat_rtp(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar2_sat_rtp(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint8_t
__spirv_ConvertFToU_Ruchar2_sat_rtz(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar2_sat_rtz(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint8_t
__spirv_ConvertFToU_Ruchar3(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar3(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint8_t
__spirv_ConvertFToU_Ruchar3_rte(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar3_rte(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint8_t
__spirv_ConvertFToU_Ruchar3_rtn(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar3_rtn(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint8_t
__spirv_ConvertFToU_Ruchar3_rtp(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar3_rtp(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint8_t
__spirv_ConvertFToU_Ruchar3_rtz(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar3_rtz(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint8_t
__spirv_ConvertFToU_Ruchar3_sat(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar3_sat(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint8_t
__spirv_ConvertFToU_Ruchar3_sat_rte(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar3_sat_rte(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint8_t
__spirv_ConvertFToU_Ruchar3_sat_rtn(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar3_sat_rtn(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint8_t
__spirv_ConvertFToU_Ruchar3_sat_rtp(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar3_sat_rtp(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint8_t
__spirv_ConvertFToU_Ruchar3_sat_rtz(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar3_sat_rtz(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint8_t
__spirv_ConvertFToU_Ruchar4(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar4(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint8_t
__spirv_ConvertFToU_Ruchar4_rte(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar4_rte(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint8_t
__spirv_ConvertFToU_Ruchar4_rtn(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar4_rtn(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint8_t
__spirv_ConvertFToU_Ruchar4_rtp(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar4_rtp(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint8_t
__spirv_ConvertFToU_Ruchar4_rtz(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar4_rtz(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint8_t
__spirv_ConvertFToU_Ruchar4_sat(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar4_sat(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint8_t
__spirv_ConvertFToU_Ruchar4_sat_rte(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar4_sat_rte(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint8_t
__spirv_ConvertFToU_Ruchar4_sat_rtn(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar4_sat_rtn(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint8_t
__spirv_ConvertFToU_Ruchar4_sat_rtp(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar4_sat_rtp(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint8_t
__spirv_ConvertFToU_Ruchar4_sat_rtz(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar4_sat_rtz(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint8_t
__spirv_ConvertFToU_Ruchar8(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar8(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint8_t
__spirv_ConvertFToU_Ruchar8_rte(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar8_rte(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint8_t
__spirv_ConvertFToU_Ruchar8_rtn(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar8_rtn(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint8_t
__spirv_ConvertFToU_Ruchar8_rtp(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar8_rtp(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint8_t
__spirv_ConvertFToU_Ruchar8_rtz(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar8_rtz(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint8_t
__spirv_ConvertFToU_Ruchar8_sat(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar8_sat(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint8_t
__spirv_ConvertFToU_Ruchar8_sat_rte(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar8_sat_rte(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint8_t
__spirv_ConvertFToU_Ruchar8_sat_rtn(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar8_sat_rtn(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint8_t
__spirv_ConvertFToU_Ruchar8_sat_rtp(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar8_sat_rtp(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint8_t
__spirv_ConvertFToU_Ruchar8_sat_rtz(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar8_sat_rtz(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint8_t
__spirv_ConvertFToU_Ruchar_rte(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar_rte(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint8_t
__spirv_ConvertFToU_Ruchar_rtn(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar_rtn(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint8_t
__spirv_ConvertFToU_Ruchar_rtp(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar_rtp(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint8_t
__spirv_ConvertFToU_Ruchar_rtz(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar_rtz(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint8_t
__spirv_ConvertFToU_Ruchar_sat(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar_sat(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint8_t
__spirv_ConvertFToU_Ruchar_sat_rte(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar_sat_rte(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint8_t
__spirv_ConvertFToU_Ruchar_sat_rtn(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar_sat_rtn(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint8_t
__spirv_ConvertFToU_Ruchar_sat_rtp(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar_sat_rtp(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint8_t
__spirv_ConvertFToU_Ruchar_sat_rtz(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Ruchar_sat_rtz(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint32_t
__spirv_ConvertFToU_Ruint(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint32_t
__spirv_ConvertFToU_Ruint16(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint16(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint32_t
__spirv_ConvertFToU_Ruint16_rte(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint16_rte(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint32_t
__spirv_ConvertFToU_Ruint16_rtn(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint16_rtn(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint32_t
__spirv_ConvertFToU_Ruint16_rtp(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint16_rtp(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint32_t
__spirv_ConvertFToU_Ruint16_rtz(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint16_rtz(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint32_t
__spirv_ConvertFToU_Ruint16_sat(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint16_sat(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint32_t
__spirv_ConvertFToU_Ruint16_sat_rte(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint16_sat_rte(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint32_t
__spirv_ConvertFToU_Ruint16_sat_rtn(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint16_sat_rtn(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint32_t
__spirv_ConvertFToU_Ruint16_sat_rtp(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint16_sat_rtp(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint32_t
__spirv_ConvertFToU_Ruint16_sat_rtz(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint16_sat_rtz(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint32_t
__spirv_ConvertFToU_Ruint2(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint2(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint32_t
__spirv_ConvertFToU_Ruint2_rte(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint2_rte(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint32_t
__spirv_ConvertFToU_Ruint2_rtn(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint2_rtn(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint32_t
__spirv_ConvertFToU_Ruint2_rtp(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint2_rtp(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint32_t
__spirv_ConvertFToU_Ruint2_rtz(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint2_rtz(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint32_t
__spirv_ConvertFToU_Ruint2_sat(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint2_sat(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint32_t
__spirv_ConvertFToU_Ruint2_sat_rte(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint2_sat_rte(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint32_t
__spirv_ConvertFToU_Ruint2_sat_rtn(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint2_sat_rtn(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint32_t
__spirv_ConvertFToU_Ruint2_sat_rtp(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint2_sat_rtp(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint32_t
__spirv_ConvertFToU_Ruint2_sat_rtz(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint2_sat_rtz(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint32_t
__spirv_ConvertFToU_Ruint3(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint3(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint32_t
__spirv_ConvertFToU_Ruint3_rte(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint3_rte(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint32_t
__spirv_ConvertFToU_Ruint3_rtn(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint3_rtn(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint32_t
__spirv_ConvertFToU_Ruint3_rtp(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint3_rtp(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint32_t
__spirv_ConvertFToU_Ruint3_rtz(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint3_rtz(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint32_t
__spirv_ConvertFToU_Ruint3_sat(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint3_sat(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint32_t
__spirv_ConvertFToU_Ruint3_sat_rte(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint3_sat_rte(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint32_t
__spirv_ConvertFToU_Ruint3_sat_rtn(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint3_sat_rtn(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint32_t
__spirv_ConvertFToU_Ruint3_sat_rtp(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint3_sat_rtp(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint32_t
__spirv_ConvertFToU_Ruint3_sat_rtz(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint3_sat_rtz(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint32_t
__spirv_ConvertFToU_Ruint4(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint4(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint32_t
__spirv_ConvertFToU_Ruint4_rte(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint4_rte(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint32_t
__spirv_ConvertFToU_Ruint4_rtn(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint4_rtn(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint32_t
__spirv_ConvertFToU_Ruint4_rtp(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint4_rtp(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint32_t
__spirv_ConvertFToU_Ruint4_rtz(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint4_rtz(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint32_t
__spirv_ConvertFToU_Ruint4_sat(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint4_sat(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint32_t
__spirv_ConvertFToU_Ruint4_sat_rte(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint4_sat_rte(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint32_t
__spirv_ConvertFToU_Ruint4_sat_rtn(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint4_sat_rtn(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint32_t
__spirv_ConvertFToU_Ruint4_sat_rtp(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint4_sat_rtp(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint32_t
__spirv_ConvertFToU_Ruint4_sat_rtz(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint4_sat_rtz(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint32_t
__spirv_ConvertFToU_Ruint8(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint8(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint32_t
__spirv_ConvertFToU_Ruint8_rte(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint8_rte(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint32_t
__spirv_ConvertFToU_Ruint8_rtn(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint8_rtn(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint32_t
__spirv_ConvertFToU_Ruint8_rtp(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint8_rtp(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint32_t
__spirv_ConvertFToU_Ruint8_rtz(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint8_rtz(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint32_t
__spirv_ConvertFToU_Ruint8_sat(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint8_sat(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint32_t
__spirv_ConvertFToU_Ruint8_sat_rte(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint8_sat_rte(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint32_t
__spirv_ConvertFToU_Ruint8_sat_rtn(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint8_sat_rtn(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint32_t
__spirv_ConvertFToU_Ruint8_sat_rtp(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint8_sat_rtp(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint32_t
__spirv_ConvertFToU_Ruint8_sat_rtz(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint8_sat_rtz(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint32_t
__spirv_ConvertFToU_Ruint_rte(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint_rte(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint32_t
__spirv_ConvertFToU_Ruint_rtn(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint_rtn(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint32_t
__spirv_ConvertFToU_Ruint_rtp(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint_rtp(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint32_t
__spirv_ConvertFToU_Ruint_rtz(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint_rtz(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint32_t
__spirv_ConvertFToU_Ruint_sat(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint_sat(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint32_t
__spirv_ConvertFToU_Ruint_sat_rte(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint_sat_rte(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint32_t
__spirv_ConvertFToU_Ruint_sat_rtn(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint_sat_rtn(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint32_t
__spirv_ConvertFToU_Ruint_sat_rtp(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint_sat_rtp(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint32_t
__spirv_ConvertFToU_Ruint_sat_rtz(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Ruint_sat_rtz(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint64_t
__spirv_ConvertFToU_Rulong(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint64_t
__spirv_ConvertFToU_Rulong16(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong16(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint64_t
__spirv_ConvertFToU_Rulong16_rte(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong16_rte(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint64_t
__spirv_ConvertFToU_Rulong16_rtn(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong16_rtn(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint64_t
__spirv_ConvertFToU_Rulong16_rtp(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong16_rtp(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint64_t
__spirv_ConvertFToU_Rulong16_rtz(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong16_rtz(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint64_t
__spirv_ConvertFToU_Rulong16_sat(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong16_sat(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint64_t
__spirv_ConvertFToU_Rulong16_sat_rte(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong16_sat_rte(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint64_t
__spirv_ConvertFToU_Rulong16_sat_rtn(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong16_sat_rtn(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint64_t
__spirv_ConvertFToU_Rulong16_sat_rtp(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong16_sat_rtp(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint64_t
__spirv_ConvertFToU_Rulong16_sat_rtz(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong16_sat_rtz(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint64_t
__spirv_ConvertFToU_Rulong2(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong2(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint64_t
__spirv_ConvertFToU_Rulong2_rte(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong2_rte(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint64_t
__spirv_ConvertFToU_Rulong2_rtn(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong2_rtn(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint64_t
__spirv_ConvertFToU_Rulong2_rtp(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong2_rtp(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint64_t
__spirv_ConvertFToU_Rulong2_rtz(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong2_rtz(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint64_t
__spirv_ConvertFToU_Rulong2_sat(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong2_sat(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint64_t
__spirv_ConvertFToU_Rulong2_sat_rte(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong2_sat_rte(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint64_t
__spirv_ConvertFToU_Rulong2_sat_rtn(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong2_sat_rtn(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint64_t
__spirv_ConvertFToU_Rulong2_sat_rtp(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong2_sat_rtp(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint64_t
__spirv_ConvertFToU_Rulong2_sat_rtz(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong2_sat_rtz(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint64_t
__spirv_ConvertFToU_Rulong3(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong3(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint64_t
__spirv_ConvertFToU_Rulong3_rte(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong3_rte(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint64_t
__spirv_ConvertFToU_Rulong3_rtn(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong3_rtn(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint64_t
__spirv_ConvertFToU_Rulong3_rtp(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong3_rtp(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint64_t
__spirv_ConvertFToU_Rulong3_rtz(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong3_rtz(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint64_t
__spirv_ConvertFToU_Rulong3_sat(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong3_sat(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint64_t
__spirv_ConvertFToU_Rulong3_sat_rte(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong3_sat_rte(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint64_t
__spirv_ConvertFToU_Rulong3_sat_rtn(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong3_sat_rtn(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint64_t
__spirv_ConvertFToU_Rulong3_sat_rtp(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong3_sat_rtp(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint64_t
__spirv_ConvertFToU_Rulong3_sat_rtz(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong3_sat_rtz(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint64_t
__spirv_ConvertFToU_Rulong4(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong4(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint64_t
__spirv_ConvertFToU_Rulong4_rte(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong4_rte(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint64_t
__spirv_ConvertFToU_Rulong4_rtn(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong4_rtn(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint64_t
__spirv_ConvertFToU_Rulong4_rtp(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong4_rtp(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint64_t
__spirv_ConvertFToU_Rulong4_rtz(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong4_rtz(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint64_t
__spirv_ConvertFToU_Rulong4_sat(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong4_sat(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint64_t
__spirv_ConvertFToU_Rulong4_sat_rte(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong4_sat_rte(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint64_t
__spirv_ConvertFToU_Rulong4_sat_rtn(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong4_sat_rtn(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint64_t
__spirv_ConvertFToU_Rulong4_sat_rtp(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong4_sat_rtp(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint64_t
__spirv_ConvertFToU_Rulong4_sat_rtz(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong4_sat_rtz(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint64_t
__spirv_ConvertFToU_Rulong8(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong8(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint64_t
__spirv_ConvertFToU_Rulong8_rte(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong8_rte(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint64_t
__spirv_ConvertFToU_Rulong8_rtn(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong8_rtn(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint64_t
__spirv_ConvertFToU_Rulong8_rtp(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong8_rtp(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint64_t
__spirv_ConvertFToU_Rulong8_rtz(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong8_rtz(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint64_t
__spirv_ConvertFToU_Rulong8_sat(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong8_sat(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint64_t
__spirv_ConvertFToU_Rulong8_sat_rte(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong8_sat_rte(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint64_t
__spirv_ConvertFToU_Rulong8_sat_rtn(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong8_sat_rtn(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint64_t
__spirv_ConvertFToU_Rulong8_sat_rtp(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong8_sat_rtp(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint64_t
__spirv_ConvertFToU_Rulong8_sat_rtz(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong8_sat_rtz(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint64_t
__spirv_ConvertFToU_Rulong_rte(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong_rte(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint64_t
__spirv_ConvertFToU_Rulong_rtn(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong_rtn(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint64_t
__spirv_ConvertFToU_Rulong_rtp(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong_rtp(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint64_t
__spirv_ConvertFToU_Rulong_rtz(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong_rtz(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint64_t
__spirv_ConvertFToU_Rulong_sat(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong_sat(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint64_t
__spirv_ConvertFToU_Rulong_sat_rte(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong_sat_rte(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint64_t
__spirv_ConvertFToU_Rulong_sat_rtn(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong_sat_rtn(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint64_t
__spirv_ConvertFToU_Rulong_sat_rtp(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong_sat_rtp(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint64_t
__spirv_ConvertFToU_Rulong_sat_rtz(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Rulong_sat_rtz(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint16_t
__spirv_ConvertFToU_Rushort(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint16_t
__spirv_ConvertFToU_Rushort16(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort16(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint16_t
__spirv_ConvertFToU_Rushort16_rte(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort16_rte(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint16_t
__spirv_ConvertFToU_Rushort16_rtn(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort16_rtn(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint16_t
__spirv_ConvertFToU_Rushort16_rtp(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort16_rtp(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint16_t
__spirv_ConvertFToU_Rushort16_rtz(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort16_rtz(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint16_t
__spirv_ConvertFToU_Rushort16_sat(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort16_sat(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint16_t
__spirv_ConvertFToU_Rushort16_sat_rte(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort16_sat_rte(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint16_t
__spirv_ConvertFToU_Rushort16_sat_rtn(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort16_sat_rtn(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint16_t
__spirv_ConvertFToU_Rushort16_sat_rtp(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort16_sat_rtp(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_uint16_t
__spirv_ConvertFToU_Rushort16_sat_rtz(__clc_vec16_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort16_sat_rtz(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint16_t
__spirv_ConvertFToU_Rushort2(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort2(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint16_t
__spirv_ConvertFToU_Rushort2_rte(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort2_rte(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint16_t
__spirv_ConvertFToU_Rushort2_rtn(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort2_rtn(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint16_t
__spirv_ConvertFToU_Rushort2_rtp(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort2_rtp(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint16_t
__spirv_ConvertFToU_Rushort2_rtz(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort2_rtz(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint16_t
__spirv_ConvertFToU_Rushort2_sat(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort2_sat(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint16_t
__spirv_ConvertFToU_Rushort2_sat_rte(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort2_sat_rte(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint16_t
__spirv_ConvertFToU_Rushort2_sat_rtn(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort2_sat_rtn(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint16_t
__spirv_ConvertFToU_Rushort2_sat_rtp(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort2_sat_rtp(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_uint16_t
__spirv_ConvertFToU_Rushort2_sat_rtz(__clc_vec2_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort2_sat_rtz(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint16_t
__spirv_ConvertFToU_Rushort3(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort3(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint16_t
__spirv_ConvertFToU_Rushort3_rte(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort3_rte(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint16_t
__spirv_ConvertFToU_Rushort3_rtn(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort3_rtn(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint16_t
__spirv_ConvertFToU_Rushort3_rtp(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort3_rtp(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint16_t
__spirv_ConvertFToU_Rushort3_rtz(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort3_rtz(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint16_t
__spirv_ConvertFToU_Rushort3_sat(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort3_sat(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint16_t
__spirv_ConvertFToU_Rushort3_sat_rte(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort3_sat_rte(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint16_t
__spirv_ConvertFToU_Rushort3_sat_rtn(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort3_sat_rtn(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint16_t
__spirv_ConvertFToU_Rushort3_sat_rtp(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort3_sat_rtp(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_uint16_t
__spirv_ConvertFToU_Rushort3_sat_rtz(__clc_vec3_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort3_sat_rtz(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint16_t
__spirv_ConvertFToU_Rushort4(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort4(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint16_t
__spirv_ConvertFToU_Rushort4_rte(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort4_rte(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint16_t
__spirv_ConvertFToU_Rushort4_rtn(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort4_rtn(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint16_t
__spirv_ConvertFToU_Rushort4_rtp(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort4_rtp(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint16_t
__spirv_ConvertFToU_Rushort4_rtz(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort4_rtz(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint16_t
__spirv_ConvertFToU_Rushort4_sat(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort4_sat(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint16_t
__spirv_ConvertFToU_Rushort4_sat_rte(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort4_sat_rte(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint16_t
__spirv_ConvertFToU_Rushort4_sat_rtn(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort4_sat_rtn(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint16_t
__spirv_ConvertFToU_Rushort4_sat_rtp(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort4_sat_rtp(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_uint16_t
__spirv_ConvertFToU_Rushort4_sat_rtz(__clc_vec4_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort4_sat_rtz(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint16_t
__spirv_ConvertFToU_Rushort8(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort8(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint16_t
__spirv_ConvertFToU_Rushort8_rte(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort8_rte(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint16_t
__spirv_ConvertFToU_Rushort8_rtn(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort8_rtn(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint16_t
__spirv_ConvertFToU_Rushort8_rtp(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort8_rtp(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint16_t
__spirv_ConvertFToU_Rushort8_rtz(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort8_rtz(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint16_t
__spirv_ConvertFToU_Rushort8_sat(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort8_sat(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint16_t
__spirv_ConvertFToU_Rushort8_sat_rte(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort8_sat_rte(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint16_t
__spirv_ConvertFToU_Rushort8_sat_rtn(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort8_sat_rtn(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint16_t
__spirv_ConvertFToU_Rushort8_sat_rtp(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort8_sat_rtp(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_uint16_t
__spirv_ConvertFToU_Rushort8_sat_rtz(__clc_vec8_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort8_sat_rtz(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint16_t
__spirv_ConvertFToU_Rushort_rte(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort_rte(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint16_t
__spirv_ConvertFToU_Rushort_rtn(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort_rtn(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint16_t
__spirv_ConvertFToU_Rushort_rtp(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort_rtp(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint16_t
__spirv_ConvertFToU_Rushort_rtz(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort_rtz(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint16_t
__spirv_ConvertFToU_Rushort_sat(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort_sat(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint16_t
__spirv_ConvertFToU_Rushort_sat_rte(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort_sat_rte(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint16_t
__spirv_ConvertFToU_Rushort_sat_rtn(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort_sat_rtn(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint16_t
__spirv_ConvertFToU_Rushort_sat_rtp(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort_sat_rtp(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_uint16_t
__spirv_ConvertFToU_Rushort_sat_rtz(__clc_float16_t args_0) {
  return __spirv_ConvertFToU_Rushort_sat_rtz(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_Dot(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_Dot(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_Dot(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_Dot(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_Dot(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_Dot(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_Dot(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_Dot(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_Dot(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1) {
  return __spirv_Dot(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp64_t
__spirv_FConvert_Rdouble(__clc_float16_t args_0) {
  return __spirv_FConvert_Rdouble(__clc_as_half(args_0));
}

#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp64_t
__spirv_FConvert_Rdouble16(__clc_vec16_float16_t args_0) {
  return __spirv_FConvert_Rdouble16(__clc_as_half16(args_0));
}

#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp64_t
__spirv_FConvert_Rdouble16_rte(__clc_vec16_float16_t args_0) {
  return __spirv_FConvert_Rdouble16_rte(__clc_as_half16(args_0));
}

#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp64_t
__spirv_FConvert_Rdouble16_rtn(__clc_vec16_float16_t args_0) {
  return __spirv_FConvert_Rdouble16_rtn(__clc_as_half16(args_0));
}

#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp64_t
__spirv_FConvert_Rdouble16_rtp(__clc_vec16_float16_t args_0) {
  return __spirv_FConvert_Rdouble16_rtp(__clc_as_half16(args_0));
}

#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp64_t
__spirv_FConvert_Rdouble16_rtz(__clc_vec16_float16_t args_0) {
  return __spirv_FConvert_Rdouble16_rtz(__clc_as_half16(args_0));
}

#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp64_t
__spirv_FConvert_Rdouble2(__clc_vec2_float16_t args_0) {
  return __spirv_FConvert_Rdouble2(__clc_as_half2(args_0));
}

#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp64_t
__spirv_FConvert_Rdouble2_rte(__clc_vec2_float16_t args_0) {
  return __spirv_FConvert_Rdouble2_rte(__clc_as_half2(args_0));
}

#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp64_t
__spirv_FConvert_Rdouble2_rtn(__clc_vec2_float16_t args_0) {
  return __spirv_FConvert_Rdouble2_rtn(__clc_as_half2(args_0));
}

#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp64_t
__spirv_FConvert_Rdouble2_rtp(__clc_vec2_float16_t args_0) {
  return __spirv_FConvert_Rdouble2_rtp(__clc_as_half2(args_0));
}

#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp64_t
__spirv_FConvert_Rdouble2_rtz(__clc_vec2_float16_t args_0) {
  return __spirv_FConvert_Rdouble2_rtz(__clc_as_half2(args_0));
}

#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp64_t
__spirv_FConvert_Rdouble3(__clc_vec3_float16_t args_0) {
  return __spirv_FConvert_Rdouble3(__clc_as_half3(args_0));
}

#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp64_t
__spirv_FConvert_Rdouble3_rte(__clc_vec3_float16_t args_0) {
  return __spirv_FConvert_Rdouble3_rte(__clc_as_half3(args_0));
}

#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp64_t
__spirv_FConvert_Rdouble3_rtn(__clc_vec3_float16_t args_0) {
  return __spirv_FConvert_Rdouble3_rtn(__clc_as_half3(args_0));
}

#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp64_t
__spirv_FConvert_Rdouble3_rtp(__clc_vec3_float16_t args_0) {
  return __spirv_FConvert_Rdouble3_rtp(__clc_as_half3(args_0));
}

#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp64_t
__spirv_FConvert_Rdouble3_rtz(__clc_vec3_float16_t args_0) {
  return __spirv_FConvert_Rdouble3_rtz(__clc_as_half3(args_0));
}

#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp64_t
__spirv_FConvert_Rdouble4(__clc_vec4_float16_t args_0) {
  return __spirv_FConvert_Rdouble4(__clc_as_half4(args_0));
}

#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp64_t
__spirv_FConvert_Rdouble4_rte(__clc_vec4_float16_t args_0) {
  return __spirv_FConvert_Rdouble4_rte(__clc_as_half4(args_0));
}

#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp64_t
__spirv_FConvert_Rdouble4_rtn(__clc_vec4_float16_t args_0) {
  return __spirv_FConvert_Rdouble4_rtn(__clc_as_half4(args_0));
}

#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp64_t
__spirv_FConvert_Rdouble4_rtp(__clc_vec4_float16_t args_0) {
  return __spirv_FConvert_Rdouble4_rtp(__clc_as_half4(args_0));
}

#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp64_t
__spirv_FConvert_Rdouble4_rtz(__clc_vec4_float16_t args_0) {
  return __spirv_FConvert_Rdouble4_rtz(__clc_as_half4(args_0));
}

#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp64_t
__spirv_FConvert_Rdouble8(__clc_vec8_float16_t args_0) {
  return __spirv_FConvert_Rdouble8(__clc_as_half8(args_0));
}

#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp64_t
__spirv_FConvert_Rdouble8_rte(__clc_vec8_float16_t args_0) {
  return __spirv_FConvert_Rdouble8_rte(__clc_as_half8(args_0));
}

#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp64_t
__spirv_FConvert_Rdouble8_rtn(__clc_vec8_float16_t args_0) {
  return __spirv_FConvert_Rdouble8_rtn(__clc_as_half8(args_0));
}

#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp64_t
__spirv_FConvert_Rdouble8_rtp(__clc_vec8_float16_t args_0) {
  return __spirv_FConvert_Rdouble8_rtp(__clc_as_half8(args_0));
}

#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp64_t
__spirv_FConvert_Rdouble8_rtz(__clc_vec8_float16_t args_0) {
  return __spirv_FConvert_Rdouble8_rtz(__clc_as_half8(args_0));
}

#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp64_t
__spirv_FConvert_Rdouble_rte(__clc_float16_t args_0) {
  return __spirv_FConvert_Rdouble_rte(__clc_as_half(args_0));
}

#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp64_t
__spirv_FConvert_Rdouble_rtn(__clc_float16_t args_0) {
  return __spirv_FConvert_Rdouble_rtn(__clc_as_half(args_0));
}

#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp64_t
__spirv_FConvert_Rdouble_rtp(__clc_float16_t args_0) {
  return __spirv_FConvert_Rdouble_rtp(__clc_as_half(args_0));
}

#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp64_t
__spirv_FConvert_Rdouble_rtz(__clc_float16_t args_0) {
  return __spirv_FConvert_Rdouble_rtz(__clc_as_half(args_0));
}

#endif

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp32_t
__spirv_FConvert_Rfloat(__clc_float16_t args_0) {
  return __spirv_FConvert_Rfloat(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp32_t
__spirv_FConvert_Rfloat16(__clc_vec16_float16_t args_0) {
  return __spirv_FConvert_Rfloat16(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp32_t
__spirv_FConvert_Rfloat16_rte(__clc_vec16_float16_t args_0) {
  return __spirv_FConvert_Rfloat16_rte(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp32_t
__spirv_FConvert_Rfloat16_rtn(__clc_vec16_float16_t args_0) {
  return __spirv_FConvert_Rfloat16_rtn(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp32_t
__spirv_FConvert_Rfloat16_rtp(__clc_vec16_float16_t args_0) {
  return __spirv_FConvert_Rfloat16_rtp(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp32_t
__spirv_FConvert_Rfloat16_rtz(__clc_vec16_float16_t args_0) {
  return __spirv_FConvert_Rfloat16_rtz(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp32_t
__spirv_FConvert_Rfloat2(__clc_vec2_float16_t args_0) {
  return __spirv_FConvert_Rfloat2(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp32_t
__spirv_FConvert_Rfloat2_rte(__clc_vec2_float16_t args_0) {
  return __spirv_FConvert_Rfloat2_rte(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp32_t
__spirv_FConvert_Rfloat2_rtn(__clc_vec2_float16_t args_0) {
  return __spirv_FConvert_Rfloat2_rtn(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp32_t
__spirv_FConvert_Rfloat2_rtp(__clc_vec2_float16_t args_0) {
  return __spirv_FConvert_Rfloat2_rtp(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp32_t
__spirv_FConvert_Rfloat2_rtz(__clc_vec2_float16_t args_0) {
  return __spirv_FConvert_Rfloat2_rtz(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp32_t
__spirv_FConvert_Rfloat3(__clc_vec3_float16_t args_0) {
  return __spirv_FConvert_Rfloat3(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp32_t
__spirv_FConvert_Rfloat3_rte(__clc_vec3_float16_t args_0) {
  return __spirv_FConvert_Rfloat3_rte(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp32_t
__spirv_FConvert_Rfloat3_rtn(__clc_vec3_float16_t args_0) {
  return __spirv_FConvert_Rfloat3_rtn(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp32_t
__spirv_FConvert_Rfloat3_rtp(__clc_vec3_float16_t args_0) {
  return __spirv_FConvert_Rfloat3_rtp(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp32_t
__spirv_FConvert_Rfloat3_rtz(__clc_vec3_float16_t args_0) {
  return __spirv_FConvert_Rfloat3_rtz(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp32_t
__spirv_FConvert_Rfloat4(__clc_vec4_float16_t args_0) {
  return __spirv_FConvert_Rfloat4(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp32_t
__spirv_FConvert_Rfloat4_rte(__clc_vec4_float16_t args_0) {
  return __spirv_FConvert_Rfloat4_rte(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp32_t
__spirv_FConvert_Rfloat4_rtn(__clc_vec4_float16_t args_0) {
  return __spirv_FConvert_Rfloat4_rtn(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp32_t
__spirv_FConvert_Rfloat4_rtp(__clc_vec4_float16_t args_0) {
  return __spirv_FConvert_Rfloat4_rtp(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp32_t
__spirv_FConvert_Rfloat4_rtz(__clc_vec4_float16_t args_0) {
  return __spirv_FConvert_Rfloat4_rtz(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp32_t
__spirv_FConvert_Rfloat8(__clc_vec8_float16_t args_0) {
  return __spirv_FConvert_Rfloat8(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp32_t
__spirv_FConvert_Rfloat8_rte(__clc_vec8_float16_t args_0) {
  return __spirv_FConvert_Rfloat8_rte(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp32_t
__spirv_FConvert_Rfloat8_rtn(__clc_vec8_float16_t args_0) {
  return __spirv_FConvert_Rfloat8_rtn(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp32_t
__spirv_FConvert_Rfloat8_rtp(__clc_vec8_float16_t args_0) {
  return __spirv_FConvert_Rfloat8_rtp(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp32_t
__spirv_FConvert_Rfloat8_rtz(__clc_vec8_float16_t args_0) {
  return __spirv_FConvert_Rfloat8_rtz(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp32_t
__spirv_FConvert_Rfloat_rte(__clc_float16_t args_0) {
  return __spirv_FConvert_Rfloat_rte(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp32_t
__spirv_FConvert_Rfloat_rtn(__clc_float16_t args_0) {
  return __spirv_FConvert_Rfloat_rtn(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp32_t
__spirv_FConvert_Rfloat_rtp(__clc_float16_t args_0) {
  return __spirv_FConvert_Rfloat_rtp(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp32_t
__spirv_FConvert_Rfloat_rtz(__clc_float16_t args_0) {
  return __spirv_FConvert_Rfloat_rtz(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_bool_t
__spirv_FOrdEqual(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_FOrdEqual(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t
__spirv_FOrdEqual(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_FOrdEqual(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t
__spirv_FOrdEqual(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_FOrdEqual(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t
__spirv_FOrdEqual(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_FOrdEqual(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t
__spirv_FOrdEqual(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_FOrdEqual(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t
__spirv_FOrdEqual(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1) {
  return __spirv_FOrdEqual(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_bool_t
__spirv_FOrdGreaterThan(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_FOrdGreaterThan(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t __spirv_FOrdGreaterThan(
    __clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_FOrdGreaterThan(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t __spirv_FOrdGreaterThan(
    __clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_FOrdGreaterThan(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t __spirv_FOrdGreaterThan(
    __clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_FOrdGreaterThan(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t __spirv_FOrdGreaterThan(
    __clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_FOrdGreaterThan(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t __spirv_FOrdGreaterThan(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1) {
  return __spirv_FOrdGreaterThan(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_bool_t
__spirv_FOrdGreaterThanEqual(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_FOrdGreaterThanEqual(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t
__spirv_FOrdGreaterThanEqual(__clc_vec2_float16_t args_0,
                             __clc_vec2_float16_t args_1) {
  return __spirv_FOrdGreaterThanEqual(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t
__spirv_FOrdGreaterThanEqual(__clc_vec3_float16_t args_0,
                             __clc_vec3_float16_t args_1) {
  return __spirv_FOrdGreaterThanEqual(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t
__spirv_FOrdGreaterThanEqual(__clc_vec4_float16_t args_0,
                             __clc_vec4_float16_t args_1) {
  return __spirv_FOrdGreaterThanEqual(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t
__spirv_FOrdGreaterThanEqual(__clc_vec8_float16_t args_0,
                             __clc_vec8_float16_t args_1) {
  return __spirv_FOrdGreaterThanEqual(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t
__spirv_FOrdGreaterThanEqual(__clc_vec16_float16_t args_0,
                             __clc_vec16_float16_t args_1) {
  return __spirv_FOrdGreaterThanEqual(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_bool_t
__spirv_FOrdLessThan(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_FOrdLessThan(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t
__spirv_FOrdLessThan(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_FOrdLessThan(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t
__spirv_FOrdLessThan(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_FOrdLessThan(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t
__spirv_FOrdLessThan(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_FOrdLessThan(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t
__spirv_FOrdLessThan(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_FOrdLessThan(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t __spirv_FOrdLessThan(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1) {
  return __spirv_FOrdLessThan(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_bool_t
__spirv_FOrdLessThanEqual(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_FOrdLessThanEqual(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t __spirv_FOrdLessThanEqual(
    __clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_FOrdLessThanEqual(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t __spirv_FOrdLessThanEqual(
    __clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_FOrdLessThanEqual(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t __spirv_FOrdLessThanEqual(
    __clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_FOrdLessThanEqual(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t __spirv_FOrdLessThanEqual(
    __clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_FOrdLessThanEqual(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t
__spirv_FOrdLessThanEqual(__clc_vec16_float16_t args_0,
                          __clc_vec16_float16_t args_1) {
  return __spirv_FOrdLessThanEqual(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_bool_t
__spirv_FOrdNotEqual(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_FOrdNotEqual(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t
__spirv_FOrdNotEqual(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_FOrdNotEqual(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t
__spirv_FOrdNotEqual(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_FOrdNotEqual(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t
__spirv_FOrdNotEqual(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_FOrdNotEqual(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t
__spirv_FOrdNotEqual(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_FOrdNotEqual(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t __spirv_FOrdNotEqual(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1) {
  return __spirv_FOrdNotEqual(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_bool_t
__spirv_FUnordEqual(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_FUnordEqual(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t
__spirv_FUnordEqual(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_FUnordEqual(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t
__spirv_FUnordEqual(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_FUnordEqual(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t
__spirv_FUnordEqual(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_FUnordEqual(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t
__spirv_FUnordEqual(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_FUnordEqual(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t __spirv_FUnordEqual(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1) {
  return __spirv_FUnordEqual(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_bool_t
__spirv_FUnordGreaterThan(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_FUnordGreaterThan(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t __spirv_FUnordGreaterThan(
    __clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_FUnordGreaterThan(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t __spirv_FUnordGreaterThan(
    __clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_FUnordGreaterThan(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t __spirv_FUnordGreaterThan(
    __clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_FUnordGreaterThan(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t __spirv_FUnordGreaterThan(
    __clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_FUnordGreaterThan(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t
__spirv_FUnordGreaterThan(__clc_vec16_float16_t args_0,
                          __clc_vec16_float16_t args_1) {
  return __spirv_FUnordGreaterThan(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_bool_t
__spirv_FUnordGreaterThanEqual(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_FUnordGreaterThanEqual(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t
__spirv_FUnordGreaterThanEqual(__clc_vec2_float16_t args_0,
                               __clc_vec2_float16_t args_1) {
  return __spirv_FUnordGreaterThanEqual(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t
__spirv_FUnordGreaterThanEqual(__clc_vec3_float16_t args_0,
                               __clc_vec3_float16_t args_1) {
  return __spirv_FUnordGreaterThanEqual(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t
__spirv_FUnordGreaterThanEqual(__clc_vec4_float16_t args_0,
                               __clc_vec4_float16_t args_1) {
  return __spirv_FUnordGreaterThanEqual(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t
__spirv_FUnordGreaterThanEqual(__clc_vec8_float16_t args_0,
                               __clc_vec8_float16_t args_1) {
  return __spirv_FUnordGreaterThanEqual(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t
__spirv_FUnordGreaterThanEqual(__clc_vec16_float16_t args_0,
                               __clc_vec16_float16_t args_1) {
  return __spirv_FUnordGreaterThanEqual(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_bool_t
__spirv_FUnordLessThan(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_FUnordLessThan(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t __spirv_FUnordLessThan(
    __clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_FUnordLessThan(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t __spirv_FUnordLessThan(
    __clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_FUnordLessThan(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t __spirv_FUnordLessThan(
    __clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_FUnordLessThan(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t __spirv_FUnordLessThan(
    __clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_FUnordLessThan(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t __spirv_FUnordLessThan(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1) {
  return __spirv_FUnordLessThan(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_bool_t
__spirv_FUnordLessThanEqual(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_FUnordLessThanEqual(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t
__spirv_FUnordLessThanEqual(__clc_vec2_float16_t args_0,
                            __clc_vec2_float16_t args_1) {
  return __spirv_FUnordLessThanEqual(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t
__spirv_FUnordLessThanEqual(__clc_vec3_float16_t args_0,
                            __clc_vec3_float16_t args_1) {
  return __spirv_FUnordLessThanEqual(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t
__spirv_FUnordLessThanEqual(__clc_vec4_float16_t args_0,
                            __clc_vec4_float16_t args_1) {
  return __spirv_FUnordLessThanEqual(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t
__spirv_FUnordLessThanEqual(__clc_vec8_float16_t args_0,
                            __clc_vec8_float16_t args_1) {
  return __spirv_FUnordLessThanEqual(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t
__spirv_FUnordLessThanEqual(__clc_vec16_float16_t args_0,
                            __clc_vec16_float16_t args_1) {
  return __spirv_FUnordLessThanEqual(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_bool_t
__spirv_FUnordNotEqual(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_FUnordNotEqual(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t __spirv_FUnordNotEqual(
    __clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_FUnordNotEqual(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t __spirv_FUnordNotEqual(
    __clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_FUnordNotEqual(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t __spirv_FUnordNotEqual(
    __clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_FUnordNotEqual(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t __spirv_FUnordNotEqual(
    __clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_FUnordNotEqual(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t __spirv_FUnordNotEqual(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1) {
  return __spirv_FUnordNotEqual(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t args_0, __clc_float16_t __local *args_1,
    __clc_float16_t const __global *args_2, __clc_size_t args_3,
    __clc_size_t args_4, __clc_event_t args_5) {
  return __spirv_GroupAsyncCopy(args_0, (__clc_fp16_t __local *)(args_1),
                                (__clc_fp16_t const __global *)(args_2), args_3,
                                args_4, args_5);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t args_0, __clc_float16_t __global *args_1,
    __clc_float16_t const __local *args_2, __clc_size_t args_3,
    __clc_size_t args_4, __clc_event_t args_5) {
  return __spirv_GroupAsyncCopy(args_0, (__clc_fp16_t __global *)(args_1),
                                (__clc_fp16_t const __local *)(args_2), args_3,
                                args_4, args_5);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t args_0, __clc_vec2_float16_t __local *args_1,
    __clc_vec2_float16_t const __global *args_2, __clc_size_t args_3,
    __clc_size_t args_4, __clc_event_t args_5) {
  return __spirv_GroupAsyncCopy(args_0, (__clc_vec2_fp16_t __local *)(args_1),
                                (__clc_vec2_fp16_t const __global *)(args_2),
                                args_3, args_4, args_5);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t args_0, __clc_vec2_float16_t __global *args_1,
    __clc_vec2_float16_t const __local *args_2, __clc_size_t args_3,
    __clc_size_t args_4, __clc_event_t args_5) {
  return __spirv_GroupAsyncCopy(args_0, (__clc_vec2_fp16_t __global *)(args_1),
                                (__clc_vec2_fp16_t const __local *)(args_2),
                                args_3, args_4, args_5);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t args_0, __clc_vec3_float16_t __local *args_1,
    __clc_vec3_float16_t const __global *args_2, __clc_size_t args_3,
    __clc_size_t args_4, __clc_event_t args_5) {
  return __spirv_GroupAsyncCopy(args_0, (__clc_vec3_fp16_t __local *)(args_1),
                                (__clc_vec3_fp16_t const __global *)(args_2),
                                args_3, args_4, args_5);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t args_0, __clc_vec3_float16_t __global *args_1,
    __clc_vec3_float16_t const __local *args_2, __clc_size_t args_3,
    __clc_size_t args_4, __clc_event_t args_5) {
  return __spirv_GroupAsyncCopy(args_0, (__clc_vec3_fp16_t __global *)(args_1),
                                (__clc_vec3_fp16_t const __local *)(args_2),
                                args_3, args_4, args_5);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t args_0, __clc_vec4_float16_t __local *args_1,
    __clc_vec4_float16_t const __global *args_2, __clc_size_t args_3,
    __clc_size_t args_4, __clc_event_t args_5) {
  return __spirv_GroupAsyncCopy(args_0, (__clc_vec4_fp16_t __local *)(args_1),
                                (__clc_vec4_fp16_t const __global *)(args_2),
                                args_3, args_4, args_5);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t args_0, __clc_vec4_float16_t __global *args_1,
    __clc_vec4_float16_t const __local *args_2, __clc_size_t args_3,
    __clc_size_t args_4, __clc_event_t args_5) {
  return __spirv_GroupAsyncCopy(args_0, (__clc_vec4_fp16_t __global *)(args_1),
                                (__clc_vec4_fp16_t const __local *)(args_2),
                                args_3, args_4, args_5);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t args_0, __clc_vec8_float16_t __local *args_1,
    __clc_vec8_float16_t const __global *args_2, __clc_size_t args_3,
    __clc_size_t args_4, __clc_event_t args_5) {
  return __spirv_GroupAsyncCopy(args_0, (__clc_vec8_fp16_t __local *)(args_1),
                                (__clc_vec8_fp16_t const __global *)(args_2),
                                args_3, args_4, args_5);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t args_0, __clc_vec8_float16_t __global *args_1,
    __clc_vec8_float16_t const __local *args_2, __clc_size_t args_3,
    __clc_size_t args_4, __clc_event_t args_5) {
  return __spirv_GroupAsyncCopy(args_0, (__clc_vec8_fp16_t __global *)(args_1),
                                (__clc_vec8_fp16_t const __local *)(args_2),
                                args_3, args_4, args_5);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t args_0, __clc_vec16_float16_t __local *args_1,
    __clc_vec16_float16_t const __global *args_2, __clc_size_t args_3,
    __clc_size_t args_4, __clc_event_t args_5) {
  return __spirv_GroupAsyncCopy(args_0, (__clc_vec16_fp16_t __local *)(args_1),
                                (__clc_vec16_fp16_t const __global *)(args_2),
                                args_3, args_4, args_5);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t args_0, __clc_vec16_float16_t __global *args_1,
    __clc_vec16_float16_t const __local *args_2, __clc_size_t args_3,
    __clc_size_t args_4, __clc_event_t args_5) {
  return __spirv_GroupAsyncCopy(args_0, (__clc_vec16_fp16_t __global *)(args_1),
                                (__clc_vec16_fp16_t const __local *)(args_2),
                                args_3, args_4, args_5);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_bool_t
__spirv_IsFinite(__clc_float16_t args_0) {
  return __spirv_IsFinite(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t
__spirv_IsFinite(__clc_vec2_float16_t args_0) {
  return __spirv_IsFinite(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t
__spirv_IsFinite(__clc_vec3_float16_t args_0) {
  return __spirv_IsFinite(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t
__spirv_IsFinite(__clc_vec4_float16_t args_0) {
  return __spirv_IsFinite(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t
__spirv_IsFinite(__clc_vec8_float16_t args_0) {
  return __spirv_IsFinite(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t
__spirv_IsFinite(__clc_vec16_float16_t args_0) {
  return __spirv_IsFinite(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_bool_t
__spirv_IsInf(__clc_float16_t args_0) {
  return __spirv_IsInf(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t
__spirv_IsInf(__clc_vec2_float16_t args_0) {
  return __spirv_IsInf(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t
__spirv_IsInf(__clc_vec3_float16_t args_0) {
  return __spirv_IsInf(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t
__spirv_IsInf(__clc_vec4_float16_t args_0) {
  return __spirv_IsInf(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t
__spirv_IsInf(__clc_vec8_float16_t args_0) {
  return __spirv_IsInf(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t
__spirv_IsInf(__clc_vec16_float16_t args_0) {
  return __spirv_IsInf(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_bool_t
__spirv_IsNan(__clc_float16_t args_0) {
  return __spirv_IsNan(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t
__spirv_IsNan(__clc_vec2_float16_t args_0) {
  return __spirv_IsNan(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t
__spirv_IsNan(__clc_vec3_float16_t args_0) {
  return __spirv_IsNan(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t
__spirv_IsNan(__clc_vec4_float16_t args_0) {
  return __spirv_IsNan(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t
__spirv_IsNan(__clc_vec8_float16_t args_0) {
  return __spirv_IsNan(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t
__spirv_IsNan(__clc_vec16_float16_t args_0) {
  return __spirv_IsNan(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_bool_t
__spirv_IsNormal(__clc_float16_t args_0) {
  return __spirv_IsNormal(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t
__spirv_IsNormal(__clc_vec2_float16_t args_0) {
  return __spirv_IsNormal(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t
__spirv_IsNormal(__clc_vec3_float16_t args_0) {
  return __spirv_IsNormal(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t
__spirv_IsNormal(__clc_vec4_float16_t args_0) {
  return __spirv_IsNormal(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t
__spirv_IsNormal(__clc_vec8_float16_t args_0) {
  return __spirv_IsNormal(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t
__spirv_IsNormal(__clc_vec16_float16_t args_0) {
  return __spirv_IsNormal(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_bool_t
__spirv_LessOrGreater(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_LessOrGreater(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t __spirv_LessOrGreater(
    __clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_LessOrGreater(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t __spirv_LessOrGreater(
    __clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_LessOrGreater(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t __spirv_LessOrGreater(
    __clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_LessOrGreater(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t __spirv_LessOrGreater(
    __clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_LessOrGreater(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t __spirv_LessOrGreater(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1) {
  return __spirv_LessOrGreater(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_bool_t
__spirv_Ordered(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_Ordered(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t
__spirv_Ordered(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_Ordered(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t
__spirv_Ordered(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_Ordered(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t
__spirv_Ordered(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_Ordered(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t
__spirv_Ordered(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_Ordered(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t
__spirv_Ordered(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1) {
  return __spirv_Ordered(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_bool_t
__spirv_SignBitSet(__clc_float16_t args_0) {
  return __spirv_SignBitSet(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t
__spirv_SignBitSet(__clc_vec2_float16_t args_0) {
  return __spirv_SignBitSet(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t
__spirv_SignBitSet(__clc_vec3_float16_t args_0) {
  return __spirv_SignBitSet(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t
__spirv_SignBitSet(__clc_vec4_float16_t args_0) {
  return __spirv_SignBitSet(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t
__spirv_SignBitSet(__clc_vec8_float16_t args_0) {
  return __spirv_SignBitSet(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t
__spirv_SignBitSet(__clc_vec16_float16_t args_0) {
  return __spirv_SignBitSet(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_bool_t
__spirv_Unordered(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_Unordered(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int8_t
__spirv_Unordered(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_Unordered(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int8_t
__spirv_Unordered(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_Unordered(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int8_t
__spirv_Unordered(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_Unordered(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int8_t
__spirv_Unordered(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_Unordered(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int8_t
__spirv_Unordered(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1) {
  return __spirv_Unordered(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_VectorTimesScalar(__clc_vec2_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_VectorTimesScalar(__clc_as_half2(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_VectorTimesScalar(__clc_vec3_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_VectorTimesScalar(__clc_as_half3(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_VectorTimesScalar(__clc_vec4_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_VectorTimesScalar(__clc_as_half4(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_VectorTimesScalar(__clc_vec8_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_VectorTimesScalar(__clc_as_half8(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_VectorTimesScalar(__clc_vec16_float16_t args_0,
                          __clc_float16_t args_1) {
  return __spirv_VectorTimesScalar(__clc_as_half16(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_acos(__clc_float16_t args_0) {
  return __spirv_ocl_acos(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_acos(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_acos(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_acos(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_acos(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_acos(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_acos(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_acos(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_acos(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_acos(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_acos(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_acosh(__clc_float16_t args_0) {
  return __spirv_ocl_acosh(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_acosh(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_acosh(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_acosh(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_acosh(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_acosh(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_acosh(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_acosh(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_acosh(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_acosh(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_acosh(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_acospi(__clc_float16_t args_0) {
  return __spirv_ocl_acospi(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_acospi(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_acospi(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_acospi(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_acospi(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_acospi(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_acospi(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_acospi(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_acospi(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_acospi(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_acospi(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_asin(__clc_float16_t args_0) {
  return __spirv_ocl_asin(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_asin(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_asin(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_asin(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_asin(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_asin(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_asin(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_asin(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_asin(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_asin(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_asin(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_asinh(__clc_float16_t args_0) {
  return __spirv_ocl_asinh(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_asinh(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_asinh(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_asinh(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_asinh(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_asinh(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_asinh(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_asinh(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_asinh(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_asinh(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_asinh(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_asinpi(__clc_float16_t args_0) {
  return __spirv_ocl_asinpi(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_asinpi(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_asinpi(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_asinpi(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_asinpi(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_asinpi(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_asinpi(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_asinpi(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_asinpi(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_asinpi(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_asinpi(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_atan(__clc_float16_t args_0) {
  return __spirv_ocl_atan(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_atan(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_atan(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_atan(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_atan(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_atan(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_atan(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_atan(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_atan(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_atan(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_atan(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_atan2(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_ocl_atan2(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_atan2(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_ocl_atan2(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_atan2(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_ocl_atan2(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_atan2(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_ocl_atan2(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_atan2(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_ocl_atan2(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_atan2(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1) {
  return __spirv_ocl_atan2(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_atan2pi(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_ocl_atan2pi(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_atan2pi(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_ocl_atan2pi(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_atan2pi(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_ocl_atan2pi(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_atan2pi(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_ocl_atan2pi(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_atan2pi(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_ocl_atan2pi(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t __spirv_ocl_atan2pi(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1) {
  return __spirv_ocl_atan2pi(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_atanh(__clc_float16_t args_0) {
  return __spirv_ocl_atanh(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_atanh(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_atanh(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_atanh(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_atanh(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_atanh(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_atanh(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_atanh(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_atanh(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_atanh(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_atanh(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_atanpi(__clc_float16_t args_0) {
  return __spirv_ocl_atanpi(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_atanpi(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_atanpi(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_atanpi(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_atanpi(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_atanpi(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_atanpi(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_atanpi(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_atanpi(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_atanpi(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_atanpi(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t __spirv_ocl_bitselect(
    __clc_float16_t args_0, __clc_float16_t args_1, __clc_float16_t args_2) {
  return __spirv_ocl_bitselect(__clc_as_half(args_0), __clc_as_half(args_1),
                               __clc_as_half(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_bitselect(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1,
                      __clc_vec2_float16_t args_2) {
  return __spirv_ocl_bitselect(__clc_as_half2(args_0), __clc_as_half2(args_1),
                               __clc_as_half2(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_bitselect(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1,
                      __clc_vec3_float16_t args_2) {
  return __spirv_ocl_bitselect(__clc_as_half3(args_0), __clc_as_half3(args_1),
                               __clc_as_half3(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_bitselect(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1,
                      __clc_vec4_float16_t args_2) {
  return __spirv_ocl_bitselect(__clc_as_half4(args_0), __clc_as_half4(args_1),
                               __clc_as_half4(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_bitselect(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1,
                      __clc_vec8_float16_t args_2) {
  return __spirv_ocl_bitselect(__clc_as_half8(args_0), __clc_as_half8(args_1),
                               __clc_as_half8(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t __spirv_ocl_bitselect(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1,
    __clc_vec16_float16_t args_2) {
  return __spirv_ocl_bitselect(__clc_as_half16(args_0), __clc_as_half16(args_1),
                               __clc_as_half16(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_cbrt(__clc_float16_t args_0) {
  return __spirv_ocl_cbrt(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_cbrt(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_cbrt(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_cbrt(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_cbrt(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_cbrt(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_cbrt(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_cbrt(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_cbrt(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_cbrt(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_cbrt(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_ceil(__clc_float16_t args_0) {
  return __spirv_ocl_ceil(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_ceil(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_ceil(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_ceil(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_ceil(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_ceil(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_ceil(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_ceil(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_ceil(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_ceil(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_ceil(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_copysign(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_ocl_copysign(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_copysign(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_ocl_copysign(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_copysign(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_ocl_copysign(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_copysign(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_ocl_copysign(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_copysign(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_ocl_copysign(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t __spirv_ocl_copysign(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1) {
  return __spirv_ocl_copysign(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_cos(__clc_float16_t args_0) {
  return __spirv_ocl_cos(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_cos(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_cos(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_cos(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_cos(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_cos(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_cos(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_cos(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_cos(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_cos(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_cos(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_cosh(__clc_float16_t args_0) {
  return __spirv_ocl_cosh(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_cosh(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_cosh(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_cosh(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_cosh(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_cosh(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_cosh(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_cosh(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_cosh(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_cosh(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_cosh(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_cospi(__clc_float16_t args_0) {
  return __spirv_ocl_cospi(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_cospi(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_cospi(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_cospi(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_cospi(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_cospi(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_cospi(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_cospi(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_cospi(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_cospi(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_cospi(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_cross(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_ocl_cross(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_cross(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_ocl_cross(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_degrees(__clc_float16_t args_0) {
  return __spirv_ocl_degrees(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_degrees(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_degrees(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_degrees(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_degrees(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_degrees(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_degrees(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_degrees(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_degrees(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_degrees(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_degrees(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_distance(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_ocl_distance(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_distance(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_ocl_distance(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_distance(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_ocl_distance(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_distance(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_ocl_distance(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_erf(__clc_float16_t args_0) {
  return __spirv_ocl_erf(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_erf(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_erf(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_erf(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_erf(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_erf(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_erf(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_erf(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_erf(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_erf(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_erf(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_erfc(__clc_float16_t args_0) {
  return __spirv_ocl_erfc(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_erfc(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_erfc(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_erfc(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_erfc(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_erfc(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_erfc(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_erfc(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_erfc(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_erfc(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_erfc(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_exp(__clc_float16_t args_0) {
  return __spirv_ocl_exp(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_exp(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_exp(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_exp(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_exp(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_exp(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_exp(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_exp(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_exp(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_exp(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_exp(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_exp10(__clc_float16_t args_0) {
  return __spirv_ocl_exp10(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_exp10(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_exp10(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_exp10(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_exp10(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_exp10(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_exp10(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_exp10(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_exp10(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_exp10(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_exp10(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_exp2(__clc_float16_t args_0) {
  return __spirv_ocl_exp2(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_exp2(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_exp2(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_exp2(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_exp2(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_exp2(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_exp2(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_exp2(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_exp2(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_exp2(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_exp2(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__clc_native_exp2(__clc_float16_t args_0) {
  return __clc_native_exp2(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__clc_native_exp2(__clc_vec2_float16_t args_0) {
  return __clc_native_exp2(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__clc_native_exp2(__clc_vec3_float16_t args_0) {
  return __clc_native_exp2(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__clc_native_exp2(__clc_vec4_float16_t args_0) {
  return __clc_native_exp2(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__clc_native_exp2(__clc_vec8_float16_t args_0) {
  return __clc_native_exp2(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__clc_native_exp2(__clc_vec16_float16_t args_0) {
  return __clc_native_exp2(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_expm1(__clc_float16_t args_0) {
  return __spirv_ocl_expm1(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_expm1(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_expm1(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_expm1(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_expm1(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_expm1(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_expm1(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_expm1(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_expm1(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_expm1(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_expm1(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_fabs(__clc_float16_t args_0) {
  return __spirv_ocl_fabs(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_fabs(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_fabs(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_fabs(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_fabs(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_fabs(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_fabs(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_fabs(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_fabs(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_fabs(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_fabs(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t __spirv_ocl_fclamp(
    __clc_float16_t args_0, __clc_float16_t args_1, __clc_float16_t args_2) {
  return __spirv_ocl_fclamp(__clc_as_half(args_0), __clc_as_half(args_1), __clc_as_half(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_fclamp(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1,
                   __clc_vec2_float16_t args_2) {
  return __spirv_ocl_fclamp(__clc_as_half2(args_0), __clc_as_half2(args_1),
                            __clc_as_half2(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_fclamp(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1,
                   __clc_vec3_float16_t args_2) {
  return __spirv_ocl_fclamp(__clc_as_half3(args_0), __clc_as_half3(args_1),
                            __clc_as_half3(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_fclamp(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1,
                   __clc_vec4_float16_t args_2) {
  return __spirv_ocl_fclamp(__clc_as_half4(args_0), __clc_as_half4(args_1),
                            __clc_as_half4(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_fclamp(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1,
                   __clc_vec8_float16_t args_2) {
  return __spirv_ocl_fclamp(__clc_as_half8(args_0), __clc_as_half8(args_1),
                            __clc_as_half8(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_fclamp(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1,
                   __clc_vec16_float16_t args_2) {
  return __spirv_ocl_fclamp(__clc_as_half16(args_0), __clc_as_half16(args_1),
                            __clc_as_half16(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_fdim(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_ocl_fdim(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_fdim(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_ocl_fdim(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_fdim(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_ocl_fdim(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_fdim(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_ocl_fdim(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_fdim(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_ocl_fdim(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_fdim(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1) {
  return __spirv_ocl_fdim(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_floor(__clc_float16_t args_0) {
  return __spirv_ocl_floor(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_floor(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_floor(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_floor(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_floor(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_floor(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_floor(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_floor(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_floor(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_floor(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_floor(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t __spirv_ocl_fma(
    __clc_float16_t args_0, __clc_float16_t args_1, __clc_float16_t args_2) {
  return __spirv_ocl_fma(__clc_as_half(args_0), __clc_as_half(args_1), __clc_as_half(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_fma(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1,
                __clc_vec2_float16_t args_2) {
  return __spirv_ocl_fma(__clc_as_half2(args_0), __clc_as_half2(args_1), __clc_as_half2(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_fma(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1,
                __clc_vec3_float16_t args_2) {
  return __spirv_ocl_fma(__clc_as_half3(args_0), __clc_as_half3(args_1), __clc_as_half3(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_fma(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1,
                __clc_vec4_float16_t args_2) {
  return __spirv_ocl_fma(__clc_as_half4(args_0), __clc_as_half4(args_1), __clc_as_half4(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_fma(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1,
                __clc_vec8_float16_t args_2) {
  return __spirv_ocl_fma(__clc_as_half8(args_0), __clc_as_half8(args_1), __clc_as_half8(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_fma(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1,
                __clc_vec16_float16_t args_2) {
  return __spirv_ocl_fma(__clc_as_half16(args_0), __clc_as_half16(args_1),
                         __clc_as_half16(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_fmax(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_ocl_fmax(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_fmax(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_ocl_fmax(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_fmax(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_ocl_fmax(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_fmax(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_ocl_fmax(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_fmax(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_ocl_fmax(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_fmax(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1) {
  return __spirv_ocl_fmax(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_fmax_common(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_ocl_fmax_common(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t __spirv_ocl_fmax_common(
    __clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_ocl_fmax_common(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t __spirv_ocl_fmax_common(
    __clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_ocl_fmax_common(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t __spirv_ocl_fmax_common(
    __clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_ocl_fmax_common(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t __spirv_ocl_fmax_common(
    __clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_ocl_fmax_common(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t __spirv_ocl_fmax_common(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1) {
  return __spirv_ocl_fmax_common(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_fmin(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_ocl_fmin(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_fmin(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_ocl_fmin(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_fmin(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_ocl_fmin(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_fmin(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_ocl_fmin(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_fmin(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_ocl_fmin(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_fmin(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1) {
  return __spirv_ocl_fmin(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_fmin_common(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_ocl_fmin_common(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t __spirv_ocl_fmin_common(
    __clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_ocl_fmin_common(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t __spirv_ocl_fmin_common(
    __clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_ocl_fmin_common(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t __spirv_ocl_fmin_common(
    __clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_ocl_fmin_common(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t __spirv_ocl_fmin_common(
    __clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_ocl_fmin_common(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t __spirv_ocl_fmin_common(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1) {
  return __spirv_ocl_fmin_common(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_fmod(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_ocl_fmod(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_fmod(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_ocl_fmod(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_fmod(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_ocl_fmod(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_fmod(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_ocl_fmod(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_fmod(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_ocl_fmod(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_fmod(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1) {
  return __spirv_ocl_fmod(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_fp16_t
__spirv_ocl_fract(__clc_float16_t args_0, __clc_float16_t __private *args_1) {
  return __spirv_ocl_fract(__clc_as_half(args_0), (__clc_fp16_t __private *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_fp16_t
__spirv_ocl_fract(__clc_float16_t args_0, __clc_float16_t __local *args_1) {
  return __spirv_ocl_fract(__clc_as_half(args_0), (__clc_fp16_t __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_fp16_t
__spirv_ocl_fract(__clc_float16_t args_0, __clc_float16_t __global *args_1) {
  return __spirv_ocl_fract(__clc_as_half(args_0), (__clc_fp16_t __global *)(args_1));
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_fp16_t
__spirv_ocl_fract(__clc_float16_t args_0, __clc_float16_t __generic *args_1) {
  return __spirv_ocl_fract(__clc_as_half(args_0), (__clc_fp16_t __generic *)(args_1));
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t __spirv_ocl_fract(
    __clc_vec2_float16_t args_0, __clc_vec2_float16_t __private *args_1) {
  return __spirv_ocl_fract(__clc_as_half2(args_0),
                           (__clc_vec2_fp16_t __private *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t __spirv_ocl_fract(
    __clc_vec2_float16_t args_0, __clc_vec2_float16_t __local *args_1) {
  return __spirv_ocl_fract(__clc_as_half2(args_0),
                           (__clc_vec2_fp16_t __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t __spirv_ocl_fract(
    __clc_vec2_float16_t args_0, __clc_vec2_float16_t __global *args_1) {
  return __spirv_ocl_fract(__clc_as_half2(args_0),
                           (__clc_vec2_fp16_t __global *)(args_1));
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t __spirv_ocl_fract(
    __clc_vec2_float16_t args_0, __clc_vec2_float16_t __generic *args_1) {
  return __spirv_ocl_fract(__clc_as_half2(args_0),
                           (__clc_vec2_fp16_t __generic *)(args_1));
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t __spirv_ocl_fract(
    __clc_vec3_float16_t args_0, __clc_vec3_float16_t __private *args_1) {
  return __spirv_ocl_fract(__clc_as_half3(args_0),
                           (__clc_vec3_fp16_t __private *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t __spirv_ocl_fract(
    __clc_vec3_float16_t args_0, __clc_vec3_float16_t __local *args_1) {
  return __spirv_ocl_fract(__clc_as_half3(args_0),
                           (__clc_vec3_fp16_t __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t __spirv_ocl_fract(
    __clc_vec3_float16_t args_0, __clc_vec3_float16_t __global *args_1) {
  return __spirv_ocl_fract(__clc_as_half3(args_0),
                           (__clc_vec3_fp16_t __global *)(args_1));
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t __spirv_ocl_fract(
    __clc_vec3_float16_t args_0, __clc_vec3_float16_t __generic *args_1) {
  return __spirv_ocl_fract(__clc_as_half3(args_0),
                           (__clc_vec3_fp16_t __generic *)(args_1));
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t __spirv_ocl_fract(
    __clc_vec4_float16_t args_0, __clc_vec4_float16_t __private *args_1) {
  return __spirv_ocl_fract(__clc_as_half4(args_0),
                           (__clc_vec4_fp16_t __private *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t __spirv_ocl_fract(
    __clc_vec4_float16_t args_0, __clc_vec4_float16_t __local *args_1) {
  return __spirv_ocl_fract(__clc_as_half4(args_0),
                           (__clc_vec4_fp16_t __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t __spirv_ocl_fract(
    __clc_vec4_float16_t args_0, __clc_vec4_float16_t __global *args_1) {
  return __spirv_ocl_fract(__clc_as_half4(args_0),
                           (__clc_vec4_fp16_t __global *)(args_1));
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t __spirv_ocl_fract(
    __clc_vec4_float16_t args_0, __clc_vec4_float16_t __generic *args_1) {
  return __spirv_ocl_fract(__clc_as_half4(args_0),
                           (__clc_vec4_fp16_t __generic *)(args_1));
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t __spirv_ocl_fract(
    __clc_vec8_float16_t args_0, __clc_vec8_float16_t __private *args_1) {
  return __spirv_ocl_fract(__clc_as_half8(args_0),
                           (__clc_vec8_fp16_t __private *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t __spirv_ocl_fract(
    __clc_vec8_float16_t args_0, __clc_vec8_float16_t __local *args_1) {
  return __spirv_ocl_fract(__clc_as_half8(args_0),
                           (__clc_vec8_fp16_t __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t __spirv_ocl_fract(
    __clc_vec8_float16_t args_0, __clc_vec8_float16_t __global *args_1) {
  return __spirv_ocl_fract(__clc_as_half8(args_0),
                           (__clc_vec8_fp16_t __global *)(args_1));
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t __spirv_ocl_fract(
    __clc_vec8_float16_t args_0, __clc_vec8_float16_t __generic *args_1) {
  return __spirv_ocl_fract(__clc_as_half8(args_0),
                           (__clc_vec8_fp16_t __generic *)(args_1));
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t __spirv_ocl_fract(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t __private *args_1) {
  return __spirv_ocl_fract(__clc_as_half16(args_0),
                           (__clc_vec16_fp16_t __private *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t __spirv_ocl_fract(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t __local *args_1) {
  return __spirv_ocl_fract(__clc_as_half16(args_0),
                           (__clc_vec16_fp16_t __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t __spirv_ocl_fract(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t __global *args_1) {
  return __spirv_ocl_fract(__clc_as_half16(args_0),
                           (__clc_vec16_fp16_t __global *)(args_1));
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t __spirv_ocl_fract(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t __generic *args_1) {
  return __spirv_ocl_fract(__clc_as_half16(args_0),
                           (__clc_vec16_fp16_t __generic *)(args_1));
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_fp16_t
__spirv_ocl_frexp(__clc_float16_t args_0, __clc_int32_t __private *args_1) {
  return __spirv_ocl_frexp(__clc_as_half(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF __clc_fp16_t
__spirv_ocl_frexp(__clc_float16_t args_0, __clc_int32_t __local *args_1) {
  return __spirv_ocl_frexp(__clc_as_half(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF __clc_fp16_t
__spirv_ocl_frexp(__clc_float16_t args_0, __clc_int32_t __global *args_1) {
  return __spirv_ocl_frexp(__clc_as_half(args_0), args_1);
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_fp16_t
__spirv_ocl_frexp(__clc_float16_t args_0, __clc_int32_t __generic *args_1) {
  return __spirv_ocl_frexp(__clc_as_half(args_0), args_1);
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t __spirv_ocl_frexp(
    __clc_vec2_float16_t args_0, __clc_vec2_int32_t __private *args_1) {
  return __spirv_ocl_frexp(__clc_as_half2(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t __spirv_ocl_frexp(
    __clc_vec2_float16_t args_0, __clc_vec2_int32_t __local *args_1) {
  return __spirv_ocl_frexp(__clc_as_half2(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t __spirv_ocl_frexp(
    __clc_vec2_float16_t args_0, __clc_vec2_int32_t __global *args_1) {
  return __spirv_ocl_frexp(__clc_as_half2(args_0), args_1);
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t __spirv_ocl_frexp(
    __clc_vec2_float16_t args_0, __clc_vec2_int32_t __generic *args_1) {
  return __spirv_ocl_frexp(__clc_as_half2(args_0), args_1);
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t __spirv_ocl_frexp(
    __clc_vec3_float16_t args_0, __clc_vec3_int32_t __private *args_1) {
  return __spirv_ocl_frexp(__clc_as_half3(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t __spirv_ocl_frexp(
    __clc_vec3_float16_t args_0, __clc_vec3_int32_t __local *args_1) {
  return __spirv_ocl_frexp(__clc_as_half3(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t __spirv_ocl_frexp(
    __clc_vec3_float16_t args_0, __clc_vec3_int32_t __global *args_1) {
  return __spirv_ocl_frexp(__clc_as_half3(args_0), args_1);
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t __spirv_ocl_frexp(
    __clc_vec3_float16_t args_0, __clc_vec3_int32_t __generic *args_1) {
  return __spirv_ocl_frexp(__clc_as_half3(args_0), args_1);
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t __spirv_ocl_frexp(
    __clc_vec4_float16_t args_0, __clc_vec4_int32_t __private *args_1) {
  return __spirv_ocl_frexp(__clc_as_half4(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t __spirv_ocl_frexp(
    __clc_vec4_float16_t args_0, __clc_vec4_int32_t __local *args_1) {
  return __spirv_ocl_frexp(__clc_as_half4(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t __spirv_ocl_frexp(
    __clc_vec4_float16_t args_0, __clc_vec4_int32_t __global *args_1) {
  return __spirv_ocl_frexp(__clc_as_half4(args_0), args_1);
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t __spirv_ocl_frexp(
    __clc_vec4_float16_t args_0, __clc_vec4_int32_t __generic *args_1) {
  return __spirv_ocl_frexp(__clc_as_half4(args_0), args_1);
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t __spirv_ocl_frexp(
    __clc_vec8_float16_t args_0, __clc_vec8_int32_t __private *args_1) {
  return __spirv_ocl_frexp(__clc_as_half8(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t __spirv_ocl_frexp(
    __clc_vec8_float16_t args_0, __clc_vec8_int32_t __local *args_1) {
  return __spirv_ocl_frexp(__clc_as_half8(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t __spirv_ocl_frexp(
    __clc_vec8_float16_t args_0, __clc_vec8_int32_t __global *args_1) {
  return __spirv_ocl_frexp(__clc_as_half8(args_0), args_1);
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t __spirv_ocl_frexp(
    __clc_vec8_float16_t args_0, __clc_vec8_int32_t __generic *args_1) {
  return __spirv_ocl_frexp(__clc_as_half8(args_0), args_1);
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t __spirv_ocl_frexp(
    __clc_vec16_float16_t args_0, __clc_vec16_int32_t __private *args_1) {
  return __spirv_ocl_frexp(__clc_as_half16(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t __spirv_ocl_frexp(
    __clc_vec16_float16_t args_0, __clc_vec16_int32_t __local *args_1) {
  return __spirv_ocl_frexp(__clc_as_half16(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t __spirv_ocl_frexp(
    __clc_vec16_float16_t args_0, __clc_vec16_int32_t __global *args_1) {
  return __spirv_ocl_frexp(__clc_as_half16(args_0), args_1);
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t __spirv_ocl_frexp(
    __clc_vec16_float16_t args_0, __clc_vec16_int32_t __generic *args_1) {
  return __spirv_ocl_frexp(__clc_as_half16(args_0), args_1);
}
#endif

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_hypot(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_ocl_hypot(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_hypot(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_ocl_hypot(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_hypot(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_ocl_hypot(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_hypot(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_ocl_hypot(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_hypot(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_ocl_hypot(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_hypot(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1) {
  return __spirv_ocl_hypot(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_int32_t
__spirv_ocl_ilogb(__clc_float16_t args_0) {
  return __spirv_ocl_ilogb(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_int32_t
__spirv_ocl_ilogb(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_ilogb(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_int32_t
__spirv_ocl_ilogb(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_ilogb(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_int32_t
__spirv_ocl_ilogb(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_ilogb(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_int32_t
__spirv_ocl_ilogb(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_ilogb(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_int32_t
__spirv_ocl_ilogb(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_ilogb(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_ldexp(__clc_float16_t args_0, __clc_int32_t args_1) {
  return __spirv_ocl_ldexp(__clc_as_half(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_ldexp(__clc_float16_t args_0, __clc_uint32_t args_1) {
  return __spirv_ocl_ldexp(__clc_as_half(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_ldexp(__clc_vec2_float16_t args_0, __clc_vec2_int32_t args_1) {
  return __spirv_ocl_ldexp(__clc_as_half2(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_ldexp(__clc_vec2_float16_t args_0, __clc_vec2_uint32_t args_1) {
  return __spirv_ocl_ldexp(__clc_as_half2(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_ldexp(__clc_vec3_float16_t args_0, __clc_vec3_int32_t args_1) {
  return __spirv_ocl_ldexp(__clc_as_half3(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_ldexp(__clc_vec3_float16_t args_0, __clc_vec3_uint32_t args_1) {
  return __spirv_ocl_ldexp(__clc_as_half3(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_ldexp(__clc_vec4_float16_t args_0, __clc_vec4_int32_t args_1) {
  return __spirv_ocl_ldexp(__clc_as_half4(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_ldexp(__clc_vec4_float16_t args_0, __clc_vec4_uint32_t args_1) {
  return __spirv_ocl_ldexp(__clc_as_half4(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_ldexp(__clc_vec8_float16_t args_0, __clc_vec8_int32_t args_1) {
  return __spirv_ocl_ldexp(__clc_as_half8(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_ldexp(__clc_vec8_float16_t args_0, __clc_vec8_uint32_t args_1) {
  return __spirv_ocl_ldexp(__clc_as_half8(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_ldexp(__clc_vec16_float16_t args_0, __clc_vec16_int32_t args_1) {
  return __spirv_ocl_ldexp(__clc_as_half16(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_ldexp(__clc_vec16_float16_t args_0, __clc_vec16_uint32_t args_1) {
  return __spirv_ocl_ldexp(__clc_as_half16(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_length(__clc_float16_t args_0) {
  return __spirv_ocl_length(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_length(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_length(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_length(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_length(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_length(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_length(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_lgamma(__clc_float16_t args_0) {
  return __spirv_ocl_lgamma(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_lgamma(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_lgamma(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_lgamma(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_lgamma(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_lgamma(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_lgamma(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_lgamma(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_lgamma(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_lgamma(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_lgamma(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF __clc_fp16_t
__spirv_ocl_lgamma_r(__clc_float16_t args_0, __clc_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(__clc_as_half(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF __clc_fp16_t
__spirv_ocl_lgamma_r(__clc_float16_t args_0, __clc_int32_t __local *args_1) {
  return __spirv_ocl_lgamma_r(__clc_as_half(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF __clc_fp16_t
__spirv_ocl_lgamma_r(__clc_float16_t args_0, __clc_int32_t __global *args_1) {
  return __spirv_ocl_lgamma_r(__clc_as_half(args_0), args_1);
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_fp16_t
__spirv_ocl_lgamma_r(__clc_float16_t args_0, __clc_int32_t __generic *args_1) {
  return __spirv_ocl_lgamma_r(__clc_as_half(args_0), args_1);
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t __spirv_ocl_lgamma_r(
    __clc_vec2_float16_t args_0, __clc_vec2_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(__clc_as_half2(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t __spirv_ocl_lgamma_r(
    __clc_vec2_float16_t args_0, __clc_vec2_int32_t __local *args_1) {
  return __spirv_ocl_lgamma_r(__clc_as_half2(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t __spirv_ocl_lgamma_r(
    __clc_vec2_float16_t args_0, __clc_vec2_int32_t __global *args_1) {
  return __spirv_ocl_lgamma_r(__clc_as_half2(args_0), args_1);
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t __spirv_ocl_lgamma_r(
    __clc_vec2_float16_t args_0, __clc_vec2_int32_t __generic *args_1) {
  return __spirv_ocl_lgamma_r(__clc_as_half2(args_0), args_1);
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t __spirv_ocl_lgamma_r(
    __clc_vec3_float16_t args_0, __clc_vec3_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(__clc_as_half3(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t __spirv_ocl_lgamma_r(
    __clc_vec3_float16_t args_0, __clc_vec3_int32_t __local *args_1) {
  return __spirv_ocl_lgamma_r(__clc_as_half3(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t __spirv_ocl_lgamma_r(
    __clc_vec3_float16_t args_0, __clc_vec3_int32_t __global *args_1) {
  return __spirv_ocl_lgamma_r(__clc_as_half3(args_0), args_1);
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t __spirv_ocl_lgamma_r(
    __clc_vec3_float16_t args_0, __clc_vec3_int32_t __generic *args_1) {
  return __spirv_ocl_lgamma_r(__clc_as_half3(args_0), args_1);
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t __spirv_ocl_lgamma_r(
    __clc_vec4_float16_t args_0, __clc_vec4_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(__clc_as_half4(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t __spirv_ocl_lgamma_r(
    __clc_vec4_float16_t args_0, __clc_vec4_int32_t __local *args_1) {
  return __spirv_ocl_lgamma_r(__clc_as_half4(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t __spirv_ocl_lgamma_r(
    __clc_vec4_float16_t args_0, __clc_vec4_int32_t __global *args_1) {
  return __spirv_ocl_lgamma_r(__clc_as_half4(args_0), args_1);
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t __spirv_ocl_lgamma_r(
    __clc_vec4_float16_t args_0, __clc_vec4_int32_t __generic *args_1) {
  return __spirv_ocl_lgamma_r(__clc_as_half4(args_0), args_1);
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t __spirv_ocl_lgamma_r(
    __clc_vec8_float16_t args_0, __clc_vec8_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(__clc_as_half8(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t __spirv_ocl_lgamma_r(
    __clc_vec8_float16_t args_0, __clc_vec8_int32_t __local *args_1) {
  return __spirv_ocl_lgamma_r(__clc_as_half8(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t __spirv_ocl_lgamma_r(
    __clc_vec8_float16_t args_0, __clc_vec8_int32_t __global *args_1) {
  return __spirv_ocl_lgamma_r(__clc_as_half8(args_0), args_1);
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t __spirv_ocl_lgamma_r(
    __clc_vec8_float16_t args_0, __clc_vec8_int32_t __generic *args_1) {
  return __spirv_ocl_lgamma_r(__clc_as_half8(args_0), args_1);
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t __spirv_ocl_lgamma_r(
    __clc_vec16_float16_t args_0, __clc_vec16_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(__clc_as_half16(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t __spirv_ocl_lgamma_r(
    __clc_vec16_float16_t args_0, __clc_vec16_int32_t __local *args_1) {
  return __spirv_ocl_lgamma_r(__clc_as_half16(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t __spirv_ocl_lgamma_r(
    __clc_vec16_float16_t args_0, __clc_vec16_int32_t __global *args_1) {
  return __spirv_ocl_lgamma_r(__clc_as_half16(args_0), args_1);
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t __spirv_ocl_lgamma_r(
    __clc_vec16_float16_t args_0, __clc_vec16_int32_t __generic *args_1) {
  return __spirv_ocl_lgamma_r(__clc_as_half16(args_0), args_1);
}
#endif

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_log(__clc_float16_t args_0) {
  return __spirv_ocl_log(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_log(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_log(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_log(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_log(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_log(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_log(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_log(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_log(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_log(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_log(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_log10(__clc_float16_t args_0) {
  return __spirv_ocl_log10(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_log10(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_log10(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_log10(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_log10(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_log10(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_log10(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_log10(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_log10(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_log10(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_log10(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_log1p(__clc_float16_t args_0) {
  return __spirv_ocl_log1p(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_log1p(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_log1p(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_log1p(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_log1p(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_log1p(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_log1p(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_log1p(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_log1p(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_log1p(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_log1p(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_log2(__clc_float16_t args_0) {
  return __spirv_ocl_log2(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_log2(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_log2(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_log2(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_log2(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_log2(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_log2(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_log2(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_log2(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_log2(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_log2(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_logb(__clc_float16_t args_0) {
  return __spirv_ocl_logb(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_logb(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_logb(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_logb(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_logb(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_logb(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_logb(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_logb(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_logb(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_logb(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_logb(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t __spirv_ocl_mad(
    __clc_float16_t args_0, __clc_float16_t args_1, __clc_float16_t args_2) {
  return __spirv_ocl_mad(__clc_as_half(args_0), __clc_as_half(args_1), __clc_as_half(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_mad(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1,
                __clc_vec2_float16_t args_2) {
  return __spirv_ocl_mad(__clc_as_half2(args_0), __clc_as_half2(args_1), __clc_as_half2(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_mad(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1,
                __clc_vec3_float16_t args_2) {
  return __spirv_ocl_mad(__clc_as_half3(args_0), __clc_as_half3(args_1), __clc_as_half3(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_mad(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1,
                __clc_vec4_float16_t args_2) {
  return __spirv_ocl_mad(__clc_as_half4(args_0), __clc_as_half4(args_1), __clc_as_half4(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_mad(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1,
                __clc_vec8_float16_t args_2) {
  return __spirv_ocl_mad(__clc_as_half8(args_0), __clc_as_half8(args_1), __clc_as_half8(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_mad(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1,
                __clc_vec16_float16_t args_2) {
  return __spirv_ocl_mad(__clc_as_half16(args_0), __clc_as_half16(args_1),
                         __clc_as_half16(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_maxmag(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_ocl_maxmag(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_maxmag(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_ocl_maxmag(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_maxmag(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_ocl_maxmag(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_maxmag(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_ocl_maxmag(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_maxmag(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_ocl_maxmag(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_maxmag(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1) {
  return __spirv_ocl_maxmag(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_minmag(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_ocl_minmag(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_minmag(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_ocl_minmag(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_minmag(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_ocl_minmag(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_minmag(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_ocl_minmag(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_minmag(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_ocl_minmag(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_minmag(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1) {
  return __spirv_ocl_minmag(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t __spirv_ocl_mix(
    __clc_float16_t args_0, __clc_float16_t args_1, __clc_float16_t args_2) {
  return __spirv_ocl_mix(__clc_as_half(args_0), __clc_as_half(args_1), __clc_as_half(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_mix(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1,
                __clc_vec2_float16_t args_2) {
  return __spirv_ocl_mix(__clc_as_half2(args_0), __clc_as_half2(args_1), __clc_as_half2(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_mix(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1,
                __clc_vec3_float16_t args_2) {
  return __spirv_ocl_mix(__clc_as_half3(args_0), __clc_as_half3(args_1), __clc_as_half3(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_mix(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1,
                __clc_vec4_float16_t args_2) {
  return __spirv_ocl_mix(__clc_as_half4(args_0), __clc_as_half4(args_1), __clc_as_half4(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_mix(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1,
                __clc_vec8_float16_t args_2) {
  return __spirv_ocl_mix(__clc_as_half8(args_0), __clc_as_half8(args_1), __clc_as_half8(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_mix(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1,
                __clc_vec16_float16_t args_2) {
  return __spirv_ocl_mix(__clc_as_half16(args_0), __clc_as_half16(args_1),
                         __clc_as_half16(args_2));
}

_CLC_OVERLOAD _CLC_DEF __clc_fp16_t
__spirv_ocl_modf(__clc_float16_t args_0, __clc_float16_t __private *args_1) {
  return __spirv_ocl_modf(__clc_as_half(args_0), (__clc_fp16_t __private *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_fp16_t
__spirv_ocl_modf(__clc_float16_t args_0, __clc_float16_t __local *args_1) {
  return __spirv_ocl_modf(__clc_as_half(args_0), (__clc_fp16_t __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_fp16_t
__spirv_ocl_modf(__clc_float16_t args_0, __clc_float16_t __global *args_1) {
  return __spirv_ocl_modf(__clc_as_half(args_0), (__clc_fp16_t __global *)(args_1));
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_fp16_t
__spirv_ocl_modf(__clc_float16_t args_0, __clc_float16_t __generic *args_1) {
  return __spirv_ocl_modf(__clc_as_half(args_0), (__clc_fp16_t __generic *)(args_1));
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t __spirv_ocl_modf(
    __clc_vec2_float16_t args_0, __clc_vec2_float16_t __private *args_1) {
  return __spirv_ocl_modf(__clc_as_half2(args_0),
                          (__clc_vec2_fp16_t __private *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t __spirv_ocl_modf(
    __clc_vec2_float16_t args_0, __clc_vec2_float16_t __local *args_1) {
  return __spirv_ocl_modf(__clc_as_half2(args_0),
                          (__clc_vec2_fp16_t __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t __spirv_ocl_modf(
    __clc_vec2_float16_t args_0, __clc_vec2_float16_t __global *args_1) {
  return __spirv_ocl_modf(__clc_as_half2(args_0),
                          (__clc_vec2_fp16_t __global *)(args_1));
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t __spirv_ocl_modf(
    __clc_vec2_float16_t args_0, __clc_vec2_float16_t __generic *args_1) {
  return __spirv_ocl_modf(__clc_as_half2(args_0),
                          (__clc_vec2_fp16_t __generic *)(args_1));
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t __spirv_ocl_modf(
    __clc_vec3_float16_t args_0, __clc_vec3_float16_t __private *args_1) {
  return __spirv_ocl_modf(__clc_as_half3(args_0),
                          (__clc_vec3_fp16_t __private *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t __spirv_ocl_modf(
    __clc_vec3_float16_t args_0, __clc_vec3_float16_t __local *args_1) {
  return __spirv_ocl_modf(__clc_as_half3(args_0),
                          (__clc_vec3_fp16_t __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t __spirv_ocl_modf(
    __clc_vec3_float16_t args_0, __clc_vec3_float16_t __global *args_1) {
  return __spirv_ocl_modf(__clc_as_half3(args_0),
                          (__clc_vec3_fp16_t __global *)(args_1));
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t __spirv_ocl_modf(
    __clc_vec3_float16_t args_0, __clc_vec3_float16_t __generic *args_1) {
  return __spirv_ocl_modf(__clc_as_half3(args_0),
                          (__clc_vec3_fp16_t __generic *)(args_1));
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t __spirv_ocl_modf(
    __clc_vec4_float16_t args_0, __clc_vec4_float16_t __private *args_1) {
  return __spirv_ocl_modf(__clc_as_half4(args_0),
                          (__clc_vec4_fp16_t __private *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t __spirv_ocl_modf(
    __clc_vec4_float16_t args_0, __clc_vec4_float16_t __local *args_1) {
  return __spirv_ocl_modf(__clc_as_half4(args_0),
                          (__clc_vec4_fp16_t __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t __spirv_ocl_modf(
    __clc_vec4_float16_t args_0, __clc_vec4_float16_t __global *args_1) {
  return __spirv_ocl_modf(__clc_as_half4(args_0),
                          (__clc_vec4_fp16_t __global *)(args_1));
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t __spirv_ocl_modf(
    __clc_vec4_float16_t args_0, __clc_vec4_float16_t __generic *args_1) {
  return __spirv_ocl_modf(__clc_as_half4(args_0),
                          (__clc_vec4_fp16_t __generic *)(args_1));
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t __spirv_ocl_modf(
    __clc_vec8_float16_t args_0, __clc_vec8_float16_t __private *args_1) {
  return __spirv_ocl_modf(__clc_as_half8(args_0),
                          (__clc_vec8_fp16_t __private *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t __spirv_ocl_modf(
    __clc_vec8_float16_t args_0, __clc_vec8_float16_t __local *args_1) {
  return __spirv_ocl_modf(__clc_as_half8(args_0),
                          (__clc_vec8_fp16_t __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t __spirv_ocl_modf(
    __clc_vec8_float16_t args_0, __clc_vec8_float16_t __global *args_1) {
  return __spirv_ocl_modf(__clc_as_half8(args_0),
                          (__clc_vec8_fp16_t __global *)(args_1));
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t __spirv_ocl_modf(
    __clc_vec8_float16_t args_0, __clc_vec8_float16_t __generic *args_1) {
  return __spirv_ocl_modf(__clc_as_half8(args_0),
                          (__clc_vec8_fp16_t __generic *)(args_1));
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t __spirv_ocl_modf(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t __private *args_1) {
  return __spirv_ocl_modf(__clc_as_half16(args_0), (__clc_vec16_fp16_t __private *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t __spirv_ocl_modf(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t __local *args_1) {
  return __spirv_ocl_modf(__clc_as_half16(args_0),
                          (__clc_vec16_fp16_t __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t __spirv_ocl_modf(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t __global *args_1) {
  return __spirv_ocl_modf(__clc_as_half16(args_0),
                          (__clc_vec16_fp16_t __global *)(args_1));
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t __spirv_ocl_modf(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t __generic *args_1) {
  return __spirv_ocl_modf(__clc_as_half16(args_0),
                          (__clc_vec16_fp16_t __generic *)(args_1));
}
#endif

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_nextafter(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_ocl_nextafter(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t __spirv_ocl_nextafter(
    __clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_ocl_nextafter(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t __spirv_ocl_nextafter(
    __clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_ocl_nextafter(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t __spirv_ocl_nextafter(
    __clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_ocl_nextafter(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t __spirv_ocl_nextafter(
    __clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_ocl_nextafter(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t __spirv_ocl_nextafter(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1) {
  return __spirv_ocl_nextafter(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_normalize(__clc_float16_t args_0) {
  return __spirv_ocl_normalize(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_normalize(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_normalize(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_normalize(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_normalize(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_normalize(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_normalize(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_pow(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_ocl_pow(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_pow(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_ocl_pow(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_pow(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_ocl_pow(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_pow(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_ocl_pow(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_pow(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_ocl_pow(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_pow(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1) {
  return __spirv_ocl_pow(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_pown(__clc_float16_t args_0, __clc_int32_t args_1) {
  return __spirv_ocl_pown(__clc_as_half(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_pown(__clc_vec2_float16_t args_0, __clc_vec2_int32_t args_1) {
  return __spirv_ocl_pown(__clc_as_half2(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_pown(__clc_vec3_float16_t args_0, __clc_vec3_int32_t args_1) {
  return __spirv_ocl_pown(__clc_as_half3(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_pown(__clc_vec4_float16_t args_0, __clc_vec4_int32_t args_1) {
  return __spirv_ocl_pown(__clc_as_half4(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_pown(__clc_vec8_float16_t args_0, __clc_vec8_int32_t args_1) {
  return __spirv_ocl_pown(__clc_as_half8(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_pown(__clc_vec16_float16_t args_0, __clc_vec16_int32_t args_1) {
  return __spirv_ocl_pown(__clc_as_half16(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_powr(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_ocl_powr(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_powr(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_ocl_powr(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_powr(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_ocl_powr(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_powr(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_ocl_powr(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_powr(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_ocl_powr(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_powr(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1) {
  return __spirv_ocl_powr(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_prefetch(__clc_float16_t const __global *args_0,
                     __clc_size_t args_1) {
  __spirv_ocl_prefetch((__clc_fp16_t const __global *)(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_prefetch(__clc_vec2_float16_t const __global *args_0,
                     __clc_size_t args_1) {
  __spirv_ocl_prefetch((__clc_vec2_fp16_t const __global *)(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_prefetch(__clc_vec3_float16_t const __global *args_0,
                     __clc_size_t args_1) {
  __spirv_ocl_prefetch((__clc_vec3_fp16_t const __global *)(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_prefetch(__clc_vec4_float16_t const __global *args_0,
                     __clc_size_t args_1) {
  __spirv_ocl_prefetch((__clc_vec4_fp16_t const __global *)(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_prefetch(__clc_vec8_float16_t const __global *args_0,
                     __clc_size_t args_1) {
  __spirv_ocl_prefetch((__clc_vec8_fp16_t const __global *)(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_prefetch(__clc_vec16_float16_t const __global *args_0,
                     __clc_size_t args_1) {
  __spirv_ocl_prefetch((__clc_vec16_fp16_t const __global *)(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_radians(__clc_float16_t args_0) {
  return __spirv_ocl_radians(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_radians(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_radians(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_radians(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_radians(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_radians(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_radians(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_radians(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_radians(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_radians(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_radians(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_remainder(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_ocl_remainder(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t __spirv_ocl_remainder(
    __clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_ocl_remainder(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t __spirv_ocl_remainder(
    __clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_ocl_remainder(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t __spirv_ocl_remainder(
    __clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_ocl_remainder(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t __spirv_ocl_remainder(
    __clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_ocl_remainder(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t __spirv_ocl_remainder(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1) {
  return __spirv_ocl_remainder(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_fp16_t
__spirv_ocl_remquo(__clc_float16_t args_0, __clc_float16_t args_1,
                   __clc_int32_t __private *args_2) {
  return __spirv_ocl_remquo(__clc_as_half(args_0), __clc_as_half(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF __clc_fp16_t
__spirv_ocl_remquo(__clc_float16_t args_0, __clc_float16_t args_1,
                   __clc_int32_t __local *args_2) {
  return __spirv_ocl_remquo(__clc_as_half(args_0), __clc_as_half(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF __clc_fp16_t
__spirv_ocl_remquo(__clc_float16_t args_0, __clc_float16_t args_1,
                   __clc_int32_t __global *args_2) {
  return __spirv_ocl_remquo(__clc_as_half(args_0), __clc_as_half(args_1), args_2);
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_fp16_t
__spirv_ocl_remquo(__clc_float16_t args_0, __clc_float16_t args_1,
                   __clc_int32_t __generic *args_2) {
  return __spirv_ocl_remquo(__clc_as_half(args_0), __clc_as_half(args_1), args_2);
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t
__spirv_ocl_remquo(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1,
                   __clc_vec2_int32_t __private *args_2) {
  return __spirv_ocl_remquo(__clc_as_half2(args_0), __clc_as_half2(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t
__spirv_ocl_remquo(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1,
                   __clc_vec2_int32_t __local *args_2) {
  return __spirv_ocl_remquo(__clc_as_half2(args_0), __clc_as_half2(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t
__spirv_ocl_remquo(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1,
                   __clc_vec2_int32_t __global *args_2) {
  return __spirv_ocl_remquo(__clc_as_half2(args_0), __clc_as_half2(args_1), args_2);
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t
__spirv_ocl_remquo(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1,
                   __clc_vec2_int32_t __generic *args_2) {
  return __spirv_ocl_remquo(__clc_as_half2(args_0), __clc_as_half2(args_1), args_2);
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t
__spirv_ocl_remquo(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1,
                   __clc_vec3_int32_t __private *args_2) {
  return __spirv_ocl_remquo(__clc_as_half3(args_0), __clc_as_half3(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t
__spirv_ocl_remquo(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1,
                   __clc_vec3_int32_t __local *args_2) {
  return __spirv_ocl_remquo(__clc_as_half3(args_0), __clc_as_half3(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t
__spirv_ocl_remquo(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1,
                   __clc_vec3_int32_t __global *args_2) {
  return __spirv_ocl_remquo(__clc_as_half3(args_0), __clc_as_half3(args_1), args_2);
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t
__spirv_ocl_remquo(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1,
                   __clc_vec3_int32_t __generic *args_2) {
  return __spirv_ocl_remquo(__clc_as_half3(args_0), __clc_as_half3(args_1), args_2);
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t
__spirv_ocl_remquo(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1,
                   __clc_vec4_int32_t __private *args_2) {
  return __spirv_ocl_remquo(__clc_as_half4(args_0), __clc_as_half4(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t
__spirv_ocl_remquo(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1,
                   __clc_vec4_int32_t __local *args_2) {
  return __spirv_ocl_remquo(__clc_as_half4(args_0), __clc_as_half4(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t
__spirv_ocl_remquo(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1,
                   __clc_vec4_int32_t __global *args_2) {
  return __spirv_ocl_remquo(__clc_as_half4(args_0), __clc_as_half4(args_1), args_2);
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t
__spirv_ocl_remquo(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1,
                   __clc_vec4_int32_t __generic *args_2) {
  return __spirv_ocl_remquo(__clc_as_half4(args_0), __clc_as_half4(args_1), args_2);
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t
__spirv_ocl_remquo(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1,
                   __clc_vec8_int32_t __private *args_2) {
  return __spirv_ocl_remquo(__clc_as_half8(args_0), __clc_as_half8(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t
__spirv_ocl_remquo(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1,
                   __clc_vec8_int32_t __local *args_2) {
  return __spirv_ocl_remquo(__clc_as_half8(args_0), __clc_as_half8(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t
__spirv_ocl_remquo(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1,
                   __clc_vec8_int32_t __global *args_2) {
  return __spirv_ocl_remquo(__clc_as_half8(args_0), __clc_as_half8(args_1), args_2);
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t
__spirv_ocl_remquo(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1,
                   __clc_vec8_int32_t __generic *args_2) {
  return __spirv_ocl_remquo(__clc_as_half8(args_0), __clc_as_half8(args_1), args_2);
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t
__spirv_ocl_remquo(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1,
                   __clc_vec16_int32_t __private *args_2) {
  return __spirv_ocl_remquo(__clc_as_half16(args_0), __clc_as_half16(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t
__spirv_ocl_remquo(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1,
                   __clc_vec16_int32_t __local *args_2) {
  return __spirv_ocl_remquo(__clc_as_half16(args_0), __clc_as_half16(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t
__spirv_ocl_remquo(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1,
                   __clc_vec16_int32_t __global *args_2) {
  return __spirv_ocl_remquo(__clc_as_half16(args_0), __clc_as_half16(args_1), args_2);
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t
__spirv_ocl_remquo(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1,
                   __clc_vec16_int32_t __generic *args_2) {
  return __spirv_ocl_remquo(__clc_as_half16(args_0), __clc_as_half16(args_1), args_2);
}
#endif

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_rint(__clc_float16_t args_0) {
  return __spirv_ocl_rint(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_rint(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_rint(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_rint(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_rint(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_rint(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_rint(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_rint(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_rint(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_rint(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_rint(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_rootn(__clc_float16_t args_0, __clc_int32_t args_1) {
  return __spirv_ocl_rootn(__clc_as_half(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_rootn(__clc_vec2_float16_t args_0, __clc_vec2_int32_t args_1) {
  return __spirv_ocl_rootn(__clc_as_half2(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_rootn(__clc_vec3_float16_t args_0, __clc_vec3_int32_t args_1) {
  return __spirv_ocl_rootn(__clc_as_half3(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_rootn(__clc_vec4_float16_t args_0, __clc_vec4_int32_t args_1) {
  return __spirv_ocl_rootn(__clc_as_half4(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_rootn(__clc_vec8_float16_t args_0, __clc_vec8_int32_t args_1) {
  return __spirv_ocl_rootn(__clc_as_half8(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_rootn(__clc_vec16_float16_t args_0, __clc_vec16_int32_t args_1) {
  return __spirv_ocl_rootn(__clc_as_half16(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_round(__clc_float16_t args_0) {
  return __spirv_ocl_round(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_round(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_round(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_round(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_round(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_round(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_round(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_round(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_round(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_round(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_round(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_rsqrt(__clc_float16_t args_0) {
  return __spirv_ocl_rsqrt(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_rsqrt(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_rsqrt(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_rsqrt(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_rsqrt(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_rsqrt(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_rsqrt(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_rsqrt(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_rsqrt(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_rsqrt(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_rsqrt(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t __spirv_ocl_select(
    __clc_float16_t args_0, __clc_float16_t args_1, __clc_int16_t args_2) {
  return __spirv_ocl_select(__clc_as_half(args_0), __clc_as_half(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t __spirv_ocl_select(
    __clc_float16_t args_0, __clc_float16_t args_1, __clc_uint16_t args_2) {
  return __spirv_ocl_select(__clc_as_half(args_0), __clc_as_half(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_select(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1,
                   __clc_vec2_int16_t args_2) {
  return __spirv_ocl_select(__clc_as_half2(args_0), __clc_as_half2(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_select(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1,
                   __clc_vec2_uint16_t args_2) {
  return __spirv_ocl_select(__clc_as_half2(args_0), __clc_as_half2(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_select(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1,
                   __clc_vec3_int16_t args_2) {
  return __spirv_ocl_select(__clc_as_half3(args_0), __clc_as_half3(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_select(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1,
                   __clc_vec3_uint16_t args_2) {
  return __spirv_ocl_select(__clc_as_half3(args_0), __clc_as_half3(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_select(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1,
                   __clc_vec4_int16_t args_2) {
  return __spirv_ocl_select(__clc_as_half4(args_0), __clc_as_half4(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_select(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1,
                   __clc_vec4_uint16_t args_2) {
  return __spirv_ocl_select(__clc_as_half4(args_0), __clc_as_half4(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_select(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1,
                   __clc_vec8_int16_t args_2) {
  return __spirv_ocl_select(__clc_as_half8(args_0), __clc_as_half8(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_select(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1,
                   __clc_vec8_uint16_t args_2) {
  return __spirv_ocl_select(__clc_as_half8(args_0), __clc_as_half8(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_select(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1,
                   __clc_vec16_int16_t args_2) {
  return __spirv_ocl_select(__clc_as_half16(args_0), __clc_as_half16(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_select(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1,
                   __clc_vec16_uint16_t args_2) {
  return __spirv_ocl_select(__clc_as_half16(args_0), __clc_as_half16(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_shuffle(__clc_vec2_float16_t args_0, __clc_vec2_uint16_t args_1) {
  return __spirv_ocl_shuffle(__clc_as_half2(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_shuffle(__clc_vec4_float16_t args_0, __clc_vec2_uint16_t args_1) {
  return __spirv_ocl_shuffle(__clc_as_half4(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_shuffle(__clc_vec8_float16_t args_0, __clc_vec2_uint16_t args_1) {
  return __spirv_ocl_shuffle(__clc_as_half8(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_shuffle(__clc_vec16_float16_t args_0, __clc_vec2_uint16_t args_1) {
  return __spirv_ocl_shuffle(__clc_as_half16(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_shuffle(__clc_vec2_float16_t args_0, __clc_vec4_uint16_t args_1) {
  return __spirv_ocl_shuffle(__clc_as_half2(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_shuffle(__clc_vec4_float16_t args_0, __clc_vec4_uint16_t args_1) {
  return __spirv_ocl_shuffle(__clc_as_half4(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_shuffle(__clc_vec8_float16_t args_0, __clc_vec4_uint16_t args_1) {
  return __spirv_ocl_shuffle(__clc_as_half8(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_shuffle(__clc_vec16_float16_t args_0, __clc_vec4_uint16_t args_1) {
  return __spirv_ocl_shuffle(__clc_as_half16(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_shuffle(__clc_vec2_float16_t args_0, __clc_vec8_uint16_t args_1) {
  return __spirv_ocl_shuffle(__clc_as_half2(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_shuffle(__clc_vec4_float16_t args_0, __clc_vec8_uint16_t args_1) {
  return __spirv_ocl_shuffle(__clc_as_half4(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_shuffle(__clc_vec8_float16_t args_0, __clc_vec8_uint16_t args_1) {
  return __spirv_ocl_shuffle(__clc_as_half8(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_shuffle(__clc_vec16_float16_t args_0, __clc_vec8_uint16_t args_1) {
  return __spirv_ocl_shuffle(__clc_as_half16(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_shuffle(__clc_vec2_float16_t args_0, __clc_vec16_uint16_t args_1) {
  return __spirv_ocl_shuffle(__clc_as_half2(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_shuffle(__clc_vec4_float16_t args_0, __clc_vec16_uint16_t args_1) {
  return __spirv_ocl_shuffle(__clc_as_half4(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_shuffle(__clc_vec8_float16_t args_0, __clc_vec16_uint16_t args_1) {
  return __spirv_ocl_shuffle(__clc_as_half8(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_shuffle(__clc_vec16_float16_t args_0, __clc_vec16_uint16_t args_1) {
  return __spirv_ocl_shuffle(__clc_as_half16(args_0), args_1);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_shuffle2(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1,
                     __clc_vec2_uint16_t args_2) {
  return __spirv_ocl_shuffle2(__clc_as_half2(args_0), __clc_as_half2(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_shuffle2(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1,
                     __clc_vec2_uint16_t args_2) {
  return __spirv_ocl_shuffle2(__clc_as_half4(args_0), __clc_as_half4(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_shuffle2(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1,
                     __clc_vec2_uint16_t args_2) {
  return __spirv_ocl_shuffle2(__clc_as_half8(args_0), __clc_as_half8(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_shuffle2(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1,
                     __clc_vec2_uint16_t args_2) {
  return __spirv_ocl_shuffle2(__clc_as_half16(args_0), __clc_as_half16(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_shuffle2(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1,
                     __clc_vec4_uint16_t args_2) {
  return __spirv_ocl_shuffle2(__clc_as_half2(args_0), __clc_as_half2(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_shuffle2(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1,
                     __clc_vec4_uint16_t args_2) {
  return __spirv_ocl_shuffle2(__clc_as_half4(args_0), __clc_as_half4(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_shuffle2(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1,
                     __clc_vec4_uint16_t args_2) {
  return __spirv_ocl_shuffle2(__clc_as_half8(args_0), __clc_as_half8(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_shuffle2(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1,
                     __clc_vec4_uint16_t args_2) {
  return __spirv_ocl_shuffle2(__clc_as_half16(args_0), __clc_as_half16(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_shuffle2(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1,
                     __clc_vec8_uint16_t args_2) {
  return __spirv_ocl_shuffle2(__clc_as_half2(args_0), __clc_as_half2(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_shuffle2(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1,
                     __clc_vec8_uint16_t args_2) {
  return __spirv_ocl_shuffle2(__clc_as_half4(args_0), __clc_as_half4(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_shuffle2(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1,
                     __clc_vec8_uint16_t args_2) {
  return __spirv_ocl_shuffle2(__clc_as_half8(args_0), __clc_as_half8(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_shuffle2(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1,
                     __clc_vec8_uint16_t args_2) {
  return __spirv_ocl_shuffle2(__clc_as_half16(args_0), __clc_as_half16(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_shuffle2(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1,
                     __clc_vec16_uint16_t args_2) {
  return __spirv_ocl_shuffle2(__clc_as_half2(args_0), __clc_as_half2(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_shuffle2(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1,
                     __clc_vec16_uint16_t args_2) {
  return __spirv_ocl_shuffle2(__clc_as_half4(args_0), __clc_as_half4(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_shuffle2(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1,
                     __clc_vec16_uint16_t args_2) {
  return __spirv_ocl_shuffle2(__clc_as_half8(args_0), __clc_as_half8(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_shuffle2(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1,
                     __clc_vec16_uint16_t args_2) {
  return __spirv_ocl_shuffle2(__clc_as_half16(args_0), __clc_as_half16(args_1), args_2);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_sign(__clc_float16_t args_0) {
  return __spirv_ocl_sign(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_sign(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_sign(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_sign(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_sign(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_sign(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_sign(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_sign(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_sign(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_sign(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_sign(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_sin(__clc_float16_t args_0) {
  return __spirv_ocl_sin(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_sin(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_sin(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_sin(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_sin(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_sin(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_sin(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_sin(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_sin(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_sin(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_sin(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF __clc_fp16_t
__spirv_ocl_sincos(__clc_float16_t args_0, __clc_float16_t __private *args_1) {
  return __spirv_ocl_sincos(__clc_as_half(args_0),
                            (__clc_fp16_t __private *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_fp16_t
__spirv_ocl_sincos(__clc_float16_t args_0, __clc_float16_t __local *args_1) {
  return __spirv_ocl_sincos(__clc_as_half(args_0), (__clc_fp16_t __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_fp16_t
__spirv_ocl_sincos(__clc_float16_t args_0, __clc_float16_t __global *args_1) {
  return __spirv_ocl_sincos(__clc_as_half(args_0), (__clc_fp16_t __global *)(args_1));
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_fp16_t
__spirv_ocl_sincos(__clc_float16_t args_0, __clc_float16_t __generic *args_1) {
  return __spirv_ocl_sincos(__clc_as_half(args_0),
                            (__clc_fp16_t __generic *)(args_1));
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t __spirv_ocl_sincos(
    __clc_vec2_float16_t args_0, __clc_vec2_float16_t __private *args_1) {
  return __spirv_ocl_sincos(__clc_as_half2(args_0), (__clc_vec2_fp16_t __private *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t __spirv_ocl_sincos(
    __clc_vec2_float16_t args_0, __clc_vec2_float16_t __local *args_1) {
  return __spirv_ocl_sincos(__clc_as_half2(args_0),
                            (__clc_vec2_fp16_t __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t __spirv_ocl_sincos(
    __clc_vec2_float16_t args_0, __clc_vec2_float16_t __global *args_1) {
  return __spirv_ocl_sincos(__clc_as_half2(args_0),
                            (__clc_vec2_fp16_t __global *)(args_1));
}
#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t __spirv_ocl_sincos(
    __clc_vec2_float16_t args_0, __clc_vec2_float16_t __generic *args_1) {
  return __spirv_ocl_sincos(__clc_as_half2(args_0), (__clc_vec2_fp16_t __generic *)(args_1));
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t __spirv_ocl_sincos(
    __clc_vec3_float16_t args_0, __clc_vec3_float16_t __private *args_1) {
  return __spirv_ocl_sincos(__clc_as_half3(args_0), (__clc_vec3_fp16_t __private *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t __spirv_ocl_sincos(
    __clc_vec3_float16_t args_0, __clc_vec3_float16_t __local *args_1) {
  return __spirv_ocl_sincos(__clc_as_half3(args_0),
                            (__clc_vec3_fp16_t __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t __spirv_ocl_sincos(
    __clc_vec3_float16_t args_0, __clc_vec3_float16_t __global *args_1) {
  return __spirv_ocl_sincos(__clc_as_half3(args_0),
                            (__clc_vec3_fp16_t __global *)(args_1));
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t __spirv_ocl_sincos(
    __clc_vec3_float16_t args_0, __clc_vec3_float16_t __generic *args_1) {
  return __spirv_ocl_sincos(__clc_as_half3(args_0), (__clc_vec3_fp16_t __generic *)(args_1));
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t
__spirv_ocl_sincos(__clc_vec4_float16_t args_0, __clc_vec4_float16_t __private *args_1) {
  return __spirv_ocl_sincos(__clc_as_half4(args_0),
                            (__clc_vec4_fp16_t __private *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t __spirv_ocl_sincos(
    __clc_vec4_float16_t args_0, __clc_vec4_float16_t __local *args_1) {
  return __spirv_ocl_sincos(__clc_as_half4(args_0),
                            (__clc_vec4_fp16_t __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t __spirv_ocl_sincos(
    __clc_vec4_float16_t args_0, __clc_vec4_float16_t __global *args_1) {
  return __spirv_ocl_sincos(__clc_as_half4(args_0),
                            (__clc_vec4_fp16_t __global *)(args_1));
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t
__spirv_ocl_sincos(__clc_vec4_float16_t args_0, __clc_vec4_float16_t __generic *args_1) {
  return __spirv_ocl_sincos(__clc_as_half4(args_0),
                            (__clc_vec4_fp16_t __generic *)(args_1));
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t __spirv_ocl_sincos(
    __clc_vec8_float16_t args_0, __clc_vec8_float16_t __private *args_1) {
  return __spirv_ocl_sincos(__clc_as_half8(args_0),
                            (__clc_vec8_fp16_t __private *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t __spirv_ocl_sincos(
    __clc_vec8_float16_t args_0, __clc_vec8_float16_t __local *args_1) {
  return __spirv_ocl_sincos(__clc_as_half8(args_0),
                            (__clc_vec8_fp16_t __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t __spirv_ocl_sincos(
    __clc_vec8_float16_t args_0, __clc_vec8_float16_t __global *args_1) {
  return __spirv_ocl_sincos(__clc_as_half8(args_0),
                            (__clc_vec8_fp16_t __global *)(args_1));
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t __spirv_ocl_sincos(
    __clc_vec8_float16_t args_0, __clc_vec8_float16_t __generic *args_1) {
  return __spirv_ocl_sincos(__clc_as_half8(args_0),
                            (__clc_vec8_fp16_t __generic *)(args_1));
}
#endif

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t __spirv_ocl_sincos(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t __private *args_1) {
  return __spirv_ocl_sincos(__clc_as_half16(args_0),
                            (__clc_vec16_fp16_t __private *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t __spirv_ocl_sincos(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t __local *args_1) {
  return __spirv_ocl_sincos(__clc_as_half16(args_0),
                            (__clc_vec16_fp16_t __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t __spirv_ocl_sincos(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t __global *args_1) {
  return __spirv_ocl_sincos(__clc_as_half16(args_0),
                            (__clc_vec16_fp16_t __global *)(args_1));
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t __spirv_ocl_sincos(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t __generic *args_1) {
  return __spirv_ocl_sincos(__clc_as_half16(args_0),
                            (__clc_vec16_fp16_t __generic *)(args_1));
}
#endif

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_sinh(__clc_float16_t args_0) {
  return __spirv_ocl_sinh(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_sinh(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_sinh(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_sinh(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_sinh(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_sinh(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_sinh(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_sinh(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_sinh(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_sinh(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_sinh(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_sinpi(__clc_float16_t args_0) {
  return __spirv_ocl_sinpi(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_sinpi(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_sinpi(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_sinpi(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_sinpi(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_sinpi(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_sinpi(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_sinpi(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_sinpi(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_sinpi(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_sinpi(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t __spirv_ocl_smoothstep(
    __clc_float16_t args_0, __clc_float16_t args_1, __clc_float16_t args_2) {
  return __spirv_ocl_smoothstep(__clc_as_half(args_0), __clc_as_half(args_1),
                                __clc_as_half(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_smoothstep(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1,
                       __clc_vec2_float16_t args_2) {
  return __spirv_ocl_smoothstep(__clc_as_half2(args_0), __clc_as_half2(args_1),
                                __clc_as_half2(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_smoothstep(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1,
                       __clc_vec3_float16_t args_2) {
  return __spirv_ocl_smoothstep(__clc_as_half3(args_0), __clc_as_half3(args_1),
                                __clc_as_half3(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_smoothstep(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1,
                       __clc_vec4_float16_t args_2) {
  return __spirv_ocl_smoothstep(__clc_as_half4(args_0), __clc_as_half4(args_1),
                                __clc_as_half4(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_smoothstep(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1,
                       __clc_vec8_float16_t args_2) {
  return __spirv_ocl_smoothstep(__clc_as_half8(args_0), __clc_as_half8(args_1),
                                __clc_as_half8(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t __spirv_ocl_smoothstep(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1,
    __clc_vec16_float16_t args_2) {
  return __spirv_ocl_smoothstep(__clc_as_half16(args_0), __clc_as_half16(args_1),
                                __clc_as_half16(args_2));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_sqrt(__clc_float16_t args_0) {
  return __spirv_ocl_sqrt(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_sqrt(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_sqrt(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_sqrt(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_sqrt(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_sqrt(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_sqrt(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_sqrt(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_sqrt(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_sqrt(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_sqrt(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_step(__clc_float16_t args_0, __clc_float16_t args_1) {
  return __spirv_ocl_step(__clc_as_half(args_0), __clc_as_half(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_step(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1) {
  return __spirv_ocl_step(__clc_as_half2(args_0), __clc_as_half2(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_step(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1) {
  return __spirv_ocl_step(__clc_as_half3(args_0), __clc_as_half3(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_step(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1) {
  return __spirv_ocl_step(__clc_as_half4(args_0), __clc_as_half4(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_step(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1) {
  return __spirv_ocl_step(__clc_as_half8(args_0), __clc_as_half8(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_step(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1) {
  return __spirv_ocl_step(__clc_as_half16(args_0), __clc_as_half16(args_1));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_tan(__clc_float16_t args_0) {
  return __spirv_ocl_tan(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_tan(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_tan(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_tan(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_tan(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_tan(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_tan(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_tan(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_tan(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_tan(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_tan(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_tanh(__clc_float16_t args_0) {
  return __spirv_ocl_tanh(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_tanh(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_tanh(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_tanh(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_tanh(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_tanh(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_tanh(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_tanh(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_tanh(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_tanh(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_tanh(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__clc_native_tanh(__clc_float16_t args_0) {
  return __clc_native_tanh(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__clc_native_tanh(__clc_vec2_float16_t args_0) {
  return __clc_native_tanh(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__clc_native_tanh(__clc_vec3_float16_t args_0) {
  return __clc_native_tanh(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__clc_native_tanh(__clc_vec4_float16_t args_0) {
  return __clc_native_tanh(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__clc_native_tanh(__clc_vec8_float16_t args_0) {
  return __clc_native_tanh(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__clc_native_tanh(__clc_vec16_float16_t args_0) {
  return __clc_native_tanh(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_tanpi(__clc_float16_t args_0) {
  return __spirv_ocl_tanpi(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_tanpi(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_tanpi(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_tanpi(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_tanpi(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_tanpi(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_tanpi(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_tanpi(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_tanpi(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_tanpi(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_tanpi(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_tgamma(__clc_float16_t args_0) {
  return __spirv_ocl_tgamma(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_tgamma(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_tgamma(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_tgamma(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_tgamma(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_tgamma(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_tgamma(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_tgamma(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_tgamma(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_tgamma(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_tgamma(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_fp16_t
__spirv_ocl_trunc(__clc_float16_t args_0) {
  return __spirv_ocl_trunc(__clc_as_half(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec2_fp16_t
__spirv_ocl_trunc(__clc_vec2_float16_t args_0) {
  return __spirv_ocl_trunc(__clc_as_half2(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec3_fp16_t
__spirv_ocl_trunc(__clc_vec3_float16_t args_0) {
  return __spirv_ocl_trunc(__clc_as_half3(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec4_fp16_t
__spirv_ocl_trunc(__clc_vec4_float16_t args_0) {
  return __spirv_ocl_trunc(__clc_as_half4(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec8_fp16_t
__spirv_ocl_trunc(__clc_vec8_float16_t args_0) {
  return __spirv_ocl_trunc(__clc_as_half8(args_0));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONSTFN __clc_vec16_fp16_t
__spirv_ocl_trunc(__clc_vec16_float16_t args_0) {
  return __spirv_ocl_trunc(__clc_as_half16(args_0));
}

_CLC_OVERLOAD _CLC_DEF __clc_fp32_t
__spirv_ocl_vload_half(__clc_size_t args_0, __clc_float16_t const *args_1) {
  return __spirv_ocl_vload_half(args_0, (__clc_fp16_t const *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_fp32_t __spirv_ocl_vload_half(
    __clc_size_t args_0, __clc_float16_t const __local *args_1) {
  return __spirv_ocl_vload_half(args_0, (__clc_fp16_t const __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_fp32_t __spirv_ocl_vload_half(
    __clc_size_t args_0, __clc_float16_t const __global *args_1) {
  return __spirv_ocl_vload_half(args_0,
                                (__clc_fp16_t const __global *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_fp32_t __spirv_ocl_vload_half(
    __clc_size_t args_0, __clc_float16_t const __constant *args_1) {
  return __spirv_ocl_vload_half(args_0,
                                (__clc_fp16_t const __constant *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp32_t __spirv_ocl_vload_halfn_Rfloat16(
    __clc_size_t args_0, __clc_float16_t const *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat16(args_0,
                                          (__clc_fp16_t const *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp32_t __spirv_ocl_vload_halfn_Rfloat16(
    __clc_size_t args_0, __clc_float16_t const __local *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat16(
      args_0, (__clc_fp16_t const __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp32_t __spirv_ocl_vload_halfn_Rfloat16(
    __clc_size_t args_0, __clc_float16_t const __global *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat16(
      args_0, (__clc_fp16_t const __global *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp32_t __spirv_ocl_vload_halfn_Rfloat16(
    __clc_size_t args_0, __clc_float16_t const __constant *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat16(
      args_0, (__clc_fp16_t const __constant *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp32_t __spirv_ocl_vload_halfn_Rfloat2(
    __clc_size_t args_0, __clc_float16_t const *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat2(args_0,
                                         (__clc_fp16_t const *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp32_t __spirv_ocl_vload_halfn_Rfloat2(
    __clc_size_t args_0, __clc_float16_t const __local *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat2(
      args_0, (__clc_fp16_t const __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp32_t __spirv_ocl_vload_halfn_Rfloat2(
    __clc_size_t args_0, __clc_float16_t const __global *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat2(
      args_0, (__clc_fp16_t const __global *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp32_t __spirv_ocl_vload_halfn_Rfloat2(
    __clc_size_t args_0, __clc_float16_t const __constant *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat2(
      args_0, (__clc_fp16_t const __constant *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp32_t __spirv_ocl_vload_halfn_Rfloat3(
    __clc_size_t args_0, __clc_float16_t const *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat3(args_0,
                                         (__clc_fp16_t const *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp32_t __spirv_ocl_vload_halfn_Rfloat3(
    __clc_size_t args_0, __clc_float16_t const __local *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat3(
      args_0, (__clc_fp16_t const __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp32_t __spirv_ocl_vload_halfn_Rfloat3(
    __clc_size_t args_0, __clc_float16_t const __global *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat3(
      args_0, (__clc_fp16_t const __global *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp32_t __spirv_ocl_vload_halfn_Rfloat3(
    __clc_size_t args_0, __clc_float16_t const __constant *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat3(
      args_0, (__clc_fp16_t const __constant *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp32_t __spirv_ocl_vload_halfn_Rfloat4(
    __clc_size_t args_0, __clc_float16_t const *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat4(args_0,
                                         (__clc_fp16_t const *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp32_t __spirv_ocl_vload_halfn_Rfloat4(
    __clc_size_t args_0, __clc_float16_t const __local *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat4(
      args_0, (__clc_fp16_t const __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp32_t __spirv_ocl_vload_halfn_Rfloat4(
    __clc_size_t args_0, __clc_float16_t const __global *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat4(
      args_0, (__clc_fp16_t const __global *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp32_t __spirv_ocl_vload_halfn_Rfloat4(
    __clc_size_t args_0, __clc_float16_t const __constant *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat4(
      args_0, (__clc_fp16_t const __constant *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp32_t __spirv_ocl_vload_halfn_Rfloat8(
    __clc_size_t args_0, __clc_float16_t const *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat8(args_0,
                                         (__clc_fp16_t const *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp32_t __spirv_ocl_vload_halfn_Rfloat8(
    __clc_size_t args_0, __clc_float16_t const __local *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat8(
      args_0, (__clc_fp16_t const __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp32_t __spirv_ocl_vload_halfn_Rfloat8(
    __clc_size_t args_0, __clc_float16_t const __global *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat8(
      args_0, (__clc_fp16_t const __global *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp32_t __spirv_ocl_vload_halfn_Rfloat8(
    __clc_size_t args_0, __clc_float16_t const __constant *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat8(
      args_0, (__clc_fp16_t const __constant *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp32_t __spirv_ocl_vloada_halfn_Rfloat16(
    __clc_size_t args_0, __clc_float16_t const *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat16(args_0,
                                           (__clc_fp16_t const *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp32_t __spirv_ocl_vloada_halfn_Rfloat16(
    __clc_size_t args_0, __clc_float16_t const __local *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat16(
      args_0, (__clc_fp16_t const __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp32_t __spirv_ocl_vloada_halfn_Rfloat16(
    __clc_size_t args_0, __clc_float16_t const __global *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat16(
      args_0, (__clc_fp16_t const __global *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp32_t __spirv_ocl_vloada_halfn_Rfloat16(
    __clc_size_t args_0, __clc_float16_t const __constant *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat16(
      args_0, (__clc_fp16_t const __constant *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp32_t __spirv_ocl_vloada_halfn_Rfloat2(
    __clc_size_t args_0, __clc_float16_t const *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat2(args_0,
                                          (__clc_fp16_t const *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp32_t __spirv_ocl_vloada_halfn_Rfloat2(
    __clc_size_t args_0, __clc_float16_t const __local *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat2(
      args_0, (__clc_fp16_t const __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp32_t __spirv_ocl_vloada_halfn_Rfloat2(
    __clc_size_t args_0, __clc_float16_t const __global *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat2(
      args_0, (__clc_fp16_t const __global *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp32_t __spirv_ocl_vloada_halfn_Rfloat2(
    __clc_size_t args_0, __clc_float16_t const __constant *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat2(
      args_0, (__clc_fp16_t const __constant *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp32_t __spirv_ocl_vloada_halfn_Rfloat3(
    __clc_size_t args_0, __clc_float16_t const *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat3(args_0,
                                          (__clc_fp16_t const *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp32_t __spirv_ocl_vloada_halfn_Rfloat3(
    __clc_size_t args_0, __clc_float16_t const __local *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat3(
      args_0, (__clc_fp16_t const __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp32_t __spirv_ocl_vloada_halfn_Rfloat3(
    __clc_size_t args_0, __clc_float16_t const __global *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat3(
      args_0, (__clc_fp16_t const __global *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp32_t __spirv_ocl_vloada_halfn_Rfloat3(
    __clc_size_t args_0, __clc_float16_t const __constant *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat3(
      args_0, (__clc_fp16_t const __constant *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp32_t __spirv_ocl_vloada_halfn_Rfloat4(
    __clc_size_t args_0, __clc_float16_t const *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat4(args_0,
                                          (__clc_fp16_t const *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp32_t __spirv_ocl_vloada_halfn_Rfloat4(
    __clc_size_t args_0, __clc_float16_t const __local *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat4(
      args_0, (__clc_fp16_t const __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp32_t __spirv_ocl_vloada_halfn_Rfloat4(
    __clc_size_t args_0, __clc_float16_t const __global *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat4(
      args_0, (__clc_fp16_t const __global *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp32_t __spirv_ocl_vloada_halfn_Rfloat4(
    __clc_size_t args_0, __clc_float16_t const __constant *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat4(
      args_0, (__clc_fp16_t const __constant *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp32_t __spirv_ocl_vloada_halfn_Rfloat8(
    __clc_size_t args_0, __clc_float16_t const *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat8(args_0,
                                          (__clc_fp16_t const *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp32_t __spirv_ocl_vloada_halfn_Rfloat8(
    __clc_size_t args_0, __clc_float16_t const __local *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat8(
      args_0, (__clc_fp16_t const __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp32_t __spirv_ocl_vloada_halfn_Rfloat8(
    __clc_size_t args_0, __clc_float16_t const __global *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat8(
      args_0, (__clc_fp16_t const __global *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp32_t __spirv_ocl_vloada_halfn_Rfloat8(
    __clc_size_t args_0, __clc_float16_t const __constant *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat8(
      args_0, (__clc_fp16_t const __constant *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t
__spirv_ocl_vloadn_Rhalf16(__clc_size_t args_0, __clc_float16_t const *args_1) {
  return __spirv_ocl_vloadn_Rhalf16(args_0, (__clc_fp16_t const *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t __spirv_ocl_vloadn_Rhalf16(
    __clc_size_t args_0, __clc_float16_t const __local *args_1) {
  return __spirv_ocl_vloadn_Rhalf16(args_0,
                                    (__clc_fp16_t const __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t __spirv_ocl_vloadn_Rhalf16(
    __clc_size_t args_0, __clc_float16_t const __global *args_1) {
  return __spirv_ocl_vloadn_Rhalf16(args_0,
                                    (__clc_fp16_t const __global *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t __spirv_ocl_vloadn_Rhalf16(
    __clc_size_t args_0, __clc_float16_t const __constant *args_1) {
  return __spirv_ocl_vloadn_Rhalf16(args_0,
                                    (__clc_fp16_t const __constant *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t
__spirv_ocl_vloadn_Rhalf2(__clc_size_t args_0, __clc_float16_t const *args_1) {
  return __spirv_ocl_vloadn_Rhalf2(args_0, (__clc_fp16_t const *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t __spirv_ocl_vloadn_Rhalf2(
    __clc_size_t args_0, __clc_float16_t const __local *args_1) {
  return __spirv_ocl_vloadn_Rhalf2(args_0,
                                   (__clc_fp16_t const __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t __spirv_ocl_vloadn_Rhalf2(
    __clc_size_t args_0, __clc_float16_t const __global *args_1) {
  return __spirv_ocl_vloadn_Rhalf2(args_0,
                                   (__clc_fp16_t const __global *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t __spirv_ocl_vloadn_Rhalf2(
    __clc_size_t args_0, __clc_float16_t const __constant *args_1) {
  return __spirv_ocl_vloadn_Rhalf2(args_0,
                                   (__clc_fp16_t const __constant *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t
__spirv_ocl_vloadn_Rhalf3(__clc_size_t args_0, __clc_float16_t const *args_1) {
  return __spirv_ocl_vloadn_Rhalf3(args_0, (__clc_fp16_t const *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t __spirv_ocl_vloadn_Rhalf3(
    __clc_size_t args_0, __clc_float16_t const __local *args_1) {
  return __spirv_ocl_vloadn_Rhalf3(args_0,
                                   (__clc_fp16_t const __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t __spirv_ocl_vloadn_Rhalf3(
    __clc_size_t args_0, __clc_float16_t const __global *args_1) {
  return __spirv_ocl_vloadn_Rhalf3(args_0,
                                   (__clc_fp16_t const __global *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t __spirv_ocl_vloadn_Rhalf3(
    __clc_size_t args_0, __clc_float16_t const __constant *args_1) {
  return __spirv_ocl_vloadn_Rhalf3(args_0,
                                   (__clc_fp16_t const __constant *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t
__spirv_ocl_vloadn_Rhalf4(__clc_size_t args_0, __clc_float16_t const *args_1) {
  return __spirv_ocl_vloadn_Rhalf4(args_0, (__clc_fp16_t const *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t __spirv_ocl_vloadn_Rhalf4(
    __clc_size_t args_0, __clc_float16_t const __local *args_1) {
  return __spirv_ocl_vloadn_Rhalf4(args_0,
                                   (__clc_fp16_t const __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t __spirv_ocl_vloadn_Rhalf4(
    __clc_size_t args_0, __clc_float16_t const __global *args_1) {
  return __spirv_ocl_vloadn_Rhalf4(args_0,
                                   (__clc_fp16_t const __global *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t __spirv_ocl_vloadn_Rhalf4(
    __clc_size_t args_0, __clc_float16_t const __constant *args_1) {
  return __spirv_ocl_vloadn_Rhalf4(args_0,
                                   (__clc_fp16_t const __constant *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t
__spirv_ocl_vloadn_Rhalf8(__clc_size_t args_0, __clc_float16_t const *args_1) {
  return __spirv_ocl_vloadn_Rhalf8(args_0, (__clc_fp16_t const *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t __spirv_ocl_vloadn_Rhalf8(
    __clc_size_t args_0, __clc_float16_t const __local *args_1) {
  return __spirv_ocl_vloadn_Rhalf8(args_0,
                                   (__clc_fp16_t const __local *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t __spirv_ocl_vloadn_Rhalf8(
    __clc_size_t args_0, __clc_float16_t const __global *args_1) {
  return __spirv_ocl_vloadn_Rhalf8(args_0,
                                   (__clc_fp16_t const __global *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t __spirv_ocl_vloadn_Rhalf8(
    __clc_size_t args_0, __clc_float16_t const __constant *args_1) {
  return __spirv_ocl_vloadn_Rhalf8(args_0,
                                   (__clc_fp16_t const __constant *)(args_1));
}

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstore_half(__clc_fp32_t args_0,
                                                    __clc_size_t args_1,
                                                    __clc_float16_t *args_2) {
  __spirv_ocl_vstore_half(args_0, args_1, (__clc_fp16_t *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_half(__clc_fp32_t args_0, __clc_size_t args_1,
                        __clc_float16_t __local *args_2) {
  __spirv_ocl_vstore_half(args_0, args_1, (__clc_fp16_t __local *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_half(__clc_fp32_t args_0, __clc_size_t args_1,
                        __clc_float16_t __global *args_2) {
  __spirv_ocl_vstore_half(args_0, args_1, (__clc_fp16_t __global *)(args_2));
}

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstore_half(__clc_fp64_t args_0,
                                                    __clc_size_t args_1,
                                                    __clc_float16_t *args_2) {
  __spirv_ocl_vstore_half(args_0, args_1, (__clc_fp16_t *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_half(__clc_fp64_t args_0, __clc_size_t args_1,
                        __clc_float16_t __local *args_2) {
  __spirv_ocl_vstore_half(args_0, args_1, (__clc_fp16_t __local *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_half(__clc_fp64_t args_0, __clc_size_t args_1,
                        __clc_float16_t __global *args_2) {
  __spirv_ocl_vstore_half(args_0, args_1, (__clc_fp16_t __global *)(args_2));
}

#endif

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstore_half_r(__clc_fp32_t args_0,
                                                      __clc_size_t args_1,
                                                      __clc_float16_t *args_2,
                                                      __clc_uint32_t args_3) {
  __spirv_ocl_vstore_half_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_half_r(__clc_fp32_t args_0, __clc_size_t args_1,
                          __clc_float16_t __local *args_2,
                          __clc_uint32_t args_3) {
  __spirv_ocl_vstore_half_r(args_0, args_1, (__clc_fp16_t __local *)(args_2),
                            args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_half_r(__clc_fp32_t args_0, __clc_size_t args_1,
                          __clc_float16_t __global *args_2,
                          __clc_uint32_t args_3) {
  __spirv_ocl_vstore_half_r(args_0, args_1, (__clc_fp16_t __global *)(args_2),
                            args_3);
}

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstore_half_r(__clc_fp64_t args_0,
                                                      __clc_size_t args_1,
                                                      __clc_float16_t *args_2,
                                                      __clc_uint32_t args_3) {
  __spirv_ocl_vstore_half_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_half_r(__clc_fp64_t args_0, __clc_size_t args_1,
                          __clc_float16_t __local *args_2,
                          __clc_uint32_t args_3) {
  __spirv_ocl_vstore_half_r(args_0, args_1, (__clc_fp16_t __local *)(args_2),
                            args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_half_r(__clc_fp64_t args_0, __clc_size_t args_1,
                          __clc_float16_t __global *args_2,
                          __clc_uint32_t args_3) {
  __spirv_ocl_vstore_half_r(args_0, args_1, (__clc_fp16_t __global *)(args_2),
                            args_3);
}

#endif

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstore_halfn(__clc_vec2_fp32_t args_0,
                                                     __clc_size_t args_1,
                                                     __clc_float16_t *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec2_fp32_t args_0, __clc_size_t args_1,
                         __clc_float16_t __local *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t __local *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec2_fp32_t args_0, __clc_size_t args_1,
                         __clc_float16_t __global *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t __global *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstore_halfn(__clc_vec3_fp32_t args_0,
                                                     __clc_size_t args_1,
                                                     __clc_float16_t *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec3_fp32_t args_0, __clc_size_t args_1,
                         __clc_float16_t __local *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t __local *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec3_fp32_t args_0, __clc_size_t args_1,
                         __clc_float16_t __global *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t __global *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstore_halfn(__clc_vec4_fp32_t args_0,
                                                     __clc_size_t args_1,
                                                     __clc_float16_t *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec4_fp32_t args_0, __clc_size_t args_1,
                         __clc_float16_t __local *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t __local *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec4_fp32_t args_0, __clc_size_t args_1,
                         __clc_float16_t __global *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t __global *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstore_halfn(__clc_vec8_fp32_t args_0,
                                                     __clc_size_t args_1,
                                                     __clc_float16_t *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec8_fp32_t args_0, __clc_size_t args_1,
                         __clc_float16_t __local *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t __local *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec8_fp32_t args_0, __clc_size_t args_1,
                         __clc_float16_t __global *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t __global *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstore_halfn(__clc_vec16_fp32_t args_0,
                                                     __clc_size_t args_1,
                                                     __clc_float16_t *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec16_fp32_t args_0, __clc_size_t args_1,
                         __clc_float16_t __local *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t __local *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec16_fp32_t args_0, __clc_size_t args_1,
                         __clc_float16_t __global *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t __global *)(args_2));
}

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstore_halfn(__clc_vec2_fp64_t args_0,
                                                     __clc_size_t args_1,
                                                     __clc_float16_t *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec2_fp64_t args_0, __clc_size_t args_1,
                         __clc_float16_t __local *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t __local *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec2_fp64_t args_0, __clc_size_t args_1,
                         __clc_float16_t __global *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t __global *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstore_halfn(__clc_vec3_fp64_t args_0,
                                                     __clc_size_t args_1,
                                                     __clc_float16_t *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec3_fp64_t args_0, __clc_size_t args_1,
                         __clc_float16_t __local *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t __local *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec3_fp64_t args_0, __clc_size_t args_1,
                         __clc_float16_t __global *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t __global *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstore_halfn(__clc_vec4_fp64_t args_0,
                                                     __clc_size_t args_1,
                                                     __clc_float16_t *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec4_fp64_t args_0, __clc_size_t args_1,
                         __clc_float16_t __local *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t __local *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec4_fp64_t args_0, __clc_size_t args_1,
                         __clc_float16_t __global *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t __global *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstore_halfn(__clc_vec8_fp64_t args_0,
                                                     __clc_size_t args_1,
                                                     __clc_float16_t *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec8_fp64_t args_0, __clc_size_t args_1,
                         __clc_float16_t __local *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t __local *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec8_fp64_t args_0, __clc_size_t args_1,
                         __clc_float16_t __global *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t __global *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstore_halfn(__clc_vec16_fp64_t args_0,
                                                     __clc_size_t args_1,
                                                     __clc_float16_t *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec16_fp64_t args_0, __clc_size_t args_1,
                         __clc_float16_t __local *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t __local *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec16_fp64_t args_0, __clc_size_t args_1,
                         __clc_float16_t __global *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t __global *)(args_2));
}

#endif

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstore_halfn_r(__clc_vec2_fp32_t args_0,
                                                       __clc_size_t args_1,
                                                       __clc_float16_t *args_2,
                                                       __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec2_fp32_t args_0, __clc_size_t args_1,
                           __clc_float16_t __local *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t __local *)(args_2),
                             args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec2_fp32_t args_0, __clc_size_t args_1,
                           __clc_float16_t __global *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t __global *)(args_2),
                             args_3);
}

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstore_halfn_r(__clc_vec3_fp32_t args_0,
                                                       __clc_size_t args_1,
                                                       __clc_float16_t *args_2,
                                                       __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec3_fp32_t args_0, __clc_size_t args_1,
                           __clc_float16_t __local *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t __local *)(args_2),
                             args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec3_fp32_t args_0, __clc_size_t args_1,
                           __clc_float16_t __global *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t __global *)(args_2),
                             args_3);
}

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstore_halfn_r(__clc_vec4_fp32_t args_0,
                                                       __clc_size_t args_1,
                                                       __clc_float16_t *args_2,
                                                       __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec4_fp32_t args_0, __clc_size_t args_1,
                           __clc_float16_t __local *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t __local *)(args_2),
                             args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec4_fp32_t args_0, __clc_size_t args_1,
                           __clc_float16_t __global *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t __global *)(args_2),
                             args_3);
}

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstore_halfn_r(__clc_vec8_fp32_t args_0,
                                                       __clc_size_t args_1,
                                                       __clc_float16_t *args_2,
                                                       __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec8_fp32_t args_0, __clc_size_t args_1,
                           __clc_float16_t __local *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t __local *)(args_2),
                             args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec8_fp32_t args_0, __clc_size_t args_1,
                           __clc_float16_t __global *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t __global *)(args_2),
                             args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec16_fp32_t args_0, __clc_size_t args_1,
                           __clc_float16_t *args_2, __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec16_fp32_t args_0, __clc_size_t args_1,
                           __clc_float16_t __local *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t __local *)(args_2),
                             args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec16_fp32_t args_0, __clc_size_t args_1,
                           __clc_float16_t __global *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t __global *)(args_2),
                             args_3);
}

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstore_halfn_r(__clc_vec2_fp64_t args_0,
                                                       __clc_size_t args_1,
                                                       __clc_float16_t *args_2,
                                                       __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec2_fp64_t args_0, __clc_size_t args_1,
                           __clc_float16_t __local *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t __local *)(args_2),
                             args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec2_fp64_t args_0, __clc_size_t args_1,
                           __clc_float16_t __global *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t __global *)(args_2),
                             args_3);
}

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstore_halfn_r(__clc_vec3_fp64_t args_0,
                                                       __clc_size_t args_1,
                                                       __clc_float16_t *args_2,
                                                       __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec3_fp64_t args_0, __clc_size_t args_1,
                           __clc_float16_t __local *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t __local *)(args_2),
                             args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec3_fp64_t args_0, __clc_size_t args_1,
                           __clc_float16_t __global *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t __global *)(args_2),
                             args_3);
}

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstore_halfn_r(__clc_vec4_fp64_t args_0,
                                                       __clc_size_t args_1,
                                                       __clc_float16_t *args_2,
                                                       __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec4_fp64_t args_0, __clc_size_t args_1,
                           __clc_float16_t __local *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t __local *)(args_2),
                             args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec4_fp64_t args_0, __clc_size_t args_1,
                           __clc_float16_t __global *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t __global *)(args_2),
                             args_3);
}

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstore_halfn_r(__clc_vec8_fp64_t args_0,
                                                       __clc_size_t args_1,
                                                       __clc_float16_t *args_2,
                                                       __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec8_fp64_t args_0, __clc_size_t args_1,
                           __clc_float16_t __local *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t __local *)(args_2),
                             args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec8_fp64_t args_0, __clc_size_t args_1,
                           __clc_float16_t __global *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t __global *)(args_2),
                             args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec16_fp64_t args_0, __clc_size_t args_1,
                           __clc_float16_t *args_2, __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec16_fp64_t args_0, __clc_size_t args_1,
                           __clc_float16_t __local *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t __local *)(args_2),
                             args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec16_fp64_t args_0, __clc_size_t args_1,
                           __clc_float16_t __global *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t __global *)(args_2),
                             args_3);
}

#endif

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstorea_halfn(__clc_vec2_fp32_t args_0,
                                                      __clc_size_t args_1,
                                                      __clc_float16_t *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec2_fp32_t args_0, __clc_size_t args_1,
                          __clc_float16_t __local *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t __local *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec2_fp32_t args_0, __clc_size_t args_1,
                          __clc_float16_t __global *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t __global *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstorea_halfn(__clc_vec3_fp32_t args_0,
                                                      __clc_size_t args_1,
                                                      __clc_float16_t *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec3_fp32_t args_0, __clc_size_t args_1,
                          __clc_float16_t __local *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t __local *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec3_fp32_t args_0, __clc_size_t args_1,
                          __clc_float16_t __global *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t __global *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstorea_halfn(__clc_vec4_fp32_t args_0,
                                                      __clc_size_t args_1,
                                                      __clc_float16_t *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec4_fp32_t args_0, __clc_size_t args_1,
                          __clc_float16_t __local *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t __local *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec4_fp32_t args_0, __clc_size_t args_1,
                          __clc_float16_t __global *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t __global *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstorea_halfn(__clc_vec8_fp32_t args_0,
                                                      __clc_size_t args_1,
                                                      __clc_float16_t *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec8_fp32_t args_0, __clc_size_t args_1,
                          __clc_float16_t __local *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t __local *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec8_fp32_t args_0, __clc_size_t args_1,
                          __clc_float16_t __global *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t __global *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstorea_halfn(__clc_vec16_fp32_t args_0,
                                                      __clc_size_t args_1,
                                                      __clc_float16_t *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec16_fp32_t args_0, __clc_size_t args_1,
                          __clc_float16_t __local *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t __local *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec16_fp32_t args_0, __clc_size_t args_1,
                          __clc_float16_t __global *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t __global *)(args_2));
}

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstorea_halfn(__clc_vec2_fp64_t args_0,
                                                      __clc_size_t args_1,
                                                      __clc_float16_t *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec2_fp64_t args_0, __clc_size_t args_1,
                          __clc_float16_t __local *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t __local *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec2_fp64_t args_0, __clc_size_t args_1,
                          __clc_float16_t __global *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t __global *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstorea_halfn(__clc_vec3_fp64_t args_0,
                                                      __clc_size_t args_1,
                                                      __clc_float16_t *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec3_fp64_t args_0, __clc_size_t args_1,
                          __clc_float16_t __local *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t __local *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec3_fp64_t args_0, __clc_size_t args_1,
                          __clc_float16_t __global *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t __global *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstorea_halfn(__clc_vec4_fp64_t args_0,
                                                      __clc_size_t args_1,
                                                      __clc_float16_t *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec4_fp64_t args_0, __clc_size_t args_1,
                          __clc_float16_t __local *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t __local *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec4_fp64_t args_0, __clc_size_t args_1,
                          __clc_float16_t __global *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t __global *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstorea_halfn(__clc_vec8_fp64_t args_0,
                                                      __clc_size_t args_1,
                                                      __clc_float16_t *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec8_fp64_t args_0, __clc_size_t args_1,
                          __clc_float16_t __local *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t __local *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec8_fp64_t args_0, __clc_size_t args_1,
                          __clc_float16_t __global *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t __global *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstorea_halfn(__clc_vec16_fp64_t args_0,
                                                      __clc_size_t args_1,
                                                      __clc_float16_t *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec16_fp64_t args_0, __clc_size_t args_1,
                          __clc_float16_t __local *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t __local *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec16_fp64_t args_0, __clc_size_t args_1,
                          __clc_float16_t __global *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t __global *)(args_2));
}

#endif

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec2_fp32_t args_0, __clc_size_t args_1,
                            __clc_float16_t *args_2, __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec2_fp32_t args_0, __clc_size_t args_1,
                            __clc_float16_t __local *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t __local *)(args_2),
                              args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec2_fp32_t args_0, __clc_size_t args_1,
                            __clc_float16_t __global *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t __global *)(args_2),
                              args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec3_fp32_t args_0, __clc_size_t args_1,
                            __clc_float16_t *args_2, __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec3_fp32_t args_0, __clc_size_t args_1,
                            __clc_float16_t __local *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t __local *)(args_2),
                              args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec3_fp32_t args_0, __clc_size_t args_1,
                            __clc_float16_t __global *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t __global *)(args_2),
                              args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec4_fp32_t args_0, __clc_size_t args_1,
                            __clc_float16_t *args_2, __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec4_fp32_t args_0, __clc_size_t args_1,
                            __clc_float16_t __local *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t __local *)(args_2),
                              args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec4_fp32_t args_0, __clc_size_t args_1,
                            __clc_float16_t __global *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t __global *)(args_2),
                              args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec8_fp32_t args_0, __clc_size_t args_1,
                            __clc_float16_t *args_2, __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec8_fp32_t args_0, __clc_size_t args_1,
                            __clc_float16_t __local *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t __local *)(args_2),
                              args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec8_fp32_t args_0, __clc_size_t args_1,
                            __clc_float16_t __global *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t __global *)(args_2),
                              args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec16_fp32_t args_0, __clc_size_t args_1,
                            __clc_float16_t *args_2, __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec16_fp32_t args_0, __clc_size_t args_1,
                            __clc_float16_t __local *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t __local *)(args_2),
                              args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec16_fp32_t args_0, __clc_size_t args_1,
                            __clc_float16_t __global *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t __global *)(args_2),
                              args_3);
}

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec2_fp64_t args_0, __clc_size_t args_1,
                            __clc_float16_t *args_2, __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec2_fp64_t args_0, __clc_size_t args_1,
                            __clc_float16_t __local *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t __local *)(args_2),
                              args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec2_fp64_t args_0, __clc_size_t args_1,
                            __clc_float16_t __global *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t __global *)(args_2),
                              args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec3_fp64_t args_0, __clc_size_t args_1,
                            __clc_float16_t *args_2, __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec3_fp64_t args_0, __clc_size_t args_1,
                            __clc_float16_t __local *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t __local *)(args_2),
                              args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec3_fp64_t args_0, __clc_size_t args_1,
                            __clc_float16_t __global *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t __global *)(args_2),
                              args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec4_fp64_t args_0, __clc_size_t args_1,
                            __clc_float16_t *args_2, __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec4_fp64_t args_0, __clc_size_t args_1,
                            __clc_float16_t __local *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t __local *)(args_2),
                              args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec4_fp64_t args_0, __clc_size_t args_1,
                            __clc_float16_t __global *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t __global *)(args_2),
                              args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec8_fp64_t args_0, __clc_size_t args_1,
                            __clc_float16_t *args_2, __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec8_fp64_t args_0, __clc_size_t args_1,
                            __clc_float16_t __local *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t __local *)(args_2),
                              args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec8_fp64_t args_0, __clc_size_t args_1,
                            __clc_float16_t __global *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t __global *)(args_2),
                              args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec16_fp64_t args_0, __clc_size_t args_1,
                            __clc_float16_t *args_2, __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec16_fp64_t args_0, __clc_size_t args_1,
                            __clc_float16_t __local *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t __local *)(args_2),
                              args_3);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec16_fp64_t args_0, __clc_size_t args_1,
                            __clc_float16_t __global *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t __global *)(args_2),
                              args_3);
}

#endif

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstoren(__clc_vec2_float16_t args_0,
                                                __clc_size_t args_1,
                                                __clc_float16_t *args_2) {
  __spirv_ocl_vstoren(__clc_as_half2(args_0), args_1, (__clc_fp16_t *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec2_float16_t args_0, __clc_size_t args_1,
                    __clc_float16_t __local *args_2) {
  __spirv_ocl_vstoren(__clc_as_half2(args_0), args_1,
                      (__clc_fp16_t __local *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec2_float16_t args_0, __clc_size_t args_1,
                    __clc_float16_t __global *args_2) {
  __spirv_ocl_vstoren(__clc_as_half2(args_0), args_1,
                      (__clc_fp16_t __global *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstoren(__clc_vec3_float16_t args_0,
                                                __clc_size_t args_1,
                                                __clc_float16_t *args_2) {
  __spirv_ocl_vstoren(__clc_as_half3(args_0), args_1, (__clc_fp16_t *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec3_float16_t args_0, __clc_size_t args_1,
                    __clc_float16_t __local *args_2) {
  __spirv_ocl_vstoren(__clc_as_half3(args_0), args_1,
                      (__clc_fp16_t __local *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec3_float16_t args_0, __clc_size_t args_1,
                    __clc_float16_t __global *args_2) {
  __spirv_ocl_vstoren(__clc_as_half3(args_0), args_1,
                      (__clc_fp16_t __global *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstoren(__clc_vec4_float16_t args_0,
                                                __clc_size_t args_1,
                                                __clc_float16_t *args_2) {
  __spirv_ocl_vstoren(__clc_as_half4(args_0), args_1, (__clc_fp16_t *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec4_float16_t args_0, __clc_size_t args_1,
                    __clc_float16_t __local *args_2) {
  __spirv_ocl_vstoren(__clc_as_half4(args_0), args_1,
                      (__clc_fp16_t __local *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec4_float16_t args_0, __clc_size_t args_1,
                    __clc_float16_t __global *args_2) {
  __spirv_ocl_vstoren(__clc_as_half4(args_0), args_1,
                      (__clc_fp16_t __global *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstoren(__clc_vec8_float16_t args_0,
                                                __clc_size_t args_1,
                                                __clc_float16_t *args_2) {
  __spirv_ocl_vstoren(__clc_as_half8(args_0), args_1, (__clc_fp16_t *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec8_float16_t args_0, __clc_size_t args_1,
                    __clc_float16_t __local *args_2) {
  __spirv_ocl_vstoren(__clc_as_half8(args_0), args_1,
                      (__clc_fp16_t __local *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec8_float16_t args_0, __clc_size_t args_1,
                    __clc_float16_t __global *args_2) {
  __spirv_ocl_vstoren(__clc_as_half8(args_0), args_1,
                      (__clc_fp16_t __global *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void __spirv_ocl_vstoren(__clc_vec16_float16_t args_0,
                                                __clc_size_t args_1,
                                                __clc_float16_t *args_2) {
  __spirv_ocl_vstoren(__clc_as_half16(args_0), args_1, (__clc_fp16_t *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec16_float16_t args_0, __clc_size_t args_1,
                    __clc_float16_t __local *args_2) {
  __spirv_ocl_vstoren(__clc_as_half16(args_0), args_1,
                      (__clc_fp16_t __local *)(args_2));
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec16_float16_t args_0, __clc_size_t args_1,
                    __clc_float16_t __global *args_2) {
  __spirv_ocl_vstoren(__clc_as_half16(args_0), args_1,
                      (__clc_fp16_t __global *)(args_2));
}

#endif
#endif
