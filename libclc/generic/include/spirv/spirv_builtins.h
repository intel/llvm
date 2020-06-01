//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <func.h>
#include <spirv/spirv_types.h>

#ifndef CLC_SPIRV_BINDING
#define CLC_SPIRV_BINDING

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int8_t __spirv_ConvertFToS_Rchar(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int8_t __spirv_ConvertFToS_Rchar(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int8_t __spirv_ConvertFToS_Rchar(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16_rte(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16_rte(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16_rte(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16_rtn(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16_rtn(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16_rtn(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16_rtp(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16_rtp(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16_rtp(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16_rtz(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16_rtz(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16_rtz(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16_sat(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16_sat(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16_sat(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16_sat_rte(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16_sat_rte(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16_sat_rte(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16_sat_rtn(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16_sat_rtn(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16_sat_rtn(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16_sat_rtp(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16_sat_rtp(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16_sat_rtp(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16_sat_rtz(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16_sat_rtz(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ConvertFToS_Rchar16_sat_rtz(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2_rte(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2_rte(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2_rte(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2_rtn(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2_rtn(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2_rtn(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2_rtp(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2_rtp(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2_rtp(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2_rtz(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2_rtz(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2_rtz(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2_sat(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2_sat(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2_sat(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2_sat_rte(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2_sat_rte(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2_sat_rte(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2_sat_rtn(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2_sat_rtn(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2_sat_rtn(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2_sat_rtp(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2_sat_rtp(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2_sat_rtp(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2_sat_rtz(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2_sat_rtz(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ConvertFToS_Rchar2_sat_rtz(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3_rte(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3_rte(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3_rte(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3_rtn(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3_rtn(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3_rtn(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3_rtp(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3_rtp(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3_rtp(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3_rtz(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3_rtz(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3_rtz(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3_sat(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3_sat(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3_sat(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3_sat_rte(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3_sat_rte(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3_sat_rte(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3_sat_rtn(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3_sat_rtn(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3_sat_rtn(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3_sat_rtp(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3_sat_rtp(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3_sat_rtp(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3_sat_rtz(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3_sat_rtz(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ConvertFToS_Rchar3_sat_rtz(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4_rte(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4_rte(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4_rte(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4_rtn(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4_rtn(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4_rtn(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4_rtp(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4_rtp(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4_rtp(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4_rtz(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4_rtz(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4_rtz(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4_sat(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4_sat(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4_sat(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4_sat_rte(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4_sat_rte(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4_sat_rte(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4_sat_rtn(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4_sat_rtn(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4_sat_rtn(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4_sat_rtp(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4_sat_rtp(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4_sat_rtp(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4_sat_rtz(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4_sat_rtz(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ConvertFToS_Rchar4_sat_rtz(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8_rte(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8_rte(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8_rte(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8_rtn(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8_rtn(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8_rtn(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8_rtp(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8_rtp(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8_rtp(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8_rtz(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8_rtz(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8_rtz(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8_sat(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8_sat(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8_sat(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8_sat_rte(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8_sat_rte(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8_sat_rte(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8_sat_rtn(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8_sat_rtn(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8_sat_rtn(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8_sat_rtp(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8_sat_rtp(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8_sat_rtp(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8_sat_rtz(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8_sat_rtz(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ConvertFToS_Rchar8_sat_rtz(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ConvertFToS_Rchar_rte(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ConvertFToS_Rchar_rte(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ConvertFToS_Rchar_rte(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ConvertFToS_Rchar_rtn(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ConvertFToS_Rchar_rtn(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ConvertFToS_Rchar_rtn(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ConvertFToS_Rchar_rtp(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ConvertFToS_Rchar_rtp(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ConvertFToS_Rchar_rtp(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ConvertFToS_Rchar_rtz(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ConvertFToS_Rchar_rtz(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ConvertFToS_Rchar_rtz(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ConvertFToS_Rchar_sat(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ConvertFToS_Rchar_sat(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ConvertFToS_Rchar_sat(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ConvertFToS_Rchar_sat_rte(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ConvertFToS_Rchar_sat_rte(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ConvertFToS_Rchar_sat_rte(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ConvertFToS_Rchar_sat_rtn(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ConvertFToS_Rchar_sat_rtn(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ConvertFToS_Rchar_sat_rtn(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ConvertFToS_Rchar_sat_rtp(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ConvertFToS_Rchar_sat_rtp(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ConvertFToS_Rchar_sat_rtp(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ConvertFToS_Rchar_sat_rtz(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ConvertFToS_Rchar_sat_rtz(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ConvertFToS_Rchar_sat_rtz(__clc_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int32_t __spirv_ConvertFToS_Rint(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int32_t __spirv_ConvertFToS_Rint(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int32_t __spirv_ConvertFToS_Rint(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16_rte(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16_rte(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16_rte(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16_rtn(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16_rtn(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16_rtn(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16_rtp(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16_rtp(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16_rtp(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16_rtz(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16_rtz(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16_rtz(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16_sat(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16_sat(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16_sat(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16_sat_rte(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16_sat_rte(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16_sat_rte(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16_sat_rtn(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16_sat_rtn(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16_sat_rtn(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16_sat_rtp(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16_sat_rtp(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16_sat_rtp(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16_sat_rtz(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16_sat_rtz(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ConvertFToS_Rint16_sat_rtz(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2_rte(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2_rte(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2_rte(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2_rtn(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2_rtn(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2_rtn(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2_rtp(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2_rtp(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2_rtp(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2_rtz(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2_rtz(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2_rtz(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2_sat(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2_sat(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2_sat(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2_sat_rte(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2_sat_rte(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2_sat_rte(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2_sat_rtn(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2_sat_rtn(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2_sat_rtn(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2_sat_rtp(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2_sat_rtp(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2_sat_rtp(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2_sat_rtz(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2_sat_rtz(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ConvertFToS_Rint2_sat_rtz(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3_rte(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3_rte(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3_rte(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3_rtn(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3_rtn(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3_rtn(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3_rtp(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3_rtp(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3_rtp(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3_rtz(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3_rtz(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3_rtz(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3_sat(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3_sat(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3_sat(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3_sat_rte(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3_sat_rte(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3_sat_rte(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3_sat_rtn(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3_sat_rtn(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3_sat_rtn(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3_sat_rtp(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3_sat_rtp(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3_sat_rtp(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3_sat_rtz(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3_sat_rtz(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ConvertFToS_Rint3_sat_rtz(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4_rte(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4_rte(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4_rte(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4_rtn(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4_rtn(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4_rtn(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4_rtp(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4_rtp(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4_rtp(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4_rtz(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4_rtz(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4_rtz(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4_sat(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4_sat(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4_sat(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4_sat_rte(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4_sat_rte(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4_sat_rte(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4_sat_rtn(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4_sat_rtn(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4_sat_rtn(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4_sat_rtp(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4_sat_rtp(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4_sat_rtp(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4_sat_rtz(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4_sat_rtz(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ConvertFToS_Rint4_sat_rtz(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8_rte(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8_rte(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8_rte(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8_rtn(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8_rtn(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8_rtn(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8_rtp(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8_rtp(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8_rtp(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8_rtz(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8_rtz(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8_rtz(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8_sat(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8_sat(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8_sat(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8_sat_rte(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8_sat_rte(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8_sat_rte(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8_sat_rtn(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8_sat_rtn(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8_sat_rtn(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8_sat_rtp(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8_sat_rtp(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8_sat_rtp(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8_sat_rtz(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8_sat_rtz(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ConvertFToS_Rint8_sat_rtz(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ConvertFToS_Rint_rte(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ConvertFToS_Rint_rte(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ConvertFToS_Rint_rte(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ConvertFToS_Rint_rtn(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ConvertFToS_Rint_rtn(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ConvertFToS_Rint_rtn(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ConvertFToS_Rint_rtp(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ConvertFToS_Rint_rtp(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ConvertFToS_Rint_rtp(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ConvertFToS_Rint_rtz(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ConvertFToS_Rint_rtz(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ConvertFToS_Rint_rtz(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ConvertFToS_Rint_sat(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ConvertFToS_Rint_sat(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ConvertFToS_Rint_sat(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ConvertFToS_Rint_sat_rte(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ConvertFToS_Rint_sat_rte(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ConvertFToS_Rint_sat_rte(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ConvertFToS_Rint_sat_rtn(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ConvertFToS_Rint_sat_rtn(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ConvertFToS_Rint_sat_rtn(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ConvertFToS_Rint_sat_rtp(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ConvertFToS_Rint_sat_rtp(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ConvertFToS_Rint_sat_rtp(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ConvertFToS_Rint_sat_rtz(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ConvertFToS_Rint_sat_rtz(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ConvertFToS_Rint_sat_rtz(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16_rte(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16_rte(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16_rte(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16_rtn(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16_rtn(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16_rtn(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16_rtp(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16_rtp(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16_rtp(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16_rtz(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16_rtz(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16_rtz(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16_sat(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16_sat(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16_sat(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16_sat_rte(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16_sat_rte(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16_sat_rte(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16_sat_rtn(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16_sat_rtn(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16_sat_rtn(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16_sat_rtp(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16_sat_rtp(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16_sat_rtp(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16_sat_rtz(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16_sat_rtz(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ConvertFToS_Rlong16_sat_rtz(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2_rte(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2_rte(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2_rte(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2_rtn(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2_rtn(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2_rtn(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2_rtp(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2_rtp(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2_rtp(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2_rtz(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2_rtz(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2_rtz(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2_sat(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2_sat(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2_sat(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2_sat_rte(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2_sat_rte(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2_sat_rte(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2_sat_rtn(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2_sat_rtn(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2_sat_rtn(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2_sat_rtp(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2_sat_rtp(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2_sat_rtp(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2_sat_rtz(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2_sat_rtz(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ConvertFToS_Rlong2_sat_rtz(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3_rte(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3_rte(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3_rte(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3_rtn(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3_rtn(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3_rtn(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3_rtp(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3_rtp(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3_rtp(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3_rtz(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3_rtz(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3_rtz(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3_sat(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3_sat(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3_sat(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3_sat_rte(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3_sat_rte(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3_sat_rte(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3_sat_rtn(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3_sat_rtn(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3_sat_rtn(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3_sat_rtp(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3_sat_rtp(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3_sat_rtp(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3_sat_rtz(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3_sat_rtz(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ConvertFToS_Rlong3_sat_rtz(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4_rte(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4_rte(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4_rte(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4_rtn(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4_rtn(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4_rtn(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4_rtp(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4_rtp(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4_rtp(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4_rtz(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4_rtz(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4_rtz(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4_sat(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4_sat(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4_sat(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4_sat_rte(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4_sat_rte(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4_sat_rte(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4_sat_rtn(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4_sat_rtn(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4_sat_rtn(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4_sat_rtp(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4_sat_rtp(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4_sat_rtp(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4_sat_rtz(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4_sat_rtz(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ConvertFToS_Rlong4_sat_rtz(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8_rte(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8_rte(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8_rte(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8_rtn(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8_rtn(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8_rtn(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8_rtp(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8_rtp(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8_rtp(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8_rtz(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8_rtz(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8_rtz(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8_sat(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8_sat(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8_sat(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8_sat_rte(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8_sat_rte(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8_sat_rte(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8_sat_rtn(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8_sat_rtn(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8_sat_rtn(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8_sat_rtp(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8_sat_rtp(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8_sat_rtp(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8_sat_rtz(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8_sat_rtz(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ConvertFToS_Rlong8_sat_rtz(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong_rte(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong_rte(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong_rte(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong_rtn(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong_rtn(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong_rtn(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong_rtp(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong_rtp(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong_rtp(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong_rtz(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong_rtz(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong_rtz(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong_sat(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong_sat(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong_sat(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong_sat_rte(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong_sat_rte(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong_sat_rte(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong_sat_rtn(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong_sat_rtn(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong_sat_rtn(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong_sat_rtp(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong_sat_rtp(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong_sat_rtp(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong_sat_rtz(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong_sat_rtz(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ConvertFToS_Rlong_sat_rtz(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16_rte(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16_rte(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16_rte(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16_rtn(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16_rtn(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16_rtn(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16_rtp(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16_rtp(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16_rtp(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16_rtz(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16_rtz(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16_rtz(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16_sat(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16_sat(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16_sat(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16_sat_rte(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16_sat_rte(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16_sat_rte(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16_sat_rtn(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16_sat_rtn(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16_sat_rtn(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16_sat_rtp(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16_sat_rtp(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16_sat_rtp(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16_sat_rtz(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16_sat_rtz(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ConvertFToS_Rshort16_sat_rtz(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2_rte(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2_rte(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2_rte(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2_rtn(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2_rtn(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2_rtn(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2_rtp(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2_rtp(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2_rtp(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2_rtz(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2_rtz(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2_rtz(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2_sat(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2_sat(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2_sat(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2_sat_rte(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2_sat_rte(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2_sat_rte(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2_sat_rtn(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2_sat_rtn(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2_sat_rtn(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2_sat_rtp(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2_sat_rtp(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2_sat_rtp(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2_sat_rtz(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2_sat_rtz(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ConvertFToS_Rshort2_sat_rtz(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3_rte(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3_rte(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3_rte(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3_rtn(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3_rtn(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3_rtn(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3_rtp(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3_rtp(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3_rtp(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3_rtz(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3_rtz(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3_rtz(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3_sat(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3_sat(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3_sat(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3_sat_rte(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3_sat_rte(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3_sat_rte(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3_sat_rtn(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3_sat_rtn(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3_sat_rtn(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3_sat_rtp(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3_sat_rtp(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3_sat_rtp(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3_sat_rtz(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3_sat_rtz(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ConvertFToS_Rshort3_sat_rtz(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4_rte(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4_rte(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4_rte(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4_rtn(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4_rtn(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4_rtn(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4_rtp(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4_rtp(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4_rtp(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4_rtz(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4_rtz(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4_rtz(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4_sat(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4_sat(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4_sat(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4_sat_rte(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4_sat_rte(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4_sat_rte(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4_sat_rtn(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4_sat_rtn(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4_sat_rtn(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4_sat_rtp(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4_sat_rtp(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4_sat_rtp(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4_sat_rtz(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4_sat_rtz(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ConvertFToS_Rshort4_sat_rtz(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8_rte(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8_rte(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8_rte(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8_rtn(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8_rtn(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8_rtn(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8_rtp(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8_rtp(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8_rtp(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8_rtz(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8_rtz(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8_rtz(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8_sat(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8_sat(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8_sat(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8_sat_rte(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8_sat_rte(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8_sat_rte(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8_sat_rtn(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8_sat_rtn(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8_sat_rtn(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8_sat_rtp(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8_sat_rtp(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8_sat_rtp(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8_sat_rtz(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8_sat_rtz(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ConvertFToS_Rshort8_sat_rtz(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort_rte(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort_rte(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort_rte(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort_rtn(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort_rtn(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort_rtn(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort_rtp(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort_rtp(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort_rtp(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort_rtz(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort_rtz(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort_rtz(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort_sat(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort_sat(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort_sat(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort_sat_rte(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort_sat_rte(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort_sat_rte(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort_sat_rtn(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort_sat_rtn(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort_sat_rtn(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort_sat_rtp(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort_sat_rtp(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort_sat_rtp(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort_sat_rtz(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort_sat_rtz(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ConvertFToS_Rshort_sat_rtz(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16_rte(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16_rte(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16_rte(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16_rtn(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16_rtn(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16_rtn(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16_rtp(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16_rtp(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16_rtp(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16_rtz(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16_rtz(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16_rtz(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16_sat(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16_sat(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16_sat(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16_sat_rte(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16_sat_rte(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16_sat_rte(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16_sat_rtn(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16_sat_rtn(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16_sat_rtn(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16_sat_rtp(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16_sat_rtp(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16_sat_rtp(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16_sat_rtz(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16_sat_rtz(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ConvertFToU_Ruchar16_sat_rtz(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2_rte(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2_rte(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2_rte(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2_rtn(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2_rtn(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2_rtn(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2_rtp(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2_rtp(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2_rtp(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2_rtz(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2_rtz(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2_rtz(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2_sat(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2_sat(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2_sat(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2_sat_rte(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2_sat_rte(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2_sat_rte(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2_sat_rtn(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2_sat_rtn(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2_sat_rtn(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2_sat_rtp(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2_sat_rtp(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2_sat_rtp(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2_sat_rtz(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2_sat_rtz(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ConvertFToU_Ruchar2_sat_rtz(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3_rte(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3_rte(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3_rte(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3_rtn(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3_rtn(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3_rtn(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3_rtp(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3_rtp(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3_rtp(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3_rtz(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3_rtz(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3_rtz(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3_sat(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3_sat(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3_sat(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3_sat_rte(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3_sat_rte(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3_sat_rte(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3_sat_rtn(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3_sat_rtn(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3_sat_rtn(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3_sat_rtp(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3_sat_rtp(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3_sat_rtp(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3_sat_rtz(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3_sat_rtz(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ConvertFToU_Ruchar3_sat_rtz(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4_rte(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4_rte(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4_rte(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4_rtn(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4_rtn(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4_rtn(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4_rtp(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4_rtp(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4_rtp(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4_rtz(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4_rtz(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4_rtz(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4_sat(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4_sat(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4_sat(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4_sat_rte(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4_sat_rte(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4_sat_rte(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4_sat_rtn(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4_sat_rtn(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4_sat_rtn(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4_sat_rtp(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4_sat_rtp(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4_sat_rtp(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4_sat_rtz(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4_sat_rtz(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ConvertFToU_Ruchar4_sat_rtz(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8_rte(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8_rte(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8_rte(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8_rtn(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8_rtn(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8_rtn(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8_rtp(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8_rtp(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8_rtp(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8_rtz(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8_rtz(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8_rtz(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8_sat(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8_sat(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8_sat(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8_sat_rte(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8_sat_rte(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8_sat_rte(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8_sat_rtn(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8_sat_rtn(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8_sat_rtn(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8_sat_rtp(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8_sat_rtp(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8_sat_rtp(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8_sat_rtz(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8_sat_rtz(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ConvertFToU_Ruchar8_sat_rtz(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar_rte(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar_rte(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar_rte(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar_rtn(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar_rtn(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar_rtn(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar_rtp(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar_rtp(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar_rtp(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar_rtz(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar_rtz(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar_rtz(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar_sat(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar_sat(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar_sat(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar_sat_rte(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar_sat_rte(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar_sat_rte(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar_sat_rtn(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar_sat_rtn(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar_sat_rtn(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar_sat_rtp(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar_sat_rtp(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar_sat_rtp(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar_sat_rtz(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar_sat_rtz(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ConvertFToU_Ruchar_sat_rtz(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16_rte(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16_rte(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16_rte(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16_rtn(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16_rtn(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16_rtn(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16_rtp(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16_rtp(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16_rtp(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16_rtz(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16_rtz(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16_rtz(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16_sat(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16_sat(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16_sat(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16_sat_rte(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16_sat_rte(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16_sat_rte(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16_sat_rtn(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16_sat_rtn(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16_sat_rtn(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16_sat_rtp(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16_sat_rtp(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16_sat_rtp(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16_sat_rtz(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16_sat_rtz(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ConvertFToU_Ruint16_sat_rtz(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2_rte(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2_rte(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2_rte(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2_rtn(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2_rtn(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2_rtn(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2_rtp(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2_rtp(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2_rtp(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2_rtz(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2_rtz(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2_rtz(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2_sat(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2_sat(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2_sat(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2_sat_rte(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2_sat_rte(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2_sat_rte(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2_sat_rtn(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2_sat_rtn(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2_sat_rtn(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2_sat_rtp(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2_sat_rtp(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2_sat_rtp(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2_sat_rtz(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2_sat_rtz(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ConvertFToU_Ruint2_sat_rtz(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3_rte(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3_rte(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3_rte(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3_rtn(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3_rtn(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3_rtn(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3_rtp(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3_rtp(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3_rtp(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3_rtz(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3_rtz(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3_rtz(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3_sat(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3_sat(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3_sat(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3_sat_rte(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3_sat_rte(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3_sat_rte(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3_sat_rtn(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3_sat_rtn(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3_sat_rtn(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3_sat_rtp(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3_sat_rtp(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3_sat_rtp(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3_sat_rtz(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3_sat_rtz(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ConvertFToU_Ruint3_sat_rtz(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4_rte(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4_rte(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4_rte(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4_rtn(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4_rtn(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4_rtn(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4_rtp(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4_rtp(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4_rtp(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4_rtz(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4_rtz(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4_rtz(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4_sat(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4_sat(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4_sat(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4_sat_rte(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4_sat_rte(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4_sat_rte(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4_sat_rtn(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4_sat_rtn(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4_sat_rtn(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4_sat_rtp(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4_sat_rtp(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4_sat_rtp(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4_sat_rtz(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4_sat_rtz(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ConvertFToU_Ruint4_sat_rtz(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8_rte(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8_rte(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8_rte(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8_rtn(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8_rtn(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8_rtn(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8_rtp(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8_rtp(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8_rtp(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8_rtz(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8_rtz(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8_rtz(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8_sat(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8_sat(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8_sat(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8_sat_rte(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8_sat_rte(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8_sat_rte(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8_sat_rtn(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8_sat_rtn(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8_sat_rtn(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8_sat_rtp(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8_sat_rtp(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8_sat_rtp(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8_sat_rtz(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8_sat_rtz(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ConvertFToU_Ruint8_sat_rtz(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint_rte(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint_rte(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint_rte(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint_rtn(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint_rtn(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint_rtn(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint_rtp(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint_rtp(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint_rtp(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint_rtz(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint_rtz(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint_rtz(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint_sat(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint_sat(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint_sat(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint_sat_rte(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint_sat_rte(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint_sat_rte(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint_sat_rtn(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint_sat_rtn(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint_sat_rtn(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint_sat_rtp(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint_sat_rtp(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint_sat_rtp(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint_sat_rtz(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint_sat_rtz(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ConvertFToU_Ruint_sat_rtz(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16_rte(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16_rte(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16_rte(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16_rtn(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16_rtn(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16_rtn(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16_rtp(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16_rtp(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16_rtp(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16_rtz(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16_rtz(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16_rtz(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16_sat(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16_sat(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16_sat(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16_sat_rte(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16_sat_rte(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16_sat_rte(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16_sat_rtn(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16_sat_rtn(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16_sat_rtn(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16_sat_rtp(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16_sat_rtp(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16_sat_rtp(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16_sat_rtz(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16_sat_rtz(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ConvertFToU_Rulong16_sat_rtz(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2_rte(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2_rte(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2_rte(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2_rtn(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2_rtn(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2_rtn(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2_rtp(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2_rtp(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2_rtp(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2_rtz(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2_rtz(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2_rtz(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2_sat(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2_sat(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2_sat(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2_sat_rte(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2_sat_rte(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2_sat_rte(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2_sat_rtn(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2_sat_rtn(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2_sat_rtn(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2_sat_rtp(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2_sat_rtp(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2_sat_rtp(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2_sat_rtz(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2_sat_rtz(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ConvertFToU_Rulong2_sat_rtz(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3_rte(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3_rte(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3_rte(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3_rtn(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3_rtn(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3_rtn(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3_rtp(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3_rtp(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3_rtp(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3_rtz(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3_rtz(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3_rtz(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3_sat(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3_sat(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3_sat(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3_sat_rte(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3_sat_rte(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3_sat_rte(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3_sat_rtn(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3_sat_rtn(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3_sat_rtn(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3_sat_rtp(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3_sat_rtp(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3_sat_rtp(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3_sat_rtz(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3_sat_rtz(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ConvertFToU_Rulong3_sat_rtz(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4_rte(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4_rte(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4_rte(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4_rtn(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4_rtn(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4_rtn(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4_rtp(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4_rtp(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4_rtp(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4_rtz(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4_rtz(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4_rtz(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4_sat(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4_sat(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4_sat(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4_sat_rte(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4_sat_rte(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4_sat_rte(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4_sat_rtn(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4_sat_rtn(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4_sat_rtn(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4_sat_rtp(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4_sat_rtp(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4_sat_rtp(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4_sat_rtz(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4_sat_rtz(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ConvertFToU_Rulong4_sat_rtz(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8_rte(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8_rte(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8_rte(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8_rtn(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8_rtn(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8_rtn(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8_rtp(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8_rtp(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8_rtp(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8_rtz(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8_rtz(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8_rtz(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8_sat(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8_sat(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8_sat(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8_sat_rte(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8_sat_rte(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8_sat_rte(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8_sat_rtn(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8_sat_rtn(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8_sat_rtn(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8_sat_rtp(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8_sat_rtp(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8_sat_rtp(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8_sat_rtz(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8_sat_rtz(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ConvertFToU_Rulong8_sat_rtz(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong_rte(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong_rte(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong_rte(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong_rtn(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong_rtn(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong_rtn(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong_rtp(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong_rtp(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong_rtp(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong_rtz(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong_rtz(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong_rtz(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong_sat(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong_sat(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong_sat(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong_sat_rte(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong_sat_rte(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong_sat_rte(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong_sat_rtn(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong_sat_rtn(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong_sat_rtn(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong_sat_rtp(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong_sat_rtp(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong_sat_rtp(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong_sat_rtz(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong_sat_rtz(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ConvertFToU_Rulong_sat_rtz(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16_rte(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16_rte(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16_rte(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16_rtn(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16_rtn(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16_rtn(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16_rtp(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16_rtp(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16_rtp(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16_rtz(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16_rtz(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16_rtz(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16_sat(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16_sat(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16_sat(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16_sat_rte(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16_sat_rte(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16_sat_rte(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16_sat_rtn(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16_sat_rtn(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16_sat_rtn(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16_sat_rtp(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16_sat_rtp(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16_sat_rtp(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16_sat_rtz(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16_sat_rtz(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ConvertFToU_Rushort16_sat_rtz(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2_rte(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2_rte(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2_rte(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2_rtn(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2_rtn(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2_rtn(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2_rtp(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2_rtp(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2_rtp(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2_rtz(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2_rtz(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2_rtz(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2_sat(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2_sat(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2_sat(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2_sat_rte(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2_sat_rte(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2_sat_rte(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2_sat_rtn(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2_sat_rtn(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2_sat_rtn(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2_sat_rtp(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2_sat_rtp(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2_sat_rtp(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2_sat_rtz(__clc_vec2_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2_sat_rtz(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ConvertFToU_Rushort2_sat_rtz(__clc_vec2_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3_rte(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3_rte(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3_rte(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3_rtn(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3_rtn(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3_rtn(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3_rtp(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3_rtp(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3_rtp(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3_rtz(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3_rtz(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3_rtz(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3_sat(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3_sat(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3_sat(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3_sat_rte(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3_sat_rte(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3_sat_rte(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3_sat_rtn(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3_sat_rtn(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3_sat_rtn(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3_sat_rtp(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3_sat_rtp(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3_sat_rtp(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3_sat_rtz(__clc_vec3_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3_sat_rtz(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ConvertFToU_Rushort3_sat_rtz(__clc_vec3_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4_rte(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4_rte(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4_rte(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4_rtn(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4_rtn(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4_rtn(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4_rtp(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4_rtp(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4_rtp(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4_rtz(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4_rtz(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4_rtz(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4_sat(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4_sat(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4_sat(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4_sat_rte(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4_sat_rte(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4_sat_rte(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4_sat_rtn(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4_sat_rtn(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4_sat_rtn(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4_sat_rtp(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4_sat_rtp(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4_sat_rtp(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4_sat_rtz(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4_sat_rtz(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ConvertFToU_Rushort4_sat_rtz(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8_rte(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8_rte(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8_rte(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8_rtn(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8_rtn(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8_rtn(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8_rtp(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8_rtp(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8_rtp(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8_rtz(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8_rtz(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8_rtz(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8_sat(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8_sat(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8_sat(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8_sat_rte(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8_sat_rte(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8_sat_rte(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8_sat_rtn(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8_sat_rtn(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8_sat_rtn(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8_sat_rtp(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8_sat_rtp(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8_sat_rtp(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8_sat_rtz(__clc_vec8_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8_sat_rtz(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ConvertFToU_Rushort8_sat_rtz(__clc_vec8_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort_rte(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort_rte(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort_rte(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort_rtn(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort_rtn(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort_rtn(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort_rtp(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort_rtp(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort_rtp(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort_rtz(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort_rtz(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort_rtz(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort_sat(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort_sat(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort_sat(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort_sat_rte(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort_sat_rte(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort_sat_rte(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort_sat_rtn(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort_sat_rtn(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort_sat_rtn(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort_sat_rtp(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort_sat_rtp(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort_sat_rtp(__clc_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort_sat_rtz(__clc_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort_sat_rtz(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ConvertFToU_Rushort_sat_rtz(__clc_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertSToF_Rdouble(__clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertSToF_Rdouble(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertSToF_Rdouble(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertSToF_Rdouble(__clc_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertSToF_Rdouble16(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertSToF_Rdouble16(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertSToF_Rdouble16(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertSToF_Rdouble16(__clc_vec16_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertSToF_Rdouble16_rte(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertSToF_Rdouble16_rte(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertSToF_Rdouble16_rte(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertSToF_Rdouble16_rte(__clc_vec16_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertSToF_Rdouble16_rtn(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertSToF_Rdouble16_rtn(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertSToF_Rdouble16_rtn(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertSToF_Rdouble16_rtn(__clc_vec16_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertSToF_Rdouble16_rtp(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertSToF_Rdouble16_rtp(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertSToF_Rdouble16_rtp(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertSToF_Rdouble16_rtp(__clc_vec16_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertSToF_Rdouble16_rtz(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertSToF_Rdouble16_rtz(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertSToF_Rdouble16_rtz(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertSToF_Rdouble16_rtz(__clc_vec16_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertSToF_Rdouble2(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertSToF_Rdouble2(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertSToF_Rdouble2(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertSToF_Rdouble2(__clc_vec2_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertSToF_Rdouble2_rte(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertSToF_Rdouble2_rte(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertSToF_Rdouble2_rte(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertSToF_Rdouble2_rte(__clc_vec2_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertSToF_Rdouble2_rtn(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertSToF_Rdouble2_rtn(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertSToF_Rdouble2_rtn(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertSToF_Rdouble2_rtn(__clc_vec2_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertSToF_Rdouble2_rtp(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertSToF_Rdouble2_rtp(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertSToF_Rdouble2_rtp(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertSToF_Rdouble2_rtp(__clc_vec2_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertSToF_Rdouble2_rtz(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertSToF_Rdouble2_rtz(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertSToF_Rdouble2_rtz(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertSToF_Rdouble2_rtz(__clc_vec2_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertSToF_Rdouble3(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertSToF_Rdouble3(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertSToF_Rdouble3(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertSToF_Rdouble3(__clc_vec3_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertSToF_Rdouble3_rte(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertSToF_Rdouble3_rte(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertSToF_Rdouble3_rte(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertSToF_Rdouble3_rte(__clc_vec3_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertSToF_Rdouble3_rtn(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertSToF_Rdouble3_rtn(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertSToF_Rdouble3_rtn(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertSToF_Rdouble3_rtn(__clc_vec3_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertSToF_Rdouble3_rtp(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertSToF_Rdouble3_rtp(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertSToF_Rdouble3_rtp(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertSToF_Rdouble3_rtp(__clc_vec3_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertSToF_Rdouble3_rtz(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertSToF_Rdouble3_rtz(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertSToF_Rdouble3_rtz(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertSToF_Rdouble3_rtz(__clc_vec3_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertSToF_Rdouble4(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertSToF_Rdouble4(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertSToF_Rdouble4(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertSToF_Rdouble4(__clc_vec4_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertSToF_Rdouble4_rte(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertSToF_Rdouble4_rte(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertSToF_Rdouble4_rte(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertSToF_Rdouble4_rte(__clc_vec4_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertSToF_Rdouble4_rtn(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertSToF_Rdouble4_rtn(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertSToF_Rdouble4_rtn(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertSToF_Rdouble4_rtn(__clc_vec4_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertSToF_Rdouble4_rtp(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertSToF_Rdouble4_rtp(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertSToF_Rdouble4_rtp(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertSToF_Rdouble4_rtp(__clc_vec4_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertSToF_Rdouble4_rtz(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertSToF_Rdouble4_rtz(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertSToF_Rdouble4_rtz(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertSToF_Rdouble4_rtz(__clc_vec4_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertSToF_Rdouble8(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertSToF_Rdouble8(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertSToF_Rdouble8(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertSToF_Rdouble8(__clc_vec8_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertSToF_Rdouble8_rte(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertSToF_Rdouble8_rte(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertSToF_Rdouble8_rte(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertSToF_Rdouble8_rte(__clc_vec8_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertSToF_Rdouble8_rtn(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertSToF_Rdouble8_rtn(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertSToF_Rdouble8_rtn(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertSToF_Rdouble8_rtn(__clc_vec8_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertSToF_Rdouble8_rtp(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertSToF_Rdouble8_rtp(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertSToF_Rdouble8_rtp(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertSToF_Rdouble8_rtp(__clc_vec8_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertSToF_Rdouble8_rtz(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertSToF_Rdouble8_rtz(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertSToF_Rdouble8_rtz(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertSToF_Rdouble8_rtz(__clc_vec8_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertSToF_Rdouble_rte(__clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertSToF_Rdouble_rte(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertSToF_Rdouble_rte(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertSToF_Rdouble_rte(__clc_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertSToF_Rdouble_rtn(__clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertSToF_Rdouble_rtn(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertSToF_Rdouble_rtn(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertSToF_Rdouble_rtn(__clc_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertSToF_Rdouble_rtp(__clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertSToF_Rdouble_rtp(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertSToF_Rdouble_rtp(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertSToF_Rdouble_rtp(__clc_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertSToF_Rdouble_rtz(__clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertSToF_Rdouble_rtz(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertSToF_Rdouble_rtz(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertSToF_Rdouble_rtz(__clc_int64_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertSToF_Rfloat(__clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertSToF_Rfloat(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertSToF_Rfloat(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertSToF_Rfloat(__clc_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertSToF_Rfloat16(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertSToF_Rfloat16(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertSToF_Rfloat16(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertSToF_Rfloat16(__clc_vec16_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertSToF_Rfloat16_rte(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertSToF_Rfloat16_rte(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertSToF_Rfloat16_rte(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertSToF_Rfloat16_rte(__clc_vec16_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertSToF_Rfloat16_rtn(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertSToF_Rfloat16_rtn(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertSToF_Rfloat16_rtn(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertSToF_Rfloat16_rtn(__clc_vec16_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertSToF_Rfloat16_rtp(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertSToF_Rfloat16_rtp(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertSToF_Rfloat16_rtp(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertSToF_Rfloat16_rtp(__clc_vec16_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertSToF_Rfloat16_rtz(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertSToF_Rfloat16_rtz(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertSToF_Rfloat16_rtz(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertSToF_Rfloat16_rtz(__clc_vec16_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertSToF_Rfloat2(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertSToF_Rfloat2(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertSToF_Rfloat2(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertSToF_Rfloat2(__clc_vec2_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertSToF_Rfloat2_rte(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertSToF_Rfloat2_rte(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertSToF_Rfloat2_rte(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertSToF_Rfloat2_rte(__clc_vec2_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertSToF_Rfloat2_rtn(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertSToF_Rfloat2_rtn(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertSToF_Rfloat2_rtn(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertSToF_Rfloat2_rtn(__clc_vec2_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertSToF_Rfloat2_rtp(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertSToF_Rfloat2_rtp(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertSToF_Rfloat2_rtp(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertSToF_Rfloat2_rtp(__clc_vec2_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertSToF_Rfloat2_rtz(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertSToF_Rfloat2_rtz(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertSToF_Rfloat2_rtz(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertSToF_Rfloat2_rtz(__clc_vec2_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertSToF_Rfloat3(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertSToF_Rfloat3(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertSToF_Rfloat3(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertSToF_Rfloat3(__clc_vec3_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertSToF_Rfloat3_rte(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertSToF_Rfloat3_rte(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertSToF_Rfloat3_rte(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertSToF_Rfloat3_rte(__clc_vec3_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertSToF_Rfloat3_rtn(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertSToF_Rfloat3_rtn(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertSToF_Rfloat3_rtn(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertSToF_Rfloat3_rtn(__clc_vec3_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertSToF_Rfloat3_rtp(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertSToF_Rfloat3_rtp(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertSToF_Rfloat3_rtp(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertSToF_Rfloat3_rtp(__clc_vec3_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertSToF_Rfloat3_rtz(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertSToF_Rfloat3_rtz(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertSToF_Rfloat3_rtz(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertSToF_Rfloat3_rtz(__clc_vec3_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertSToF_Rfloat4(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertSToF_Rfloat4(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertSToF_Rfloat4(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertSToF_Rfloat4(__clc_vec4_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertSToF_Rfloat4_rte(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertSToF_Rfloat4_rte(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertSToF_Rfloat4_rte(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertSToF_Rfloat4_rte(__clc_vec4_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertSToF_Rfloat4_rtn(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertSToF_Rfloat4_rtn(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertSToF_Rfloat4_rtn(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertSToF_Rfloat4_rtn(__clc_vec4_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertSToF_Rfloat4_rtp(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertSToF_Rfloat4_rtp(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertSToF_Rfloat4_rtp(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertSToF_Rfloat4_rtp(__clc_vec4_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertSToF_Rfloat4_rtz(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertSToF_Rfloat4_rtz(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertSToF_Rfloat4_rtz(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertSToF_Rfloat4_rtz(__clc_vec4_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertSToF_Rfloat8(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertSToF_Rfloat8(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertSToF_Rfloat8(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertSToF_Rfloat8(__clc_vec8_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertSToF_Rfloat8_rte(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertSToF_Rfloat8_rte(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertSToF_Rfloat8_rte(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertSToF_Rfloat8_rte(__clc_vec8_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertSToF_Rfloat8_rtn(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertSToF_Rfloat8_rtn(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertSToF_Rfloat8_rtn(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertSToF_Rfloat8_rtn(__clc_vec8_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertSToF_Rfloat8_rtp(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertSToF_Rfloat8_rtp(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertSToF_Rfloat8_rtp(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertSToF_Rfloat8_rtp(__clc_vec8_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertSToF_Rfloat8_rtz(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertSToF_Rfloat8_rtz(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertSToF_Rfloat8_rtz(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertSToF_Rfloat8_rtz(__clc_vec8_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertSToF_Rfloat_rte(__clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertSToF_Rfloat_rte(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertSToF_Rfloat_rte(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertSToF_Rfloat_rte(__clc_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertSToF_Rfloat_rtn(__clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertSToF_Rfloat_rtn(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertSToF_Rfloat_rtn(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertSToF_Rfloat_rtn(__clc_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertSToF_Rfloat_rtp(__clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertSToF_Rfloat_rtp(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertSToF_Rfloat_rtp(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertSToF_Rfloat_rtp(__clc_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertSToF_Rfloat_rtz(__clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertSToF_Rfloat_rtz(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertSToF_Rfloat_rtz(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertSToF_Rfloat_rtz(__clc_int64_t);

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ConvertSToF_Rhalf(__clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertSToF_Rhalf(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertSToF_Rhalf(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertSToF_Rhalf(__clc_int64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertSToF_Rhalf16(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertSToF_Rhalf16(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertSToF_Rhalf16(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertSToF_Rhalf16(__clc_vec16_int64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertSToF_Rhalf16_rte(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertSToF_Rhalf16_rte(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertSToF_Rhalf16_rte(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertSToF_Rhalf16_rte(__clc_vec16_int64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertSToF_Rhalf16_rtn(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertSToF_Rhalf16_rtn(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertSToF_Rhalf16_rtn(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertSToF_Rhalf16_rtn(__clc_vec16_int64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertSToF_Rhalf16_rtp(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertSToF_Rhalf16_rtp(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertSToF_Rhalf16_rtp(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertSToF_Rhalf16_rtp(__clc_vec16_int64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertSToF_Rhalf16_rtz(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertSToF_Rhalf16_rtz(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertSToF_Rhalf16_rtz(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertSToF_Rhalf16_rtz(__clc_vec16_int64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertSToF_Rhalf2(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertSToF_Rhalf2(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertSToF_Rhalf2(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertSToF_Rhalf2(__clc_vec2_int64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertSToF_Rhalf2_rte(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertSToF_Rhalf2_rte(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertSToF_Rhalf2_rte(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertSToF_Rhalf2_rte(__clc_vec2_int64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertSToF_Rhalf2_rtn(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertSToF_Rhalf2_rtn(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertSToF_Rhalf2_rtn(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertSToF_Rhalf2_rtn(__clc_vec2_int64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertSToF_Rhalf2_rtp(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertSToF_Rhalf2_rtp(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertSToF_Rhalf2_rtp(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertSToF_Rhalf2_rtp(__clc_vec2_int64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertSToF_Rhalf2_rtz(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertSToF_Rhalf2_rtz(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertSToF_Rhalf2_rtz(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertSToF_Rhalf2_rtz(__clc_vec2_int64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertSToF_Rhalf3(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertSToF_Rhalf3(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertSToF_Rhalf3(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertSToF_Rhalf3(__clc_vec3_int64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertSToF_Rhalf3_rte(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertSToF_Rhalf3_rte(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertSToF_Rhalf3_rte(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertSToF_Rhalf3_rte(__clc_vec3_int64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertSToF_Rhalf3_rtn(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertSToF_Rhalf3_rtn(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertSToF_Rhalf3_rtn(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertSToF_Rhalf3_rtn(__clc_vec3_int64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertSToF_Rhalf3_rtp(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertSToF_Rhalf3_rtp(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertSToF_Rhalf3_rtp(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertSToF_Rhalf3_rtp(__clc_vec3_int64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertSToF_Rhalf3_rtz(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertSToF_Rhalf3_rtz(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertSToF_Rhalf3_rtz(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertSToF_Rhalf3_rtz(__clc_vec3_int64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertSToF_Rhalf4(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertSToF_Rhalf4(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertSToF_Rhalf4(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertSToF_Rhalf4(__clc_vec4_int64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertSToF_Rhalf4_rte(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertSToF_Rhalf4_rte(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertSToF_Rhalf4_rte(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertSToF_Rhalf4_rte(__clc_vec4_int64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertSToF_Rhalf4_rtn(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertSToF_Rhalf4_rtn(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertSToF_Rhalf4_rtn(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertSToF_Rhalf4_rtn(__clc_vec4_int64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertSToF_Rhalf4_rtp(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertSToF_Rhalf4_rtp(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertSToF_Rhalf4_rtp(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertSToF_Rhalf4_rtp(__clc_vec4_int64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertSToF_Rhalf4_rtz(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertSToF_Rhalf4_rtz(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertSToF_Rhalf4_rtz(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertSToF_Rhalf4_rtz(__clc_vec4_int64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertSToF_Rhalf8(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertSToF_Rhalf8(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertSToF_Rhalf8(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertSToF_Rhalf8(__clc_vec8_int64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertSToF_Rhalf8_rte(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertSToF_Rhalf8_rte(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertSToF_Rhalf8_rte(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertSToF_Rhalf8_rte(__clc_vec8_int64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertSToF_Rhalf8_rtn(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertSToF_Rhalf8_rtn(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertSToF_Rhalf8_rtn(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertSToF_Rhalf8_rtn(__clc_vec8_int64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertSToF_Rhalf8_rtp(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertSToF_Rhalf8_rtp(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertSToF_Rhalf8_rtp(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertSToF_Rhalf8_rtp(__clc_vec8_int64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertSToF_Rhalf8_rtz(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertSToF_Rhalf8_rtz(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertSToF_Rhalf8_rtz(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertSToF_Rhalf8_rtz(__clc_vec8_int64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertSToF_Rhalf_rte(__clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertSToF_Rhalf_rte(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertSToF_Rhalf_rte(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertSToF_Rhalf_rte(__clc_int64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertSToF_Rhalf_rtn(__clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertSToF_Rhalf_rtn(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertSToF_Rhalf_rtn(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertSToF_Rhalf_rtn(__clc_int64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertSToF_Rhalf_rtp(__clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertSToF_Rhalf_rtp(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertSToF_Rhalf_rtp(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertSToF_Rhalf_rtp(__clc_int64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertSToF_Rhalf_rtz(__clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertSToF_Rhalf_rtz(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertSToF_Rhalf_rtz(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertSToF_Rhalf_rtz(__clc_int64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertUToF_Rdouble(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertUToF_Rdouble(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertUToF_Rdouble(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertUToF_Rdouble(__clc_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertUToF_Rdouble16(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertUToF_Rdouble16(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertUToF_Rdouble16(__clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertUToF_Rdouble16(__clc_vec16_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertUToF_Rdouble16_rte(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertUToF_Rdouble16_rte(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertUToF_Rdouble16_rte(__clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertUToF_Rdouble16_rte(__clc_vec16_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertUToF_Rdouble16_rtn(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertUToF_Rdouble16_rtn(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertUToF_Rdouble16_rtn(__clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertUToF_Rdouble16_rtn(__clc_vec16_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertUToF_Rdouble16_rtp(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertUToF_Rdouble16_rtp(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertUToF_Rdouble16_rtp(__clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertUToF_Rdouble16_rtp(__clc_vec16_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertUToF_Rdouble16_rtz(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertUToF_Rdouble16_rtz(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertUToF_Rdouble16_rtz(__clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ConvertUToF_Rdouble16_rtz(__clc_vec16_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertUToF_Rdouble2(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertUToF_Rdouble2(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertUToF_Rdouble2(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertUToF_Rdouble2(__clc_vec2_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertUToF_Rdouble2_rte(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertUToF_Rdouble2_rte(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertUToF_Rdouble2_rte(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertUToF_Rdouble2_rte(__clc_vec2_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertUToF_Rdouble2_rtn(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertUToF_Rdouble2_rtn(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertUToF_Rdouble2_rtn(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertUToF_Rdouble2_rtn(__clc_vec2_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertUToF_Rdouble2_rtp(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertUToF_Rdouble2_rtp(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertUToF_Rdouble2_rtp(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertUToF_Rdouble2_rtp(__clc_vec2_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertUToF_Rdouble2_rtz(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertUToF_Rdouble2_rtz(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertUToF_Rdouble2_rtz(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ConvertUToF_Rdouble2_rtz(__clc_vec2_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertUToF_Rdouble3(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertUToF_Rdouble3(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertUToF_Rdouble3(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertUToF_Rdouble3(__clc_vec3_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertUToF_Rdouble3_rte(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertUToF_Rdouble3_rte(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertUToF_Rdouble3_rte(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertUToF_Rdouble3_rte(__clc_vec3_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertUToF_Rdouble3_rtn(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertUToF_Rdouble3_rtn(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertUToF_Rdouble3_rtn(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertUToF_Rdouble3_rtn(__clc_vec3_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertUToF_Rdouble3_rtp(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertUToF_Rdouble3_rtp(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertUToF_Rdouble3_rtp(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertUToF_Rdouble3_rtp(__clc_vec3_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertUToF_Rdouble3_rtz(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertUToF_Rdouble3_rtz(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertUToF_Rdouble3_rtz(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ConvertUToF_Rdouble3_rtz(__clc_vec3_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertUToF_Rdouble4(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertUToF_Rdouble4(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertUToF_Rdouble4(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertUToF_Rdouble4(__clc_vec4_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertUToF_Rdouble4_rte(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertUToF_Rdouble4_rte(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertUToF_Rdouble4_rte(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertUToF_Rdouble4_rte(__clc_vec4_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertUToF_Rdouble4_rtn(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertUToF_Rdouble4_rtn(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertUToF_Rdouble4_rtn(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertUToF_Rdouble4_rtn(__clc_vec4_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertUToF_Rdouble4_rtp(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertUToF_Rdouble4_rtp(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertUToF_Rdouble4_rtp(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertUToF_Rdouble4_rtp(__clc_vec4_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertUToF_Rdouble4_rtz(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertUToF_Rdouble4_rtz(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertUToF_Rdouble4_rtz(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ConvertUToF_Rdouble4_rtz(__clc_vec4_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertUToF_Rdouble8(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertUToF_Rdouble8(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertUToF_Rdouble8(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertUToF_Rdouble8(__clc_vec8_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertUToF_Rdouble8_rte(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertUToF_Rdouble8_rte(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertUToF_Rdouble8_rte(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertUToF_Rdouble8_rte(__clc_vec8_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertUToF_Rdouble8_rtn(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertUToF_Rdouble8_rtn(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertUToF_Rdouble8_rtn(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertUToF_Rdouble8_rtn(__clc_vec8_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertUToF_Rdouble8_rtp(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertUToF_Rdouble8_rtp(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertUToF_Rdouble8_rtp(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertUToF_Rdouble8_rtp(__clc_vec8_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertUToF_Rdouble8_rtz(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertUToF_Rdouble8_rtz(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertUToF_Rdouble8_rtz(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ConvertUToF_Rdouble8_rtz(__clc_vec8_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertUToF_Rdouble_rte(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertUToF_Rdouble_rte(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertUToF_Rdouble_rte(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertUToF_Rdouble_rte(__clc_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertUToF_Rdouble_rtn(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertUToF_Rdouble_rtn(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertUToF_Rdouble_rtn(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertUToF_Rdouble_rtn(__clc_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertUToF_Rdouble_rtp(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertUToF_Rdouble_rtp(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertUToF_Rdouble_rtp(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertUToF_Rdouble_rtp(__clc_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertUToF_Rdouble_rtz(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertUToF_Rdouble_rtz(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertUToF_Rdouble_rtz(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ConvertUToF_Rdouble_rtz(__clc_uint64_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertUToF_Rfloat(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertUToF_Rfloat(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertUToF_Rfloat(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertUToF_Rfloat(__clc_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertUToF_Rfloat16(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertUToF_Rfloat16(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertUToF_Rfloat16(__clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertUToF_Rfloat16(__clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertUToF_Rfloat16_rte(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertUToF_Rfloat16_rte(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertUToF_Rfloat16_rte(__clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertUToF_Rfloat16_rte(__clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertUToF_Rfloat16_rtn(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertUToF_Rfloat16_rtn(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertUToF_Rfloat16_rtn(__clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertUToF_Rfloat16_rtn(__clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertUToF_Rfloat16_rtp(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertUToF_Rfloat16_rtp(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertUToF_Rfloat16_rtp(__clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertUToF_Rfloat16_rtp(__clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertUToF_Rfloat16_rtz(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertUToF_Rfloat16_rtz(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertUToF_Rfloat16_rtz(__clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ConvertUToF_Rfloat16_rtz(__clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertUToF_Rfloat2(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertUToF_Rfloat2(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertUToF_Rfloat2(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertUToF_Rfloat2(__clc_vec2_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertUToF_Rfloat2_rte(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertUToF_Rfloat2_rte(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertUToF_Rfloat2_rte(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertUToF_Rfloat2_rte(__clc_vec2_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertUToF_Rfloat2_rtn(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertUToF_Rfloat2_rtn(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertUToF_Rfloat2_rtn(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertUToF_Rfloat2_rtn(__clc_vec2_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertUToF_Rfloat2_rtp(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertUToF_Rfloat2_rtp(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertUToF_Rfloat2_rtp(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertUToF_Rfloat2_rtp(__clc_vec2_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertUToF_Rfloat2_rtz(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertUToF_Rfloat2_rtz(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertUToF_Rfloat2_rtz(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ConvertUToF_Rfloat2_rtz(__clc_vec2_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertUToF_Rfloat3(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertUToF_Rfloat3(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertUToF_Rfloat3(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertUToF_Rfloat3(__clc_vec3_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertUToF_Rfloat3_rte(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertUToF_Rfloat3_rte(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertUToF_Rfloat3_rte(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertUToF_Rfloat3_rte(__clc_vec3_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertUToF_Rfloat3_rtn(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertUToF_Rfloat3_rtn(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertUToF_Rfloat3_rtn(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertUToF_Rfloat3_rtn(__clc_vec3_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertUToF_Rfloat3_rtp(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertUToF_Rfloat3_rtp(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertUToF_Rfloat3_rtp(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertUToF_Rfloat3_rtp(__clc_vec3_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertUToF_Rfloat3_rtz(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertUToF_Rfloat3_rtz(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertUToF_Rfloat3_rtz(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ConvertUToF_Rfloat3_rtz(__clc_vec3_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertUToF_Rfloat4(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertUToF_Rfloat4(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertUToF_Rfloat4(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertUToF_Rfloat4(__clc_vec4_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertUToF_Rfloat4_rte(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertUToF_Rfloat4_rte(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertUToF_Rfloat4_rte(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertUToF_Rfloat4_rte(__clc_vec4_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertUToF_Rfloat4_rtn(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertUToF_Rfloat4_rtn(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertUToF_Rfloat4_rtn(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertUToF_Rfloat4_rtn(__clc_vec4_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertUToF_Rfloat4_rtp(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertUToF_Rfloat4_rtp(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertUToF_Rfloat4_rtp(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertUToF_Rfloat4_rtp(__clc_vec4_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertUToF_Rfloat4_rtz(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertUToF_Rfloat4_rtz(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertUToF_Rfloat4_rtz(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ConvertUToF_Rfloat4_rtz(__clc_vec4_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertUToF_Rfloat8(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertUToF_Rfloat8(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertUToF_Rfloat8(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertUToF_Rfloat8(__clc_vec8_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertUToF_Rfloat8_rte(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertUToF_Rfloat8_rte(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertUToF_Rfloat8_rte(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertUToF_Rfloat8_rte(__clc_vec8_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertUToF_Rfloat8_rtn(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertUToF_Rfloat8_rtn(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertUToF_Rfloat8_rtn(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertUToF_Rfloat8_rtn(__clc_vec8_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertUToF_Rfloat8_rtp(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertUToF_Rfloat8_rtp(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertUToF_Rfloat8_rtp(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertUToF_Rfloat8_rtp(__clc_vec8_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertUToF_Rfloat8_rtz(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertUToF_Rfloat8_rtz(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertUToF_Rfloat8_rtz(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ConvertUToF_Rfloat8_rtz(__clc_vec8_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertUToF_Rfloat_rte(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertUToF_Rfloat_rte(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertUToF_Rfloat_rte(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertUToF_Rfloat_rte(__clc_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertUToF_Rfloat_rtn(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertUToF_Rfloat_rtn(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertUToF_Rfloat_rtn(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertUToF_Rfloat_rtn(__clc_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertUToF_Rfloat_rtp(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertUToF_Rfloat_rtp(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertUToF_Rfloat_rtp(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertUToF_Rfloat_rtp(__clc_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertUToF_Rfloat_rtz(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertUToF_Rfloat_rtz(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertUToF_Rfloat_rtz(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ConvertUToF_Rfloat_rtz(__clc_uint64_t);

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertUToF_Rhalf(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertUToF_Rhalf(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertUToF_Rhalf(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertUToF_Rhalf(__clc_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertUToF_Rhalf16(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertUToF_Rhalf16(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertUToF_Rhalf16(__clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertUToF_Rhalf16(__clc_vec16_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertUToF_Rhalf16_rte(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertUToF_Rhalf16_rte(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertUToF_Rhalf16_rte(__clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertUToF_Rhalf16_rte(__clc_vec16_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertUToF_Rhalf16_rtn(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertUToF_Rhalf16_rtn(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertUToF_Rhalf16_rtn(__clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertUToF_Rhalf16_rtn(__clc_vec16_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertUToF_Rhalf16_rtp(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertUToF_Rhalf16_rtp(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertUToF_Rhalf16_rtp(__clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertUToF_Rhalf16_rtp(__clc_vec16_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertUToF_Rhalf16_rtz(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertUToF_Rhalf16_rtz(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertUToF_Rhalf16_rtz(__clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ConvertUToF_Rhalf16_rtz(__clc_vec16_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertUToF_Rhalf2(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertUToF_Rhalf2(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertUToF_Rhalf2(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertUToF_Rhalf2(__clc_vec2_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertUToF_Rhalf2_rte(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertUToF_Rhalf2_rte(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertUToF_Rhalf2_rte(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertUToF_Rhalf2_rte(__clc_vec2_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertUToF_Rhalf2_rtn(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertUToF_Rhalf2_rtn(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertUToF_Rhalf2_rtn(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertUToF_Rhalf2_rtn(__clc_vec2_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertUToF_Rhalf2_rtp(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertUToF_Rhalf2_rtp(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertUToF_Rhalf2_rtp(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertUToF_Rhalf2_rtp(__clc_vec2_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertUToF_Rhalf2_rtz(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertUToF_Rhalf2_rtz(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertUToF_Rhalf2_rtz(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ConvertUToF_Rhalf2_rtz(__clc_vec2_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertUToF_Rhalf3(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertUToF_Rhalf3(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertUToF_Rhalf3(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertUToF_Rhalf3(__clc_vec3_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertUToF_Rhalf3_rte(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertUToF_Rhalf3_rte(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertUToF_Rhalf3_rte(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertUToF_Rhalf3_rte(__clc_vec3_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertUToF_Rhalf3_rtn(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertUToF_Rhalf3_rtn(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertUToF_Rhalf3_rtn(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertUToF_Rhalf3_rtn(__clc_vec3_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertUToF_Rhalf3_rtp(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertUToF_Rhalf3_rtp(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertUToF_Rhalf3_rtp(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertUToF_Rhalf3_rtp(__clc_vec3_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertUToF_Rhalf3_rtz(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertUToF_Rhalf3_rtz(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertUToF_Rhalf3_rtz(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ConvertUToF_Rhalf3_rtz(__clc_vec3_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertUToF_Rhalf4(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertUToF_Rhalf4(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertUToF_Rhalf4(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertUToF_Rhalf4(__clc_vec4_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertUToF_Rhalf4_rte(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertUToF_Rhalf4_rte(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertUToF_Rhalf4_rte(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertUToF_Rhalf4_rte(__clc_vec4_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertUToF_Rhalf4_rtn(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertUToF_Rhalf4_rtn(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertUToF_Rhalf4_rtn(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertUToF_Rhalf4_rtn(__clc_vec4_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertUToF_Rhalf4_rtp(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertUToF_Rhalf4_rtp(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertUToF_Rhalf4_rtp(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertUToF_Rhalf4_rtp(__clc_vec4_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertUToF_Rhalf4_rtz(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertUToF_Rhalf4_rtz(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertUToF_Rhalf4_rtz(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ConvertUToF_Rhalf4_rtz(__clc_vec4_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertUToF_Rhalf8(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertUToF_Rhalf8(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertUToF_Rhalf8(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertUToF_Rhalf8(__clc_vec8_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertUToF_Rhalf8_rte(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertUToF_Rhalf8_rte(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertUToF_Rhalf8_rte(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertUToF_Rhalf8_rte(__clc_vec8_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertUToF_Rhalf8_rtn(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertUToF_Rhalf8_rtn(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertUToF_Rhalf8_rtn(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertUToF_Rhalf8_rtn(__clc_vec8_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertUToF_Rhalf8_rtp(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertUToF_Rhalf8_rtp(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertUToF_Rhalf8_rtp(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertUToF_Rhalf8_rtp(__clc_vec8_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertUToF_Rhalf8_rtz(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertUToF_Rhalf8_rtz(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertUToF_Rhalf8_rtz(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ConvertUToF_Rhalf8_rtz(__clc_vec8_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertUToF_Rhalf_rte(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertUToF_Rhalf_rte(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertUToF_Rhalf_rte(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertUToF_Rhalf_rte(__clc_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertUToF_Rhalf_rtn(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertUToF_Rhalf_rtn(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertUToF_Rhalf_rtn(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertUToF_Rhalf_rtn(__clc_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertUToF_Rhalf_rtp(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertUToF_Rhalf_rtp(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertUToF_Rhalf_rtp(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertUToF_Rhalf_rtp(__clc_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertUToF_Rhalf_rtz(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertUToF_Rhalf_rtz(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertUToF_Rhalf_rtz(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ConvertUToF_Rhalf_rtz(__clc_uint64_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_FConvert_Rdouble(__clc_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_FConvert_Rdouble(__clc_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_FConvert_Rdouble16(__clc_vec16_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_FConvert_Rdouble16(__clc_vec16_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_FConvert_Rdouble16_rte(__clc_vec16_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_FConvert_Rdouble16_rte(__clc_vec16_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_FConvert_Rdouble16_rtn(__clc_vec16_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_FConvert_Rdouble16_rtn(__clc_vec16_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_FConvert_Rdouble16_rtp(__clc_vec16_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_FConvert_Rdouble16_rtp(__clc_vec16_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_FConvert_Rdouble16_rtz(__clc_vec16_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_FConvert_Rdouble16_rtz(__clc_vec16_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_FConvert_Rdouble2(__clc_vec2_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_FConvert_Rdouble2(__clc_vec2_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_FConvert_Rdouble2_rte(__clc_vec2_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_FConvert_Rdouble2_rte(__clc_vec2_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_FConvert_Rdouble2_rtn(__clc_vec2_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_FConvert_Rdouble2_rtn(__clc_vec2_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_FConvert_Rdouble2_rtp(__clc_vec2_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_FConvert_Rdouble2_rtp(__clc_vec2_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_FConvert_Rdouble2_rtz(__clc_vec2_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_FConvert_Rdouble2_rtz(__clc_vec2_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_FConvert_Rdouble3(__clc_vec3_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_FConvert_Rdouble3(__clc_vec3_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_FConvert_Rdouble3_rte(__clc_vec3_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_FConvert_Rdouble3_rte(__clc_vec3_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_FConvert_Rdouble3_rtn(__clc_vec3_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_FConvert_Rdouble3_rtn(__clc_vec3_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_FConvert_Rdouble3_rtp(__clc_vec3_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_FConvert_Rdouble3_rtp(__clc_vec3_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_FConvert_Rdouble3_rtz(__clc_vec3_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_FConvert_Rdouble3_rtz(__clc_vec3_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_FConvert_Rdouble4(__clc_vec4_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_FConvert_Rdouble4(__clc_vec4_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_FConvert_Rdouble4_rte(__clc_vec4_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_FConvert_Rdouble4_rte(__clc_vec4_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_FConvert_Rdouble4_rtn(__clc_vec4_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_FConvert_Rdouble4_rtn(__clc_vec4_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_FConvert_Rdouble4_rtp(__clc_vec4_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_FConvert_Rdouble4_rtp(__clc_vec4_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_FConvert_Rdouble4_rtz(__clc_vec4_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_FConvert_Rdouble4_rtz(__clc_vec4_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_FConvert_Rdouble8(__clc_vec8_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_FConvert_Rdouble8(__clc_vec8_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_FConvert_Rdouble8_rte(__clc_vec8_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_FConvert_Rdouble8_rte(__clc_vec8_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_FConvert_Rdouble8_rtn(__clc_vec8_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_FConvert_Rdouble8_rtn(__clc_vec8_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_FConvert_Rdouble8_rtp(__clc_vec8_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_FConvert_Rdouble8_rtp(__clc_vec8_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_FConvert_Rdouble8_rtz(__clc_vec8_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_FConvert_Rdouble8_rtz(__clc_vec8_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_FConvert_Rdouble_rte(__clc_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_FConvert_Rdouble_rte(__clc_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_FConvert_Rdouble_rtn(__clc_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_FConvert_Rdouble_rtn(__clc_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_FConvert_Rdouble_rtp(__clc_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_FConvert_Rdouble_rtp(__clc_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_FConvert_Rdouble_rtz(__clc_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_FConvert_Rdouble_rtz(__clc_fp16_t);
#endif
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_FConvert_Rfloat(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_FConvert_Rfloat(__clc_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_FConvert_Rfloat16(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_FConvert_Rfloat16(__clc_vec16_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_FConvert_Rfloat16_rte(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_FConvert_Rfloat16_rte(__clc_vec16_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_FConvert_Rfloat16_rtn(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_FConvert_Rfloat16_rtn(__clc_vec16_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_FConvert_Rfloat16_rtp(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_FConvert_Rfloat16_rtp(__clc_vec16_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_FConvert_Rfloat16_rtz(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_FConvert_Rfloat16_rtz(__clc_vec16_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_FConvert_Rfloat2(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_FConvert_Rfloat2(__clc_vec2_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_FConvert_Rfloat2_rte(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_FConvert_Rfloat2_rte(__clc_vec2_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_FConvert_Rfloat2_rtn(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_FConvert_Rfloat2_rtn(__clc_vec2_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_FConvert_Rfloat2_rtp(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_FConvert_Rfloat2_rtp(__clc_vec2_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_FConvert_Rfloat2_rtz(__clc_vec2_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_FConvert_Rfloat2_rtz(__clc_vec2_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_FConvert_Rfloat3(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_FConvert_Rfloat3(__clc_vec3_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_FConvert_Rfloat3_rte(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_FConvert_Rfloat3_rte(__clc_vec3_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_FConvert_Rfloat3_rtn(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_FConvert_Rfloat3_rtn(__clc_vec3_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_FConvert_Rfloat3_rtp(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_FConvert_Rfloat3_rtp(__clc_vec3_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_FConvert_Rfloat3_rtz(__clc_vec3_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_FConvert_Rfloat3_rtz(__clc_vec3_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_FConvert_Rfloat4(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_FConvert_Rfloat4(__clc_vec4_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_FConvert_Rfloat4_rte(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_FConvert_Rfloat4_rte(__clc_vec4_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_FConvert_Rfloat4_rtn(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_FConvert_Rfloat4_rtn(__clc_vec4_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_FConvert_Rfloat4_rtp(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_FConvert_Rfloat4_rtp(__clc_vec4_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_FConvert_Rfloat4_rtz(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_FConvert_Rfloat4_rtz(__clc_vec4_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_FConvert_Rfloat8(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_FConvert_Rfloat8(__clc_vec8_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_FConvert_Rfloat8_rte(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_FConvert_Rfloat8_rte(__clc_vec8_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_FConvert_Rfloat8_rtn(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_FConvert_Rfloat8_rtn(__clc_vec8_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_FConvert_Rfloat8_rtp(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_FConvert_Rfloat8_rtp(__clc_vec8_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_FConvert_Rfloat8_rtz(__clc_vec8_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_FConvert_Rfloat8_rtz(__clc_vec8_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_FConvert_Rfloat_rte(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_FConvert_Rfloat_rte(__clc_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_FConvert_Rfloat_rtn(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_FConvert_Rfloat_rtn(__clc_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_FConvert_Rfloat_rtp(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_FConvert_Rfloat_rtp(__clc_fp16_t);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_FConvert_Rfloat_rtz(__clc_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_FConvert_Rfloat_rtz(__clc_fp16_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_FConvert_Rhalf(__clc_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_FConvert_Rhalf(__clc_fp64_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_FConvert_Rhalf16(__clc_vec16_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_FConvert_Rhalf16(__clc_vec16_fp64_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_FConvert_Rhalf16_rte(__clc_vec16_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_FConvert_Rhalf16_rte(__clc_vec16_fp64_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_FConvert_Rhalf16_rtn(__clc_vec16_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_FConvert_Rhalf16_rtn(__clc_vec16_fp64_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_FConvert_Rhalf16_rtp(__clc_vec16_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_FConvert_Rhalf16_rtp(__clc_vec16_fp64_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_FConvert_Rhalf16_rtz(__clc_vec16_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_FConvert_Rhalf16_rtz(__clc_vec16_fp64_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_FConvert_Rhalf2(__clc_vec2_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_FConvert_Rhalf2(__clc_vec2_fp64_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_FConvert_Rhalf2_rte(__clc_vec2_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_FConvert_Rhalf2_rte(__clc_vec2_fp64_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_FConvert_Rhalf2_rtn(__clc_vec2_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_FConvert_Rhalf2_rtn(__clc_vec2_fp64_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_FConvert_Rhalf2_rtp(__clc_vec2_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_FConvert_Rhalf2_rtp(__clc_vec2_fp64_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_FConvert_Rhalf2_rtz(__clc_vec2_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_FConvert_Rhalf2_rtz(__clc_vec2_fp64_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_FConvert_Rhalf3(__clc_vec3_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_FConvert_Rhalf3(__clc_vec3_fp64_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_FConvert_Rhalf3_rte(__clc_vec3_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_FConvert_Rhalf3_rte(__clc_vec3_fp64_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_FConvert_Rhalf3_rtn(__clc_vec3_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_FConvert_Rhalf3_rtn(__clc_vec3_fp64_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_FConvert_Rhalf3_rtp(__clc_vec3_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_FConvert_Rhalf3_rtp(__clc_vec3_fp64_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_FConvert_Rhalf3_rtz(__clc_vec3_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_FConvert_Rhalf3_rtz(__clc_vec3_fp64_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_FConvert_Rhalf4(__clc_vec4_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_FConvert_Rhalf4(__clc_vec4_fp64_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_FConvert_Rhalf4_rte(__clc_vec4_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_FConvert_Rhalf4_rte(__clc_vec4_fp64_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_FConvert_Rhalf4_rtn(__clc_vec4_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_FConvert_Rhalf4_rtn(__clc_vec4_fp64_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_FConvert_Rhalf4_rtp(__clc_vec4_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_FConvert_Rhalf4_rtp(__clc_vec4_fp64_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_FConvert_Rhalf4_rtz(__clc_vec4_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_FConvert_Rhalf4_rtz(__clc_vec4_fp64_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_FConvert_Rhalf8(__clc_vec8_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_FConvert_Rhalf8(__clc_vec8_fp64_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_FConvert_Rhalf8_rte(__clc_vec8_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_FConvert_Rhalf8_rte(__clc_vec8_fp64_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_FConvert_Rhalf8_rtn(__clc_vec8_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_FConvert_Rhalf8_rtn(__clc_vec8_fp64_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_FConvert_Rhalf8_rtp(__clc_vec8_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_FConvert_Rhalf8_rtp(__clc_vec8_fp64_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_FConvert_Rhalf8_rtz(__clc_vec8_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_FConvert_Rhalf8_rtz(__clc_vec8_fp64_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_FConvert_Rhalf_rte(__clc_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_FConvert_Rhalf_rte(__clc_fp64_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_FConvert_Rhalf_rtn(__clc_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_FConvert_Rhalf_rtn(__clc_fp64_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_FConvert_Rhalf_rtp(__clc_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_FConvert_Rhalf_rtp(__clc_fp64_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_FConvert_Rhalf_rtz(__clc_fp32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_FConvert_Rhalf_rtz(__clc_fp64_t);
#endif
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int8_t __spirv_SConvert_Rchar(__clc_int16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int8_t __spirv_SConvert_Rchar(__clc_int32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int8_t __spirv_SConvert_Rchar(__clc_int64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int8_t __spirv_SConvert_Rchar(__clc_uint16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int8_t __spirv_SConvert_Rchar(__clc_uint32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int8_t __spirv_SConvert_Rchar(__clc_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_SConvert_Rchar16(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_SConvert_Rchar16(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_SConvert_Rchar16(__clc_vec16_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_SConvert_Rchar16(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_SConvert_Rchar16(__clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_SConvert_Rchar16(__clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_SConvert_Rchar16_sat(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_SConvert_Rchar16_sat(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_SConvert_Rchar16_sat(__clc_vec16_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_SConvert_Rchar16_sat(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_SConvert_Rchar16_sat(__clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_SConvert_Rchar16_sat(__clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_SConvert_Rchar2(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_SConvert_Rchar2(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_SConvert_Rchar2(__clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_SConvert_Rchar2(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_SConvert_Rchar2(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_SConvert_Rchar2(__clc_vec2_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_SConvert_Rchar2_sat(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_SConvert_Rchar2_sat(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_SConvert_Rchar2_sat(__clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_SConvert_Rchar2_sat(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_SConvert_Rchar2_sat(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_SConvert_Rchar2_sat(__clc_vec2_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_SConvert_Rchar3(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_SConvert_Rchar3(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_SConvert_Rchar3(__clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_SConvert_Rchar3(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_SConvert_Rchar3(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_SConvert_Rchar3(__clc_vec3_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_SConvert_Rchar3_sat(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_SConvert_Rchar3_sat(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_SConvert_Rchar3_sat(__clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_SConvert_Rchar3_sat(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_SConvert_Rchar3_sat(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_SConvert_Rchar3_sat(__clc_vec3_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_SConvert_Rchar4(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_SConvert_Rchar4(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_SConvert_Rchar4(__clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_SConvert_Rchar4(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_SConvert_Rchar4(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_SConvert_Rchar4(__clc_vec4_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_SConvert_Rchar4_sat(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_SConvert_Rchar4_sat(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_SConvert_Rchar4_sat(__clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_SConvert_Rchar4_sat(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_SConvert_Rchar4_sat(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_SConvert_Rchar4_sat(__clc_vec4_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_SConvert_Rchar8(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_SConvert_Rchar8(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_SConvert_Rchar8(__clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_SConvert_Rchar8(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_SConvert_Rchar8(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_SConvert_Rchar8(__clc_vec8_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_SConvert_Rchar8_sat(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_SConvert_Rchar8_sat(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_SConvert_Rchar8_sat(__clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_SConvert_Rchar8_sat(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_SConvert_Rchar8_sat(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_SConvert_Rchar8_sat(__clc_vec8_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_SConvert_Rchar_sat(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_SConvert_Rchar_sat(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_SConvert_Rchar_sat(__clc_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_SConvert_Rchar_sat(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_SConvert_Rchar_sat(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_SConvert_Rchar_sat(__clc_uint64_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int32_t __spirv_SConvert_Rint(__clc_int8_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int32_t __spirv_SConvert_Rint(__clc_int16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int32_t __spirv_SConvert_Rint(__clc_int64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int32_t __spirv_SConvert_Rint(__clc_uint8_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int32_t __spirv_SConvert_Rint(__clc_uint16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int32_t __spirv_SConvert_Rint(__clc_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_SConvert_Rint16(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_SConvert_Rint16(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_SConvert_Rint16(__clc_vec16_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_SConvert_Rint16(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_SConvert_Rint16(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_SConvert_Rint16(__clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_SConvert_Rint16_sat(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_SConvert_Rint16_sat(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_SConvert_Rint16_sat(__clc_vec16_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_SConvert_Rint16_sat(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_SConvert_Rint16_sat(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_SConvert_Rint16_sat(__clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_SConvert_Rint2(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_SConvert_Rint2(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_SConvert_Rint2(__clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_SConvert_Rint2(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_SConvert_Rint2(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_SConvert_Rint2(__clc_vec2_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_SConvert_Rint2_sat(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_SConvert_Rint2_sat(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_SConvert_Rint2_sat(__clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_SConvert_Rint2_sat(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_SConvert_Rint2_sat(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_SConvert_Rint2_sat(__clc_vec2_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_SConvert_Rint3(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_SConvert_Rint3(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_SConvert_Rint3(__clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_SConvert_Rint3(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_SConvert_Rint3(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_SConvert_Rint3(__clc_vec3_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_SConvert_Rint3_sat(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_SConvert_Rint3_sat(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_SConvert_Rint3_sat(__clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_SConvert_Rint3_sat(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_SConvert_Rint3_sat(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_SConvert_Rint3_sat(__clc_vec3_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_SConvert_Rint4(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_SConvert_Rint4(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_SConvert_Rint4(__clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_SConvert_Rint4(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_SConvert_Rint4(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_SConvert_Rint4(__clc_vec4_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_SConvert_Rint4_sat(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_SConvert_Rint4_sat(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_SConvert_Rint4_sat(__clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_SConvert_Rint4_sat(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_SConvert_Rint4_sat(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_SConvert_Rint4_sat(__clc_vec4_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_SConvert_Rint8(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_SConvert_Rint8(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_SConvert_Rint8(__clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_SConvert_Rint8(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_SConvert_Rint8(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_SConvert_Rint8(__clc_vec8_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_SConvert_Rint8_sat(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_SConvert_Rint8_sat(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_SConvert_Rint8_sat(__clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_SConvert_Rint8_sat(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_SConvert_Rint8_sat(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_SConvert_Rint8_sat(__clc_vec8_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_SConvert_Rint_sat(__clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_SConvert_Rint_sat(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_SConvert_Rint_sat(__clc_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_SConvert_Rint_sat(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_SConvert_Rint_sat(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_SConvert_Rint_sat(__clc_uint64_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int64_t __spirv_SConvert_Rlong(__clc_int8_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int64_t __spirv_SConvert_Rlong(__clc_int16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int64_t __spirv_SConvert_Rlong(__clc_int32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int64_t __spirv_SConvert_Rlong(__clc_uint8_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int64_t __spirv_SConvert_Rlong(__clc_uint16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int64_t __spirv_SConvert_Rlong(__clc_uint32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_SConvert_Rlong16(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_SConvert_Rlong16(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_SConvert_Rlong16(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_SConvert_Rlong16(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_SConvert_Rlong16(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_SConvert_Rlong16(__clc_vec16_uint32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_SConvert_Rlong16_sat(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_SConvert_Rlong16_sat(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_SConvert_Rlong16_sat(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_SConvert_Rlong16_sat(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_SConvert_Rlong16_sat(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_SConvert_Rlong16_sat(__clc_vec16_uint32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_SConvert_Rlong2(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_SConvert_Rlong2(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_SConvert_Rlong2(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_SConvert_Rlong2(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_SConvert_Rlong2(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_SConvert_Rlong2(__clc_vec2_uint32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_SConvert_Rlong2_sat(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_SConvert_Rlong2_sat(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_SConvert_Rlong2_sat(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_SConvert_Rlong2_sat(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_SConvert_Rlong2_sat(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_SConvert_Rlong2_sat(__clc_vec2_uint32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_SConvert_Rlong3(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_SConvert_Rlong3(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_SConvert_Rlong3(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_SConvert_Rlong3(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_SConvert_Rlong3(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_SConvert_Rlong3(__clc_vec3_uint32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_SConvert_Rlong3_sat(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_SConvert_Rlong3_sat(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_SConvert_Rlong3_sat(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_SConvert_Rlong3_sat(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_SConvert_Rlong3_sat(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_SConvert_Rlong3_sat(__clc_vec3_uint32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_SConvert_Rlong4(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_SConvert_Rlong4(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_SConvert_Rlong4(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_SConvert_Rlong4(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_SConvert_Rlong4(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_SConvert_Rlong4(__clc_vec4_uint32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_SConvert_Rlong4_sat(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_SConvert_Rlong4_sat(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_SConvert_Rlong4_sat(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_SConvert_Rlong4_sat(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_SConvert_Rlong4_sat(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_SConvert_Rlong4_sat(__clc_vec4_uint32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_SConvert_Rlong8(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_SConvert_Rlong8(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_SConvert_Rlong8(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_SConvert_Rlong8(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_SConvert_Rlong8(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_SConvert_Rlong8(__clc_vec8_uint32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_SConvert_Rlong8_sat(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_SConvert_Rlong8_sat(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_SConvert_Rlong8_sat(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_SConvert_Rlong8_sat(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_SConvert_Rlong8_sat(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_SConvert_Rlong8_sat(__clc_vec8_uint32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_SConvert_Rlong_sat(__clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_SConvert_Rlong_sat(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_SConvert_Rlong_sat(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_SConvert_Rlong_sat(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_SConvert_Rlong_sat(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_SConvert_Rlong_sat(__clc_uint32_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int16_t __spirv_SConvert_Rshort(__clc_int8_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int16_t __spirv_SConvert_Rshort(__clc_int32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int16_t __spirv_SConvert_Rshort(__clc_int64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int16_t __spirv_SConvert_Rshort(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_SConvert_Rshort(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_SConvert_Rshort(__clc_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_SConvert_Rshort16(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_SConvert_Rshort16(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_SConvert_Rshort16(__clc_vec16_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_SConvert_Rshort16(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_SConvert_Rshort16(__clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_SConvert_Rshort16(__clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_SConvert_Rshort16_sat(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_SConvert_Rshort16_sat(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_SConvert_Rshort16_sat(__clc_vec16_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_SConvert_Rshort16_sat(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_SConvert_Rshort16_sat(__clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_SConvert_Rshort16_sat(__clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_SConvert_Rshort2(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_SConvert_Rshort2(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_SConvert_Rshort2(__clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_SConvert_Rshort2(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_SConvert_Rshort2(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_SConvert_Rshort2(__clc_vec2_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_SConvert_Rshort2_sat(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_SConvert_Rshort2_sat(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_SConvert_Rshort2_sat(__clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_SConvert_Rshort2_sat(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_SConvert_Rshort2_sat(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_SConvert_Rshort2_sat(__clc_vec2_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_SConvert_Rshort3(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_SConvert_Rshort3(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_SConvert_Rshort3(__clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_SConvert_Rshort3(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_SConvert_Rshort3(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_SConvert_Rshort3(__clc_vec3_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_SConvert_Rshort3_sat(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_SConvert_Rshort3_sat(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_SConvert_Rshort3_sat(__clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_SConvert_Rshort3_sat(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_SConvert_Rshort3_sat(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_SConvert_Rshort3_sat(__clc_vec3_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_SConvert_Rshort4(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_SConvert_Rshort4(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_SConvert_Rshort4(__clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_SConvert_Rshort4(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_SConvert_Rshort4(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_SConvert_Rshort4(__clc_vec4_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_SConvert_Rshort4_sat(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_SConvert_Rshort4_sat(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_SConvert_Rshort4_sat(__clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_SConvert_Rshort4_sat(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_SConvert_Rshort4_sat(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_SConvert_Rshort4_sat(__clc_vec4_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_SConvert_Rshort8(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_SConvert_Rshort8(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_SConvert_Rshort8(__clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_SConvert_Rshort8(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_SConvert_Rshort8(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_SConvert_Rshort8(__clc_vec8_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_SConvert_Rshort8_sat(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_SConvert_Rshort8_sat(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_SConvert_Rshort8_sat(__clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_SConvert_Rshort8_sat(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_SConvert_Rshort8_sat(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_SConvert_Rshort8_sat(__clc_vec8_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_SConvert_Rshort_sat(__clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_SConvert_Rshort_sat(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_SConvert_Rshort_sat(__clc_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_SConvert_Rshort_sat(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_SConvert_Rshort_sat(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_SConvert_Rshort_sat(__clc_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_SatConvertSToU_Ruchar(__clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_SatConvertSToU_Ruchar(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_SatConvertSToU_Ruchar(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_SatConvertSToU_Ruchar(__clc_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_SatConvertSToU_Ruchar16(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_SatConvertSToU_Ruchar16(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_SatConvertSToU_Ruchar16(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_SatConvertSToU_Ruchar16(__clc_vec16_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_SatConvertSToU_Ruchar2(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_SatConvertSToU_Ruchar2(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_SatConvertSToU_Ruchar2(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_SatConvertSToU_Ruchar2(__clc_vec2_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_SatConvertSToU_Ruchar3(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_SatConvertSToU_Ruchar3(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_SatConvertSToU_Ruchar3(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_SatConvertSToU_Ruchar3(__clc_vec3_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_SatConvertSToU_Ruchar4(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_SatConvertSToU_Ruchar4(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_SatConvertSToU_Ruchar4(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_SatConvertSToU_Ruchar4(__clc_vec4_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_SatConvertSToU_Ruchar8(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_SatConvertSToU_Ruchar8(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_SatConvertSToU_Ruchar8(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_SatConvertSToU_Ruchar8(__clc_vec8_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_SatConvertSToU_Ruint(__clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_SatConvertSToU_Ruint(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_SatConvertSToU_Ruint(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_SatConvertSToU_Ruint(__clc_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_SatConvertSToU_Ruint16(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_SatConvertSToU_Ruint16(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_SatConvertSToU_Ruint16(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_SatConvertSToU_Ruint16(__clc_vec16_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_SatConvertSToU_Ruint2(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_SatConvertSToU_Ruint2(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_SatConvertSToU_Ruint2(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_SatConvertSToU_Ruint2(__clc_vec2_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_SatConvertSToU_Ruint3(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_SatConvertSToU_Ruint3(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_SatConvertSToU_Ruint3(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_SatConvertSToU_Ruint3(__clc_vec3_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_SatConvertSToU_Ruint4(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_SatConvertSToU_Ruint4(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_SatConvertSToU_Ruint4(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_SatConvertSToU_Ruint4(__clc_vec4_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_SatConvertSToU_Ruint8(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_SatConvertSToU_Ruint8(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_SatConvertSToU_Ruint8(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_SatConvertSToU_Ruint8(__clc_vec8_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_SatConvertSToU_Rulong(__clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_SatConvertSToU_Rulong(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_SatConvertSToU_Rulong(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_SatConvertSToU_Rulong(__clc_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_SatConvertSToU_Rulong16(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_SatConvertSToU_Rulong16(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_SatConvertSToU_Rulong16(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_SatConvertSToU_Rulong16(__clc_vec16_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_SatConvertSToU_Rulong2(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_SatConvertSToU_Rulong2(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_SatConvertSToU_Rulong2(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_SatConvertSToU_Rulong2(__clc_vec2_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_SatConvertSToU_Rulong3(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_SatConvertSToU_Rulong3(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_SatConvertSToU_Rulong3(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_SatConvertSToU_Rulong3(__clc_vec3_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_SatConvertSToU_Rulong4(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_SatConvertSToU_Rulong4(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_SatConvertSToU_Rulong4(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_SatConvertSToU_Rulong4(__clc_vec4_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_SatConvertSToU_Rulong8(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_SatConvertSToU_Rulong8(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_SatConvertSToU_Rulong8(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_SatConvertSToU_Rulong8(__clc_vec8_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_SatConvertSToU_Rushort(__clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_SatConvertSToU_Rushort(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_SatConvertSToU_Rushort(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_SatConvertSToU_Rushort(__clc_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_SatConvertSToU_Rushort16(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_SatConvertSToU_Rushort16(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_SatConvertSToU_Rushort16(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_SatConvertSToU_Rushort16(__clc_vec16_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_SatConvertSToU_Rushort2(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_SatConvertSToU_Rushort2(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_SatConvertSToU_Rushort2(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_SatConvertSToU_Rushort2(__clc_vec2_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_SatConvertSToU_Rushort3(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_SatConvertSToU_Rushort3(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_SatConvertSToU_Rushort3(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_SatConvertSToU_Rushort3(__clc_vec3_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_SatConvertSToU_Rushort4(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_SatConvertSToU_Rushort4(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_SatConvertSToU_Rushort4(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_SatConvertSToU_Rushort4(__clc_vec4_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_SatConvertSToU_Rushort8(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_SatConvertSToU_Rushort8(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_SatConvertSToU_Rushort8(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_SatConvertSToU_Rushort8(__clc_vec8_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_SatConvertUToS_Rchar(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_SatConvertUToS_Rchar(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_SatConvertUToS_Rchar(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_SatConvertUToS_Rchar(__clc_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_SatConvertUToS_Rchar16(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_SatConvertUToS_Rchar16(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_SatConvertUToS_Rchar16(__clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_SatConvertUToS_Rchar16(__clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_SatConvertUToS_Rchar2(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_SatConvertUToS_Rchar2(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_SatConvertUToS_Rchar2(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_SatConvertUToS_Rchar2(__clc_vec2_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_SatConvertUToS_Rchar3(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_SatConvertUToS_Rchar3(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_SatConvertUToS_Rchar3(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_SatConvertUToS_Rchar3(__clc_vec3_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_SatConvertUToS_Rchar4(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_SatConvertUToS_Rchar4(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_SatConvertUToS_Rchar4(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_SatConvertUToS_Rchar4(__clc_vec4_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_SatConvertUToS_Rchar8(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_SatConvertUToS_Rchar8(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_SatConvertUToS_Rchar8(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_SatConvertUToS_Rchar8(__clc_vec8_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_SatConvertUToS_Rint(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_SatConvertUToS_Rint(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_SatConvertUToS_Rint(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_SatConvertUToS_Rint(__clc_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_SatConvertUToS_Rint16(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_SatConvertUToS_Rint16(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_SatConvertUToS_Rint16(__clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_SatConvertUToS_Rint16(__clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_SatConvertUToS_Rint2(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_SatConvertUToS_Rint2(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_SatConvertUToS_Rint2(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_SatConvertUToS_Rint2(__clc_vec2_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_SatConvertUToS_Rint3(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_SatConvertUToS_Rint3(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_SatConvertUToS_Rint3(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_SatConvertUToS_Rint3(__clc_vec3_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_SatConvertUToS_Rint4(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_SatConvertUToS_Rint4(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_SatConvertUToS_Rint4(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_SatConvertUToS_Rint4(__clc_vec4_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_SatConvertUToS_Rint8(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_SatConvertUToS_Rint8(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_SatConvertUToS_Rint8(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_SatConvertUToS_Rint8(__clc_vec8_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_SatConvertUToS_Rlong(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_SatConvertUToS_Rlong(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_SatConvertUToS_Rlong(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_SatConvertUToS_Rlong(__clc_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_SatConvertUToS_Rlong16(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_SatConvertUToS_Rlong16(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_SatConvertUToS_Rlong16(__clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_SatConvertUToS_Rlong16(__clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_SatConvertUToS_Rlong2(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_SatConvertUToS_Rlong2(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_SatConvertUToS_Rlong2(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_SatConvertUToS_Rlong2(__clc_vec2_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_SatConvertUToS_Rlong3(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_SatConvertUToS_Rlong3(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_SatConvertUToS_Rlong3(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_SatConvertUToS_Rlong3(__clc_vec3_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_SatConvertUToS_Rlong4(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_SatConvertUToS_Rlong4(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_SatConvertUToS_Rlong4(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_SatConvertUToS_Rlong4(__clc_vec4_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_SatConvertUToS_Rlong8(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_SatConvertUToS_Rlong8(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_SatConvertUToS_Rlong8(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_SatConvertUToS_Rlong8(__clc_vec8_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_SatConvertUToS_Rshort(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_SatConvertUToS_Rshort(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_SatConvertUToS_Rshort(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_SatConvertUToS_Rshort(__clc_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_SatConvertUToS_Rshort16(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_SatConvertUToS_Rshort16(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_SatConvertUToS_Rshort16(__clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_SatConvertUToS_Rshort16(__clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_SatConvertUToS_Rshort2(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_SatConvertUToS_Rshort2(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_SatConvertUToS_Rshort2(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_SatConvertUToS_Rshort2(__clc_vec2_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_SatConvertUToS_Rshort3(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_SatConvertUToS_Rshort3(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_SatConvertUToS_Rshort3(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_SatConvertUToS_Rshort3(__clc_vec3_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_SatConvertUToS_Rshort4(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_SatConvertUToS_Rshort4(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_SatConvertUToS_Rshort4(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_SatConvertUToS_Rshort4(__clc_vec4_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_SatConvertUToS_Rshort8(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_SatConvertUToS_Rshort8(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_SatConvertUToS_Rshort8(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_SatConvertUToS_Rshort8(__clc_vec8_uint64_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_uint8_t __spirv_UConvert_Ruchar(__clc_int16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_uint8_t __spirv_UConvert_Ruchar(__clc_int32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_uint8_t __spirv_UConvert_Ruchar(__clc_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_UConvert_Ruchar(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_UConvert_Ruchar(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_UConvert_Ruchar(__clc_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_UConvert_Ruchar16(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_UConvert_Ruchar16(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_UConvert_Ruchar16(__clc_vec16_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_UConvert_Ruchar16(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_UConvert_Ruchar16(__clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_UConvert_Ruchar16(__clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_UConvert_Ruchar16_sat(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_UConvert_Ruchar16_sat(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_UConvert_Ruchar16_sat(__clc_vec16_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_UConvert_Ruchar16_sat(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_UConvert_Ruchar16_sat(__clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_UConvert_Ruchar16_sat(__clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_UConvert_Ruchar2(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_UConvert_Ruchar2(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_UConvert_Ruchar2(__clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_UConvert_Ruchar2(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_UConvert_Ruchar2(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_UConvert_Ruchar2(__clc_vec2_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_UConvert_Ruchar2_sat(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_UConvert_Ruchar2_sat(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_UConvert_Ruchar2_sat(__clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_UConvert_Ruchar2_sat(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_UConvert_Ruchar2_sat(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_UConvert_Ruchar2_sat(__clc_vec2_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_UConvert_Ruchar3(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_UConvert_Ruchar3(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_UConvert_Ruchar3(__clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_UConvert_Ruchar3(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_UConvert_Ruchar3(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_UConvert_Ruchar3(__clc_vec3_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_UConvert_Ruchar3_sat(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_UConvert_Ruchar3_sat(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_UConvert_Ruchar3_sat(__clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_UConvert_Ruchar3_sat(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_UConvert_Ruchar3_sat(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_UConvert_Ruchar3_sat(__clc_vec3_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_UConvert_Ruchar4(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_UConvert_Ruchar4(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_UConvert_Ruchar4(__clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_UConvert_Ruchar4(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_UConvert_Ruchar4(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_UConvert_Ruchar4(__clc_vec4_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_UConvert_Ruchar4_sat(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_UConvert_Ruchar4_sat(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_UConvert_Ruchar4_sat(__clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_UConvert_Ruchar4_sat(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_UConvert_Ruchar4_sat(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_UConvert_Ruchar4_sat(__clc_vec4_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_UConvert_Ruchar8(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_UConvert_Ruchar8(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_UConvert_Ruchar8(__clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_UConvert_Ruchar8(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_UConvert_Ruchar8(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_UConvert_Ruchar8(__clc_vec8_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_UConvert_Ruchar8_sat(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_UConvert_Ruchar8_sat(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_UConvert_Ruchar8_sat(__clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_UConvert_Ruchar8_sat(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_UConvert_Ruchar8_sat(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_UConvert_Ruchar8_sat(__clc_vec8_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_UConvert_Ruchar_sat(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_UConvert_Ruchar_sat(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_UConvert_Ruchar_sat(__clc_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_UConvert_Ruchar_sat(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_UConvert_Ruchar_sat(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_UConvert_Ruchar_sat(__clc_uint64_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_uint32_t __spirv_UConvert_Ruint(__clc_int8_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_uint32_t __spirv_UConvert_Ruint(__clc_int16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_uint32_t __spirv_UConvert_Ruint(__clc_int64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_uint32_t __spirv_UConvert_Ruint(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_UConvert_Ruint(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_UConvert_Ruint(__clc_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_UConvert_Ruint16(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_UConvert_Ruint16(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_UConvert_Ruint16(__clc_vec16_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_UConvert_Ruint16(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_UConvert_Ruint16(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_UConvert_Ruint16(__clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_UConvert_Ruint16_sat(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_UConvert_Ruint16_sat(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_UConvert_Ruint16_sat(__clc_vec16_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_UConvert_Ruint16_sat(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_UConvert_Ruint16_sat(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_UConvert_Ruint16_sat(__clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_UConvert_Ruint2(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_UConvert_Ruint2(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_UConvert_Ruint2(__clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_UConvert_Ruint2(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_UConvert_Ruint2(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_UConvert_Ruint2(__clc_vec2_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_UConvert_Ruint2_sat(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_UConvert_Ruint2_sat(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_UConvert_Ruint2_sat(__clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_UConvert_Ruint2_sat(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_UConvert_Ruint2_sat(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_UConvert_Ruint2_sat(__clc_vec2_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_UConvert_Ruint3(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_UConvert_Ruint3(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_UConvert_Ruint3(__clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_UConvert_Ruint3(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_UConvert_Ruint3(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_UConvert_Ruint3(__clc_vec3_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_UConvert_Ruint3_sat(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_UConvert_Ruint3_sat(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_UConvert_Ruint3_sat(__clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_UConvert_Ruint3_sat(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_UConvert_Ruint3_sat(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_UConvert_Ruint3_sat(__clc_vec3_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_UConvert_Ruint4(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_UConvert_Ruint4(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_UConvert_Ruint4(__clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_UConvert_Ruint4(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_UConvert_Ruint4(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_UConvert_Ruint4(__clc_vec4_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_UConvert_Ruint4_sat(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_UConvert_Ruint4_sat(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_UConvert_Ruint4_sat(__clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_UConvert_Ruint4_sat(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_UConvert_Ruint4_sat(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_UConvert_Ruint4_sat(__clc_vec4_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_UConvert_Ruint8(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_UConvert_Ruint8(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_UConvert_Ruint8(__clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_UConvert_Ruint8(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_UConvert_Ruint8(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_UConvert_Ruint8(__clc_vec8_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_UConvert_Ruint8_sat(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_UConvert_Ruint8_sat(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_UConvert_Ruint8_sat(__clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_UConvert_Ruint8_sat(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_UConvert_Ruint8_sat(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_UConvert_Ruint8_sat(__clc_vec8_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_UConvert_Ruint_sat(__clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_UConvert_Ruint_sat(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_UConvert_Ruint_sat(__clc_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_UConvert_Ruint_sat(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_UConvert_Ruint_sat(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_UConvert_Ruint_sat(__clc_uint64_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_uint64_t __spirv_UConvert_Rulong(__clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_UConvert_Rulong(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_UConvert_Rulong(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_UConvert_Rulong(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_UConvert_Rulong(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_UConvert_Rulong(__clc_uint32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_UConvert_Rulong16(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_UConvert_Rulong16(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_UConvert_Rulong16(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_UConvert_Rulong16(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_UConvert_Rulong16(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_UConvert_Rulong16(__clc_vec16_uint32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_UConvert_Rulong16_sat(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_UConvert_Rulong16_sat(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_UConvert_Rulong16_sat(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_UConvert_Rulong16_sat(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_UConvert_Rulong16_sat(__clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_UConvert_Rulong16_sat(__clc_vec16_uint32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_UConvert_Rulong2(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_UConvert_Rulong2(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_UConvert_Rulong2(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_UConvert_Rulong2(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_UConvert_Rulong2(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_UConvert_Rulong2(__clc_vec2_uint32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_UConvert_Rulong2_sat(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_UConvert_Rulong2_sat(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_UConvert_Rulong2_sat(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_UConvert_Rulong2_sat(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_UConvert_Rulong2_sat(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_UConvert_Rulong2_sat(__clc_vec2_uint32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_UConvert_Rulong3(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_UConvert_Rulong3(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_UConvert_Rulong3(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_UConvert_Rulong3(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_UConvert_Rulong3(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_UConvert_Rulong3(__clc_vec3_uint32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_UConvert_Rulong3_sat(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_UConvert_Rulong3_sat(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_UConvert_Rulong3_sat(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_UConvert_Rulong3_sat(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_UConvert_Rulong3_sat(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_UConvert_Rulong3_sat(__clc_vec3_uint32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_UConvert_Rulong4(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_UConvert_Rulong4(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_UConvert_Rulong4(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_UConvert_Rulong4(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_UConvert_Rulong4(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_UConvert_Rulong4(__clc_vec4_uint32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_UConvert_Rulong4_sat(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_UConvert_Rulong4_sat(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_UConvert_Rulong4_sat(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_UConvert_Rulong4_sat(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_UConvert_Rulong4_sat(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_UConvert_Rulong4_sat(__clc_vec4_uint32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_UConvert_Rulong8(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_UConvert_Rulong8(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_UConvert_Rulong8(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_UConvert_Rulong8(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_UConvert_Rulong8(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_UConvert_Rulong8(__clc_vec8_uint32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_UConvert_Rulong8_sat(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_UConvert_Rulong8_sat(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_UConvert_Rulong8_sat(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_UConvert_Rulong8_sat(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_UConvert_Rulong8_sat(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_UConvert_Rulong8_sat(__clc_vec8_uint32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_UConvert_Rulong_sat(__clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_UConvert_Rulong_sat(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_UConvert_Rulong_sat(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_UConvert_Rulong_sat(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_UConvert_Rulong_sat(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_UConvert_Rulong_sat(__clc_uint32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_UConvert_Rushort(__clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_UConvert_Rushort(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_UConvert_Rushort(__clc_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_UConvert_Rushort(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_UConvert_Rushort(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_UConvert_Rushort(__clc_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_UConvert_Rushort16(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_UConvert_Rushort16(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_UConvert_Rushort16(__clc_vec16_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_UConvert_Rushort16(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_UConvert_Rushort16(__clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_UConvert_Rushort16(__clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_UConvert_Rushort16_sat(__clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_UConvert_Rushort16_sat(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_UConvert_Rushort16_sat(__clc_vec16_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_UConvert_Rushort16_sat(__clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_UConvert_Rushort16_sat(__clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_UConvert_Rushort16_sat(__clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_UConvert_Rushort2(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_UConvert_Rushort2(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_UConvert_Rushort2(__clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_UConvert_Rushort2(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_UConvert_Rushort2(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_UConvert_Rushort2(__clc_vec2_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_UConvert_Rushort2_sat(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_UConvert_Rushort2_sat(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_UConvert_Rushort2_sat(__clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_UConvert_Rushort2_sat(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_UConvert_Rushort2_sat(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_UConvert_Rushort2_sat(__clc_vec2_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_UConvert_Rushort3(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_UConvert_Rushort3(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_UConvert_Rushort3(__clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_UConvert_Rushort3(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_UConvert_Rushort3(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_UConvert_Rushort3(__clc_vec3_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_UConvert_Rushort3_sat(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_UConvert_Rushort3_sat(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_UConvert_Rushort3_sat(__clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_UConvert_Rushort3_sat(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_UConvert_Rushort3_sat(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_UConvert_Rushort3_sat(__clc_vec3_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_UConvert_Rushort4(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_UConvert_Rushort4(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_UConvert_Rushort4(__clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_UConvert_Rushort4(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_UConvert_Rushort4(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_UConvert_Rushort4(__clc_vec4_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_UConvert_Rushort4_sat(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_UConvert_Rushort4_sat(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_UConvert_Rushort4_sat(__clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_UConvert_Rushort4_sat(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_UConvert_Rushort4_sat(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_UConvert_Rushort4_sat(__clc_vec4_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_UConvert_Rushort8(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_UConvert_Rushort8(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_UConvert_Rushort8(__clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_UConvert_Rushort8(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_UConvert_Rushort8(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_UConvert_Rushort8(__clc_vec8_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_UConvert_Rushort8_sat(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_UConvert_Rushort8_sat(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_UConvert_Rushort8_sat(__clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_UConvert_Rushort8_sat(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_UConvert_Rushort8_sat(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_UConvert_Rushort8_sat(__clc_vec8_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_UConvert_Rushort_sat(__clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_UConvert_Rushort_sat(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_UConvert_Rushort_sat(__clc_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_UConvert_Rushort_sat(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_UConvert_Rushort_sat(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_UConvert_Rushort_sat(__clc_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_fclamp(__clc_fp32_t, __clc_fp32_t, __clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_fclamp(__clc_vec2_fp32_t, __clc_vec2_fp32_t, __clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_fclamp(__clc_vec3_fp32_t, __clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_fclamp(__clc_vec4_fp32_t, __clc_vec4_fp32_t, __clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_fclamp(__clc_vec8_fp32_t, __clc_vec8_fp32_t, __clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t __spirv_ocl_fclamp(
    __clc_vec16_fp32_t, __clc_vec16_fp32_t, __clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ocl_fclamp(__clc_fp64_t, __clc_fp64_t, __clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_fclamp(__clc_vec2_fp64_t, __clc_vec2_fp64_t, __clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_fclamp(__clc_vec3_fp64_t, __clc_vec3_fp64_t, __clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_fclamp(__clc_vec4_fp64_t, __clc_vec4_fp64_t, __clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_fclamp(__clc_vec8_fp64_t, __clc_vec8_fp64_t, __clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t __spirv_ocl_fclamp(
    __clc_vec16_fp64_t, __clc_vec16_fp64_t, __clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ocl_fclamp(__clc_fp16_t, __clc_fp16_t, __clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_fclamp(__clc_vec2_fp16_t, __clc_vec2_fp16_t, __clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_fclamp(__clc_vec3_fp16_t, __clc_vec3_fp16_t, __clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_fclamp(__clc_vec4_fp16_t, __clc_vec4_fp16_t, __clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_fclamp(__clc_vec8_fp16_t, __clc_vec8_fp16_t, __clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t __spirv_ocl_fclamp(
    __clc_vec16_fp16_t, __clc_vec16_fp16_t, __clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_uint8_t __spirv_ocl_s_abs(__clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ocl_s_abs(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ocl_s_abs(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ocl_s_abs(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ocl_s_abs(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ocl_s_abs(__clc_vec16_int8_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_uint16_t __spirv_ocl_s_abs(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ocl_s_abs(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ocl_s_abs(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ocl_s_abs(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ocl_s_abs(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ocl_s_abs(__clc_vec16_int16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_uint32_t __spirv_ocl_s_abs(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ocl_s_abs(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ocl_s_abs(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ocl_s_abs(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ocl_s_abs(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ocl_s_abs(__clc_vec16_int32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_uint64_t __spirv_ocl_s_abs(__clc_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ocl_s_abs(__clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ocl_s_abs(__clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ocl_s_abs(__clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ocl_s_abs(__clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ocl_s_abs(__clc_vec16_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ocl_s_abs_diff(__clc_int8_t, __clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ocl_s_abs_diff(__clc_vec2_int8_t, __clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ocl_s_abs_diff(__clc_vec3_int8_t, __clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ocl_s_abs_diff(__clc_vec4_int8_t, __clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ocl_s_abs_diff(__clc_vec8_int8_t, __clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ocl_s_abs_diff(__clc_vec16_int8_t, __clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ocl_s_abs_diff(__clc_int16_t, __clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ocl_s_abs_diff(__clc_vec2_int16_t, __clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ocl_s_abs_diff(__clc_vec3_int16_t, __clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ocl_s_abs_diff(__clc_vec4_int16_t, __clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ocl_s_abs_diff(__clc_vec8_int16_t, __clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ocl_s_abs_diff(__clc_vec16_int16_t, __clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ocl_s_abs_diff(__clc_int32_t, __clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ocl_s_abs_diff(__clc_vec2_int32_t, __clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ocl_s_abs_diff(__clc_vec3_int32_t, __clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ocl_s_abs_diff(__clc_vec4_int32_t, __clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ocl_s_abs_diff(__clc_vec8_int32_t, __clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ocl_s_abs_diff(__clc_vec16_int32_t, __clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ocl_s_abs_diff(__clc_int64_t, __clc_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ocl_s_abs_diff(__clc_vec2_int64_t, __clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ocl_s_abs_diff(__clc_vec3_int64_t, __clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ocl_s_abs_diff(__clc_vec4_int64_t, __clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ocl_s_abs_diff(__clc_vec8_int64_t, __clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ocl_s_abs_diff(__clc_vec16_int64_t, __clc_vec16_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ocl_s_add_sat(__clc_int8_t, __clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ocl_s_add_sat(__clc_vec2_int8_t, __clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ocl_s_add_sat(__clc_vec3_int8_t, __clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ocl_s_add_sat(__clc_vec4_int8_t, __clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ocl_s_add_sat(__clc_vec8_int8_t, __clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ocl_s_add_sat(__clc_vec16_int8_t, __clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ocl_s_add_sat(__clc_int16_t, __clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ocl_s_add_sat(__clc_vec2_int16_t, __clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ocl_s_add_sat(__clc_vec3_int16_t, __clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ocl_s_add_sat(__clc_vec4_int16_t, __clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ocl_s_add_sat(__clc_vec8_int16_t, __clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ocl_s_add_sat(__clc_vec16_int16_t, __clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ocl_s_add_sat(__clc_int32_t, __clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ocl_s_add_sat(__clc_vec2_int32_t, __clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ocl_s_add_sat(__clc_vec3_int32_t, __clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ocl_s_add_sat(__clc_vec4_int32_t, __clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ocl_s_add_sat(__clc_vec8_int32_t, __clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ocl_s_add_sat(__clc_vec16_int32_t, __clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ocl_s_add_sat(__clc_int64_t, __clc_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ocl_s_add_sat(__clc_vec2_int64_t, __clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ocl_s_add_sat(__clc_vec3_int64_t, __clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ocl_s_add_sat(__clc_vec4_int64_t, __clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ocl_s_add_sat(__clc_vec8_int64_t, __clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ocl_s_add_sat(__clc_vec16_int64_t, __clc_vec16_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ocl_s_clamp(__clc_int8_t, __clc_int8_t, __clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t __spirv_ocl_s_clamp(
    __clc_vec2_int8_t, __clc_vec2_int8_t, __clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t __spirv_ocl_s_clamp(
    __clc_vec3_int8_t, __clc_vec3_int8_t, __clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t __spirv_ocl_s_clamp(
    __clc_vec4_int8_t, __clc_vec4_int8_t, __clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t __spirv_ocl_s_clamp(
    __clc_vec8_int8_t, __clc_vec8_int8_t, __clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t __spirv_ocl_s_clamp(
    __clc_vec16_int8_t, __clc_vec16_int8_t, __clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ocl_s_clamp(__clc_int16_t, __clc_int16_t, __clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t __spirv_ocl_s_clamp(
    __clc_vec2_int16_t, __clc_vec2_int16_t, __clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t __spirv_ocl_s_clamp(
    __clc_vec3_int16_t, __clc_vec3_int16_t, __clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t __spirv_ocl_s_clamp(
    __clc_vec4_int16_t, __clc_vec4_int16_t, __clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t __spirv_ocl_s_clamp(
    __clc_vec8_int16_t, __clc_vec8_int16_t, __clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t __spirv_ocl_s_clamp(
    __clc_vec16_int16_t, __clc_vec16_int16_t, __clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ocl_s_clamp(__clc_int32_t, __clc_int32_t, __clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t __spirv_ocl_s_clamp(
    __clc_vec2_int32_t, __clc_vec2_int32_t, __clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t __spirv_ocl_s_clamp(
    __clc_vec3_int32_t, __clc_vec3_int32_t, __clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t __spirv_ocl_s_clamp(
    __clc_vec4_int32_t, __clc_vec4_int32_t, __clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t __spirv_ocl_s_clamp(
    __clc_vec8_int32_t, __clc_vec8_int32_t, __clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t __spirv_ocl_s_clamp(
    __clc_vec16_int32_t, __clc_vec16_int32_t, __clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ocl_s_clamp(__clc_int64_t, __clc_int64_t, __clc_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t __spirv_ocl_s_clamp(
    __clc_vec2_int64_t, __clc_vec2_int64_t, __clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t __spirv_ocl_s_clamp(
    __clc_vec3_int64_t, __clc_vec3_int64_t, __clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t __spirv_ocl_s_clamp(
    __clc_vec4_int64_t, __clc_vec4_int64_t, __clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t __spirv_ocl_s_clamp(
    __clc_vec8_int64_t, __clc_vec8_int64_t, __clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t __spirv_ocl_s_clamp(
    __clc_vec16_int64_t, __clc_vec16_int64_t, __clc_vec16_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ocl_s_hadd(__clc_int8_t, __clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ocl_s_hadd(__clc_vec2_int8_t, __clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ocl_s_hadd(__clc_vec3_int8_t, __clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ocl_s_hadd(__clc_vec4_int8_t, __clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ocl_s_hadd(__clc_vec8_int8_t, __clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ocl_s_hadd(__clc_vec16_int8_t, __clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ocl_s_hadd(__clc_int16_t, __clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ocl_s_hadd(__clc_vec2_int16_t, __clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ocl_s_hadd(__clc_vec3_int16_t, __clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ocl_s_hadd(__clc_vec4_int16_t, __clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ocl_s_hadd(__clc_vec8_int16_t, __clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ocl_s_hadd(__clc_vec16_int16_t, __clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ocl_s_hadd(__clc_int32_t, __clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ocl_s_hadd(__clc_vec2_int32_t, __clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ocl_s_hadd(__clc_vec3_int32_t, __clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ocl_s_hadd(__clc_vec4_int32_t, __clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ocl_s_hadd(__clc_vec8_int32_t, __clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ocl_s_hadd(__clc_vec16_int32_t, __clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ocl_s_hadd(__clc_int64_t, __clc_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ocl_s_hadd(__clc_vec2_int64_t, __clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ocl_s_hadd(__clc_vec3_int64_t, __clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ocl_s_hadd(__clc_vec4_int64_t, __clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ocl_s_hadd(__clc_vec8_int64_t, __clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ocl_s_hadd(__clc_vec16_int64_t, __clc_vec16_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ocl_s_mad24(__clc_int32_t, __clc_int32_t, __clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t __spirv_ocl_s_mad24(
    __clc_vec2_int32_t, __clc_vec2_int32_t, __clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t __spirv_ocl_s_mad24(
    __clc_vec3_int32_t, __clc_vec3_int32_t, __clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t __spirv_ocl_s_mad24(
    __clc_vec4_int32_t, __clc_vec4_int32_t, __clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t __spirv_ocl_s_mad24(
    __clc_vec8_int32_t, __clc_vec8_int32_t, __clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t __spirv_ocl_s_mad24(
    __clc_vec16_int32_t, __clc_vec16_int32_t, __clc_vec16_int32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ocl_s_mad_hi(__clc_int8_t, __clc_int8_t, __clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t __spirv_ocl_s_mad_hi(
    __clc_vec2_int8_t, __clc_vec2_int8_t, __clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t __spirv_ocl_s_mad_hi(
    __clc_vec3_int8_t, __clc_vec3_int8_t, __clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t __spirv_ocl_s_mad_hi(
    __clc_vec4_int8_t, __clc_vec4_int8_t, __clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t __spirv_ocl_s_mad_hi(
    __clc_vec8_int8_t, __clc_vec8_int8_t, __clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t __spirv_ocl_s_mad_hi(
    __clc_vec16_int8_t, __clc_vec16_int8_t, __clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ocl_s_mad_hi(__clc_int16_t, __clc_int16_t, __clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t __spirv_ocl_s_mad_hi(
    __clc_vec2_int16_t, __clc_vec2_int16_t, __clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t __spirv_ocl_s_mad_hi(
    __clc_vec3_int16_t, __clc_vec3_int16_t, __clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t __spirv_ocl_s_mad_hi(
    __clc_vec4_int16_t, __clc_vec4_int16_t, __clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t __spirv_ocl_s_mad_hi(
    __clc_vec8_int16_t, __clc_vec8_int16_t, __clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t __spirv_ocl_s_mad_hi(
    __clc_vec16_int16_t, __clc_vec16_int16_t, __clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ocl_s_mad_hi(__clc_int32_t, __clc_int32_t, __clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t __spirv_ocl_s_mad_hi(
    __clc_vec2_int32_t, __clc_vec2_int32_t, __clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t __spirv_ocl_s_mad_hi(
    __clc_vec3_int32_t, __clc_vec3_int32_t, __clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t __spirv_ocl_s_mad_hi(
    __clc_vec4_int32_t, __clc_vec4_int32_t, __clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t __spirv_ocl_s_mad_hi(
    __clc_vec8_int32_t, __clc_vec8_int32_t, __clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t __spirv_ocl_s_mad_hi(
    __clc_vec16_int32_t, __clc_vec16_int32_t, __clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ocl_s_mad_hi(__clc_int64_t, __clc_int64_t, __clc_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t __spirv_ocl_s_mad_hi(
    __clc_vec2_int64_t, __clc_vec2_int64_t, __clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t __spirv_ocl_s_mad_hi(
    __clc_vec3_int64_t, __clc_vec3_int64_t, __clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t __spirv_ocl_s_mad_hi(
    __clc_vec4_int64_t, __clc_vec4_int64_t, __clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t __spirv_ocl_s_mad_hi(
    __clc_vec8_int64_t, __clc_vec8_int64_t, __clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t __spirv_ocl_s_mad_hi(
    __clc_vec16_int64_t, __clc_vec16_int64_t, __clc_vec16_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ocl_s_mad_sat(__clc_int8_t, __clc_int8_t, __clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t __spirv_ocl_s_mad_sat(
    __clc_vec2_int8_t, __clc_vec2_int8_t, __clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t __spirv_ocl_s_mad_sat(
    __clc_vec3_int8_t, __clc_vec3_int8_t, __clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t __spirv_ocl_s_mad_sat(
    __clc_vec4_int8_t, __clc_vec4_int8_t, __clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t __spirv_ocl_s_mad_sat(
    __clc_vec8_int8_t, __clc_vec8_int8_t, __clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t __spirv_ocl_s_mad_sat(
    __clc_vec16_int8_t, __clc_vec16_int8_t, __clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ocl_s_mad_sat(__clc_int16_t, __clc_int16_t, __clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t __spirv_ocl_s_mad_sat(
    __clc_vec2_int16_t, __clc_vec2_int16_t, __clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t __spirv_ocl_s_mad_sat(
    __clc_vec3_int16_t, __clc_vec3_int16_t, __clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t __spirv_ocl_s_mad_sat(
    __clc_vec4_int16_t, __clc_vec4_int16_t, __clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t __spirv_ocl_s_mad_sat(
    __clc_vec8_int16_t, __clc_vec8_int16_t, __clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t __spirv_ocl_s_mad_sat(
    __clc_vec16_int16_t, __clc_vec16_int16_t, __clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ocl_s_mad_sat(__clc_int32_t, __clc_int32_t, __clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t __spirv_ocl_s_mad_sat(
    __clc_vec2_int32_t, __clc_vec2_int32_t, __clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t __spirv_ocl_s_mad_sat(
    __clc_vec3_int32_t, __clc_vec3_int32_t, __clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t __spirv_ocl_s_mad_sat(
    __clc_vec4_int32_t, __clc_vec4_int32_t, __clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t __spirv_ocl_s_mad_sat(
    __clc_vec8_int32_t, __clc_vec8_int32_t, __clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t __spirv_ocl_s_mad_sat(
    __clc_vec16_int32_t, __clc_vec16_int32_t, __clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ocl_s_mad_sat(__clc_int64_t, __clc_int64_t, __clc_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t __spirv_ocl_s_mad_sat(
    __clc_vec2_int64_t, __clc_vec2_int64_t, __clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t __spirv_ocl_s_mad_sat(
    __clc_vec3_int64_t, __clc_vec3_int64_t, __clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t __spirv_ocl_s_mad_sat(
    __clc_vec4_int64_t, __clc_vec4_int64_t, __clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t __spirv_ocl_s_mad_sat(
    __clc_vec8_int64_t, __clc_vec8_int64_t, __clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t __spirv_ocl_s_mad_sat(
    __clc_vec16_int64_t, __clc_vec16_int64_t, __clc_vec16_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ocl_s_max(__clc_int8_t, __clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ocl_s_max(__clc_vec2_int8_t, __clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ocl_s_max(__clc_vec3_int8_t, __clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ocl_s_max(__clc_vec4_int8_t, __clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ocl_s_max(__clc_vec8_int8_t, __clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ocl_s_max(__clc_vec16_int8_t, __clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ocl_s_max(__clc_int16_t, __clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ocl_s_max(__clc_vec2_int16_t, __clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ocl_s_max(__clc_vec3_int16_t, __clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ocl_s_max(__clc_vec4_int16_t, __clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ocl_s_max(__clc_vec8_int16_t, __clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ocl_s_max(__clc_vec16_int16_t, __clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ocl_s_max(__clc_int32_t, __clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ocl_s_max(__clc_vec2_int32_t, __clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ocl_s_max(__clc_vec3_int32_t, __clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ocl_s_max(__clc_vec4_int32_t, __clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ocl_s_max(__clc_vec8_int32_t, __clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ocl_s_max(__clc_vec16_int32_t, __clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ocl_s_max(__clc_int64_t, __clc_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ocl_s_max(__clc_vec2_int64_t, __clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ocl_s_max(__clc_vec3_int64_t, __clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ocl_s_max(__clc_vec4_int64_t, __clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ocl_s_max(__clc_vec8_int64_t, __clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ocl_s_max(__clc_vec16_int64_t, __clc_vec16_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ocl_s_min(__clc_int8_t, __clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ocl_s_min(__clc_vec2_int8_t, __clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ocl_s_min(__clc_vec3_int8_t, __clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ocl_s_min(__clc_vec4_int8_t, __clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ocl_s_min(__clc_vec8_int8_t, __clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ocl_s_min(__clc_vec16_int8_t, __clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ocl_s_min(__clc_int16_t, __clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ocl_s_min(__clc_vec2_int16_t, __clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ocl_s_min(__clc_vec3_int16_t, __clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ocl_s_min(__clc_vec4_int16_t, __clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ocl_s_min(__clc_vec8_int16_t, __clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ocl_s_min(__clc_vec16_int16_t, __clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ocl_s_min(__clc_int32_t, __clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ocl_s_min(__clc_vec2_int32_t, __clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ocl_s_min(__clc_vec3_int32_t, __clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ocl_s_min(__clc_vec4_int32_t, __clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ocl_s_min(__clc_vec8_int32_t, __clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ocl_s_min(__clc_vec16_int32_t, __clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ocl_s_min(__clc_int64_t, __clc_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ocl_s_min(__clc_vec2_int64_t, __clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ocl_s_min(__clc_vec3_int64_t, __clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ocl_s_min(__clc_vec4_int64_t, __clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ocl_s_min(__clc_vec8_int64_t, __clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ocl_s_min(__clc_vec16_int64_t, __clc_vec16_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ocl_s_mul24(__clc_int32_t, __clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ocl_s_mul24(__clc_vec2_int32_t, __clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ocl_s_mul24(__clc_vec3_int32_t, __clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ocl_s_mul24(__clc_vec4_int32_t, __clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ocl_s_mul24(__clc_vec8_int32_t, __clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ocl_s_mul24(__clc_vec16_int32_t, __clc_vec16_int32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ocl_s_mul_hi(__clc_int8_t, __clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ocl_s_mul_hi(__clc_vec2_int8_t, __clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ocl_s_mul_hi(__clc_vec3_int8_t, __clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ocl_s_mul_hi(__clc_vec4_int8_t, __clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ocl_s_mul_hi(__clc_vec8_int8_t, __clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ocl_s_mul_hi(__clc_vec16_int8_t, __clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ocl_s_mul_hi(__clc_int16_t, __clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ocl_s_mul_hi(__clc_vec2_int16_t, __clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ocl_s_mul_hi(__clc_vec3_int16_t, __clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ocl_s_mul_hi(__clc_vec4_int16_t, __clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ocl_s_mul_hi(__clc_vec8_int16_t, __clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ocl_s_mul_hi(__clc_vec16_int16_t, __clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ocl_s_mul_hi(__clc_int32_t, __clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ocl_s_mul_hi(__clc_vec2_int32_t, __clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ocl_s_mul_hi(__clc_vec3_int32_t, __clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ocl_s_mul_hi(__clc_vec4_int32_t, __clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ocl_s_mul_hi(__clc_vec8_int32_t, __clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ocl_s_mul_hi(__clc_vec16_int32_t, __clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ocl_s_mul_hi(__clc_int64_t, __clc_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ocl_s_mul_hi(__clc_vec2_int64_t, __clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ocl_s_mul_hi(__clc_vec3_int64_t, __clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ocl_s_mul_hi(__clc_vec4_int64_t, __clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ocl_s_mul_hi(__clc_vec8_int64_t, __clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ocl_s_mul_hi(__clc_vec16_int64_t, __clc_vec16_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ocl_s_rhadd(__clc_int8_t, __clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ocl_s_rhadd(__clc_vec2_int8_t, __clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ocl_s_rhadd(__clc_vec3_int8_t, __clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ocl_s_rhadd(__clc_vec4_int8_t, __clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ocl_s_rhadd(__clc_vec8_int8_t, __clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ocl_s_rhadd(__clc_vec16_int8_t, __clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ocl_s_rhadd(__clc_int16_t, __clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ocl_s_rhadd(__clc_vec2_int16_t, __clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ocl_s_rhadd(__clc_vec3_int16_t, __clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ocl_s_rhadd(__clc_vec4_int16_t, __clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ocl_s_rhadd(__clc_vec8_int16_t, __clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ocl_s_rhadd(__clc_vec16_int16_t, __clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ocl_s_rhadd(__clc_int32_t, __clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ocl_s_rhadd(__clc_vec2_int32_t, __clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ocl_s_rhadd(__clc_vec3_int32_t, __clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ocl_s_rhadd(__clc_vec4_int32_t, __clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ocl_s_rhadd(__clc_vec8_int32_t, __clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ocl_s_rhadd(__clc_vec16_int32_t, __clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ocl_s_rhadd(__clc_int64_t, __clc_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ocl_s_rhadd(__clc_vec2_int64_t, __clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ocl_s_rhadd(__clc_vec3_int64_t, __clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ocl_s_rhadd(__clc_vec4_int64_t, __clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ocl_s_rhadd(__clc_vec8_int64_t, __clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ocl_s_rhadd(__clc_vec16_int64_t, __clc_vec16_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ocl_s_sub_sat(__clc_int8_t, __clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ocl_s_sub_sat(__clc_vec2_int8_t, __clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ocl_s_sub_sat(__clc_vec3_int8_t, __clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ocl_s_sub_sat(__clc_vec4_int8_t, __clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ocl_s_sub_sat(__clc_vec8_int8_t, __clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ocl_s_sub_sat(__clc_vec16_int8_t, __clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ocl_s_sub_sat(__clc_int16_t, __clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ocl_s_sub_sat(__clc_vec2_int16_t, __clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ocl_s_sub_sat(__clc_vec3_int16_t, __clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ocl_s_sub_sat(__clc_vec4_int16_t, __clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ocl_s_sub_sat(__clc_vec8_int16_t, __clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ocl_s_sub_sat(__clc_vec16_int16_t, __clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ocl_s_sub_sat(__clc_int32_t, __clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ocl_s_sub_sat(__clc_vec2_int32_t, __clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ocl_s_sub_sat(__clc_vec3_int32_t, __clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ocl_s_sub_sat(__clc_vec4_int32_t, __clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ocl_s_sub_sat(__clc_vec8_int32_t, __clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ocl_s_sub_sat(__clc_vec16_int32_t, __clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ocl_s_sub_sat(__clc_int64_t, __clc_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ocl_s_sub_sat(__clc_vec2_int64_t, __clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ocl_s_sub_sat(__clc_vec3_int64_t, __clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ocl_s_sub_sat(__clc_vec4_int64_t, __clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ocl_s_sub_sat(__clc_vec8_int64_t, __clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ocl_s_sub_sat(__clc_vec16_int64_t, __clc_vec16_int64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ocl_s_upsample(__clc_int8_t, __clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ocl_s_upsample(__clc_vec2_int8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ocl_s_upsample(__clc_vec3_int8_t, __clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ocl_s_upsample(__clc_vec4_int8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ocl_s_upsample(__clc_vec8_int8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ocl_s_upsample(__clc_vec16_int8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ocl_s_upsample(__clc_int16_t, __clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ocl_s_upsample(__clc_vec2_int16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ocl_s_upsample(__clc_vec3_int16_t, __clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ocl_s_upsample(__clc_vec4_int16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ocl_s_upsample(__clc_vec8_int16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ocl_s_upsample(__clc_vec16_int16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ocl_s_upsample(__clc_int32_t, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ocl_s_upsample(__clc_vec2_int32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ocl_s_upsample(__clc_vec3_int32_t, __clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ocl_s_upsample(__clc_vec4_int32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ocl_s_upsample(__clc_vec8_int32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ocl_s_upsample(__clc_vec16_int32_t, __clc_vec16_uint32_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_uint8_t __spirv_ocl_u_abs(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ocl_u_abs(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ocl_u_abs(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ocl_u_abs(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ocl_u_abs(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ocl_u_abs(__clc_vec16_uint8_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_uint16_t __spirv_ocl_u_abs(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ocl_u_abs(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ocl_u_abs(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ocl_u_abs(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ocl_u_abs(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ocl_u_abs(__clc_vec16_uint16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_uint32_t __spirv_ocl_u_abs(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ocl_u_abs(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ocl_u_abs(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ocl_u_abs(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ocl_u_abs(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ocl_u_abs(__clc_vec16_uint32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_uint64_t __spirv_ocl_u_abs(__clc_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ocl_u_abs(__clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ocl_u_abs(__clc_vec3_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ocl_u_abs(__clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ocl_u_abs(__clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ocl_u_abs(__clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ocl_u_abs_diff(__clc_uint8_t, __clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ocl_u_abs_diff(__clc_vec2_uint8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ocl_u_abs_diff(__clc_vec3_uint8_t, __clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ocl_u_abs_diff(__clc_vec4_uint8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ocl_u_abs_diff(__clc_vec8_uint8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ocl_u_abs_diff(__clc_vec16_uint8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ocl_u_abs_diff(__clc_uint16_t, __clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ocl_u_abs_diff(__clc_vec2_uint16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ocl_u_abs_diff(__clc_vec3_uint16_t, __clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ocl_u_abs_diff(__clc_vec4_uint16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ocl_u_abs_diff(__clc_vec8_uint16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ocl_u_abs_diff(__clc_vec16_uint16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ocl_u_abs_diff(__clc_uint32_t, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ocl_u_abs_diff(__clc_vec2_uint32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ocl_u_abs_diff(__clc_vec3_uint32_t, __clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ocl_u_abs_diff(__clc_vec4_uint32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ocl_u_abs_diff(__clc_vec8_uint32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ocl_u_abs_diff(__clc_vec16_uint32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ocl_u_abs_diff(__clc_uint64_t, __clc_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ocl_u_abs_diff(__clc_vec2_uint64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ocl_u_abs_diff(__clc_vec3_uint64_t, __clc_vec3_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ocl_u_abs_diff(__clc_vec4_uint64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ocl_u_abs_diff(__clc_vec8_uint64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ocl_u_abs_diff(__clc_vec16_uint64_t, __clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ocl_u_add_sat(__clc_uint8_t, __clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ocl_u_add_sat(__clc_vec2_uint8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ocl_u_add_sat(__clc_vec3_uint8_t, __clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ocl_u_add_sat(__clc_vec4_uint8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ocl_u_add_sat(__clc_vec8_uint8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ocl_u_add_sat(__clc_vec16_uint8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ocl_u_add_sat(__clc_uint16_t, __clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ocl_u_add_sat(__clc_vec2_uint16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ocl_u_add_sat(__clc_vec3_uint16_t, __clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ocl_u_add_sat(__clc_vec4_uint16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ocl_u_add_sat(__clc_vec8_uint16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ocl_u_add_sat(__clc_vec16_uint16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ocl_u_add_sat(__clc_uint32_t, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ocl_u_add_sat(__clc_vec2_uint32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ocl_u_add_sat(__clc_vec3_uint32_t, __clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ocl_u_add_sat(__clc_vec4_uint32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ocl_u_add_sat(__clc_vec8_uint32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ocl_u_add_sat(__clc_vec16_uint32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ocl_u_add_sat(__clc_uint64_t, __clc_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ocl_u_add_sat(__clc_vec2_uint64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ocl_u_add_sat(__clc_vec3_uint64_t, __clc_vec3_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ocl_u_add_sat(__clc_vec4_uint64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ocl_u_add_sat(__clc_vec8_uint64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ocl_u_add_sat(__clc_vec16_uint64_t, __clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ocl_u_clamp(__clc_uint8_t, __clc_uint8_t, __clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t __spirv_ocl_u_clamp(
    __clc_vec2_uint8_t, __clc_vec2_uint8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t __spirv_ocl_u_clamp(
    __clc_vec3_uint8_t, __clc_vec3_uint8_t, __clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t __spirv_ocl_u_clamp(
    __clc_vec4_uint8_t, __clc_vec4_uint8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t __spirv_ocl_u_clamp(
    __clc_vec8_uint8_t, __clc_vec8_uint8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t __spirv_ocl_u_clamp(
    __clc_vec16_uint8_t, __clc_vec16_uint8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ocl_u_clamp(__clc_uint16_t, __clc_uint16_t, __clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t __spirv_ocl_u_clamp(
    __clc_vec2_uint16_t, __clc_vec2_uint16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t __spirv_ocl_u_clamp(
    __clc_vec3_uint16_t, __clc_vec3_uint16_t, __clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t __spirv_ocl_u_clamp(
    __clc_vec4_uint16_t, __clc_vec4_uint16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t __spirv_ocl_u_clamp(
    __clc_vec8_uint16_t, __clc_vec8_uint16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t __spirv_ocl_u_clamp(
    __clc_vec16_uint16_t, __clc_vec16_uint16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ocl_u_clamp(__clc_uint32_t, __clc_uint32_t, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t __spirv_ocl_u_clamp(
    __clc_vec2_uint32_t, __clc_vec2_uint32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t __spirv_ocl_u_clamp(
    __clc_vec3_uint32_t, __clc_vec3_uint32_t, __clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t __spirv_ocl_u_clamp(
    __clc_vec4_uint32_t, __clc_vec4_uint32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t __spirv_ocl_u_clamp(
    __clc_vec8_uint32_t, __clc_vec8_uint32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t __spirv_ocl_u_clamp(
    __clc_vec16_uint32_t, __clc_vec16_uint32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ocl_u_clamp(__clc_uint64_t, __clc_uint64_t, __clc_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t __spirv_ocl_u_clamp(
    __clc_vec2_uint64_t, __clc_vec2_uint64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t __spirv_ocl_u_clamp(
    __clc_vec3_uint64_t, __clc_vec3_uint64_t, __clc_vec3_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t __spirv_ocl_u_clamp(
    __clc_vec4_uint64_t, __clc_vec4_uint64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t __spirv_ocl_u_clamp(
    __clc_vec8_uint64_t, __clc_vec8_uint64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t __spirv_ocl_u_clamp(
    __clc_vec16_uint64_t, __clc_vec16_uint64_t, __clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ocl_u_hadd(__clc_uint8_t, __clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ocl_u_hadd(__clc_vec2_uint8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ocl_u_hadd(__clc_vec3_uint8_t, __clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ocl_u_hadd(__clc_vec4_uint8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ocl_u_hadd(__clc_vec8_uint8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ocl_u_hadd(__clc_vec16_uint8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ocl_u_hadd(__clc_uint16_t, __clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ocl_u_hadd(__clc_vec2_uint16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ocl_u_hadd(__clc_vec3_uint16_t, __clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ocl_u_hadd(__clc_vec4_uint16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ocl_u_hadd(__clc_vec8_uint16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ocl_u_hadd(__clc_vec16_uint16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ocl_u_hadd(__clc_uint32_t, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ocl_u_hadd(__clc_vec2_uint32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ocl_u_hadd(__clc_vec3_uint32_t, __clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ocl_u_hadd(__clc_vec4_uint32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ocl_u_hadd(__clc_vec8_uint32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ocl_u_hadd(__clc_vec16_uint32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ocl_u_hadd(__clc_uint64_t, __clc_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ocl_u_hadd(__clc_vec2_uint64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ocl_u_hadd(__clc_vec3_uint64_t, __clc_vec3_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ocl_u_hadd(__clc_vec4_uint64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ocl_u_hadd(__clc_vec8_uint64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ocl_u_hadd(__clc_vec16_uint64_t, __clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ocl_u_mad24(__clc_uint32_t, __clc_uint32_t, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t __spirv_ocl_u_mad24(
    __clc_vec2_uint32_t, __clc_vec2_uint32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t __spirv_ocl_u_mad24(
    __clc_vec3_uint32_t, __clc_vec3_uint32_t, __clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t __spirv_ocl_u_mad24(
    __clc_vec4_uint32_t, __clc_vec4_uint32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t __spirv_ocl_u_mad24(
    __clc_vec8_uint32_t, __clc_vec8_uint32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t __spirv_ocl_u_mad24(
    __clc_vec16_uint32_t, __clc_vec16_uint32_t, __clc_vec16_uint32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ocl_u_mad_hi(__clc_uint8_t, __clc_uint8_t, __clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t __spirv_ocl_u_mad_hi(
    __clc_vec2_uint8_t, __clc_vec2_uint8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t __spirv_ocl_u_mad_hi(
    __clc_vec3_uint8_t, __clc_vec3_uint8_t, __clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t __spirv_ocl_u_mad_hi(
    __clc_vec4_uint8_t, __clc_vec4_uint8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t __spirv_ocl_u_mad_hi(
    __clc_vec8_uint8_t, __clc_vec8_uint8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t __spirv_ocl_u_mad_hi(
    __clc_vec16_uint8_t, __clc_vec16_uint8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ocl_u_mad_hi(__clc_uint16_t, __clc_uint16_t, __clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t __spirv_ocl_u_mad_hi(
    __clc_vec2_uint16_t, __clc_vec2_uint16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t __spirv_ocl_u_mad_hi(
    __clc_vec3_uint16_t, __clc_vec3_uint16_t, __clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t __spirv_ocl_u_mad_hi(
    __clc_vec4_uint16_t, __clc_vec4_uint16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t __spirv_ocl_u_mad_hi(
    __clc_vec8_uint16_t, __clc_vec8_uint16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t __spirv_ocl_u_mad_hi(
    __clc_vec16_uint16_t, __clc_vec16_uint16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ocl_u_mad_hi(__clc_uint32_t, __clc_uint32_t, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t __spirv_ocl_u_mad_hi(
    __clc_vec2_uint32_t, __clc_vec2_uint32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t __spirv_ocl_u_mad_hi(
    __clc_vec3_uint32_t, __clc_vec3_uint32_t, __clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t __spirv_ocl_u_mad_hi(
    __clc_vec4_uint32_t, __clc_vec4_uint32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t __spirv_ocl_u_mad_hi(
    __clc_vec8_uint32_t, __clc_vec8_uint32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t __spirv_ocl_u_mad_hi(
    __clc_vec16_uint32_t, __clc_vec16_uint32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ocl_u_mad_hi(__clc_uint64_t, __clc_uint64_t, __clc_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t __spirv_ocl_u_mad_hi(
    __clc_vec2_uint64_t, __clc_vec2_uint64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t __spirv_ocl_u_mad_hi(
    __clc_vec3_uint64_t, __clc_vec3_uint64_t, __clc_vec3_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t __spirv_ocl_u_mad_hi(
    __clc_vec4_uint64_t, __clc_vec4_uint64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t __spirv_ocl_u_mad_hi(
    __clc_vec8_uint64_t, __clc_vec8_uint64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t __spirv_ocl_u_mad_hi(
    __clc_vec16_uint64_t, __clc_vec16_uint64_t, __clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ocl_u_mad_sat(__clc_uint8_t, __clc_uint8_t, __clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t __spirv_ocl_u_mad_sat(
    __clc_vec2_uint8_t, __clc_vec2_uint8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t __spirv_ocl_u_mad_sat(
    __clc_vec3_uint8_t, __clc_vec3_uint8_t, __clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t __spirv_ocl_u_mad_sat(
    __clc_vec4_uint8_t, __clc_vec4_uint8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t __spirv_ocl_u_mad_sat(
    __clc_vec8_uint8_t, __clc_vec8_uint8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t __spirv_ocl_u_mad_sat(
    __clc_vec16_uint8_t, __clc_vec16_uint8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ocl_u_mad_sat(__clc_uint16_t, __clc_uint16_t, __clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t __spirv_ocl_u_mad_sat(
    __clc_vec2_uint16_t, __clc_vec2_uint16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t __spirv_ocl_u_mad_sat(
    __clc_vec3_uint16_t, __clc_vec3_uint16_t, __clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t __spirv_ocl_u_mad_sat(
    __clc_vec4_uint16_t, __clc_vec4_uint16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t __spirv_ocl_u_mad_sat(
    __clc_vec8_uint16_t, __clc_vec8_uint16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t __spirv_ocl_u_mad_sat(
    __clc_vec16_uint16_t, __clc_vec16_uint16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ocl_u_mad_sat(__clc_uint32_t, __clc_uint32_t, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t __spirv_ocl_u_mad_sat(
    __clc_vec2_uint32_t, __clc_vec2_uint32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t __spirv_ocl_u_mad_sat(
    __clc_vec3_uint32_t, __clc_vec3_uint32_t, __clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t __spirv_ocl_u_mad_sat(
    __clc_vec4_uint32_t, __clc_vec4_uint32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t __spirv_ocl_u_mad_sat(
    __clc_vec8_uint32_t, __clc_vec8_uint32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t __spirv_ocl_u_mad_sat(
    __clc_vec16_uint32_t, __clc_vec16_uint32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ocl_u_mad_sat(__clc_uint64_t, __clc_uint64_t, __clc_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t __spirv_ocl_u_mad_sat(
    __clc_vec2_uint64_t, __clc_vec2_uint64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t __spirv_ocl_u_mad_sat(
    __clc_vec3_uint64_t, __clc_vec3_uint64_t, __clc_vec3_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t __spirv_ocl_u_mad_sat(
    __clc_vec4_uint64_t, __clc_vec4_uint64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t __spirv_ocl_u_mad_sat(
    __clc_vec8_uint64_t, __clc_vec8_uint64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t __spirv_ocl_u_mad_sat(
    __clc_vec16_uint64_t, __clc_vec16_uint64_t, __clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ocl_u_max(__clc_uint8_t, __clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ocl_u_max(__clc_vec2_uint8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ocl_u_max(__clc_vec3_uint8_t, __clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ocl_u_max(__clc_vec4_uint8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ocl_u_max(__clc_vec8_uint8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ocl_u_max(__clc_vec16_uint8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ocl_u_max(__clc_uint16_t, __clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ocl_u_max(__clc_vec2_uint16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ocl_u_max(__clc_vec3_uint16_t, __clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ocl_u_max(__clc_vec4_uint16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ocl_u_max(__clc_vec8_uint16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ocl_u_max(__clc_vec16_uint16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ocl_u_max(__clc_uint32_t, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ocl_u_max(__clc_vec2_uint32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ocl_u_max(__clc_vec3_uint32_t, __clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ocl_u_max(__clc_vec4_uint32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ocl_u_max(__clc_vec8_uint32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ocl_u_max(__clc_vec16_uint32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ocl_u_max(__clc_uint64_t, __clc_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ocl_u_max(__clc_vec2_uint64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ocl_u_max(__clc_vec3_uint64_t, __clc_vec3_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ocl_u_max(__clc_vec4_uint64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ocl_u_max(__clc_vec8_uint64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ocl_u_max(__clc_vec16_uint64_t, __clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ocl_u_min(__clc_uint8_t, __clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ocl_u_min(__clc_vec2_uint8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ocl_u_min(__clc_vec3_uint8_t, __clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ocl_u_min(__clc_vec4_uint8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ocl_u_min(__clc_vec8_uint8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ocl_u_min(__clc_vec16_uint8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ocl_u_min(__clc_uint16_t, __clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ocl_u_min(__clc_vec2_uint16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ocl_u_min(__clc_vec3_uint16_t, __clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ocl_u_min(__clc_vec4_uint16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ocl_u_min(__clc_vec8_uint16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ocl_u_min(__clc_vec16_uint16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ocl_u_min(__clc_uint32_t, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ocl_u_min(__clc_vec2_uint32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ocl_u_min(__clc_vec3_uint32_t, __clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ocl_u_min(__clc_vec4_uint32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ocl_u_min(__clc_vec8_uint32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ocl_u_min(__clc_vec16_uint32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ocl_u_min(__clc_uint64_t, __clc_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ocl_u_min(__clc_vec2_uint64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ocl_u_min(__clc_vec3_uint64_t, __clc_vec3_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ocl_u_min(__clc_vec4_uint64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ocl_u_min(__clc_vec8_uint64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ocl_u_min(__clc_vec16_uint64_t, __clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ocl_u_mul24(__clc_uint32_t, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ocl_u_mul24(__clc_vec2_uint32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ocl_u_mul24(__clc_vec3_uint32_t, __clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ocl_u_mul24(__clc_vec4_uint32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ocl_u_mul24(__clc_vec8_uint32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ocl_u_mul24(__clc_vec16_uint32_t, __clc_vec16_uint32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ocl_u_mul_hi(__clc_uint8_t, __clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ocl_u_mul_hi(__clc_vec2_uint8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ocl_u_mul_hi(__clc_vec3_uint8_t, __clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ocl_u_mul_hi(__clc_vec4_uint8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ocl_u_mul_hi(__clc_vec8_uint8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ocl_u_mul_hi(__clc_vec16_uint8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ocl_u_mul_hi(__clc_uint16_t, __clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ocl_u_mul_hi(__clc_vec2_uint16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ocl_u_mul_hi(__clc_vec3_uint16_t, __clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ocl_u_mul_hi(__clc_vec4_uint16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ocl_u_mul_hi(__clc_vec8_uint16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ocl_u_mul_hi(__clc_vec16_uint16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ocl_u_mul_hi(__clc_uint32_t, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ocl_u_mul_hi(__clc_vec2_uint32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ocl_u_mul_hi(__clc_vec3_uint32_t, __clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ocl_u_mul_hi(__clc_vec4_uint32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ocl_u_mul_hi(__clc_vec8_uint32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ocl_u_mul_hi(__clc_vec16_uint32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ocl_u_mul_hi(__clc_uint64_t, __clc_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ocl_u_mul_hi(__clc_vec2_uint64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ocl_u_mul_hi(__clc_vec3_uint64_t, __clc_vec3_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ocl_u_mul_hi(__clc_vec4_uint64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ocl_u_mul_hi(__clc_vec8_uint64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ocl_u_mul_hi(__clc_vec16_uint64_t, __clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ocl_u_rhadd(__clc_uint8_t, __clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ocl_u_rhadd(__clc_vec2_uint8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ocl_u_rhadd(__clc_vec3_uint8_t, __clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ocl_u_rhadd(__clc_vec4_uint8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ocl_u_rhadd(__clc_vec8_uint8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ocl_u_rhadd(__clc_vec16_uint8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ocl_u_rhadd(__clc_uint16_t, __clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ocl_u_rhadd(__clc_vec2_uint16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ocl_u_rhadd(__clc_vec3_uint16_t, __clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ocl_u_rhadd(__clc_vec4_uint16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ocl_u_rhadd(__clc_vec8_uint16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ocl_u_rhadd(__clc_vec16_uint16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ocl_u_rhadd(__clc_uint32_t, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ocl_u_rhadd(__clc_vec2_uint32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ocl_u_rhadd(__clc_vec3_uint32_t, __clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ocl_u_rhadd(__clc_vec4_uint32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ocl_u_rhadd(__clc_vec8_uint32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ocl_u_rhadd(__clc_vec16_uint32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ocl_u_rhadd(__clc_uint64_t, __clc_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ocl_u_rhadd(__clc_vec2_uint64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ocl_u_rhadd(__clc_vec3_uint64_t, __clc_vec3_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ocl_u_rhadd(__clc_vec4_uint64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ocl_u_rhadd(__clc_vec8_uint64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ocl_u_rhadd(__clc_vec16_uint64_t, __clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ocl_u_sub_sat(__clc_uint8_t, __clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ocl_u_sub_sat(__clc_vec2_uint8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ocl_u_sub_sat(__clc_vec3_uint8_t, __clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ocl_u_sub_sat(__clc_vec4_uint8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ocl_u_sub_sat(__clc_vec8_uint8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ocl_u_sub_sat(__clc_vec16_uint8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ocl_u_sub_sat(__clc_uint16_t, __clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ocl_u_sub_sat(__clc_vec2_uint16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ocl_u_sub_sat(__clc_vec3_uint16_t, __clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ocl_u_sub_sat(__clc_vec4_uint16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ocl_u_sub_sat(__clc_vec8_uint16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ocl_u_sub_sat(__clc_vec16_uint16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ocl_u_sub_sat(__clc_uint32_t, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ocl_u_sub_sat(__clc_vec2_uint32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ocl_u_sub_sat(__clc_vec3_uint32_t, __clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ocl_u_sub_sat(__clc_vec4_uint32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ocl_u_sub_sat(__clc_vec8_uint32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ocl_u_sub_sat(__clc_vec16_uint32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ocl_u_sub_sat(__clc_uint64_t, __clc_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ocl_u_sub_sat(__clc_vec2_uint64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ocl_u_sub_sat(__clc_vec3_uint64_t, __clc_vec3_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ocl_u_sub_sat(__clc_vec4_uint64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ocl_u_sub_sat(__clc_vec8_uint64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ocl_u_sub_sat(__clc_vec16_uint64_t, __clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ocl_u_upsample(__clc_uint8_t, __clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ocl_u_upsample(__clc_vec2_uint8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ocl_u_upsample(__clc_vec3_uint8_t, __clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ocl_u_upsample(__clc_vec4_uint8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ocl_u_upsample(__clc_vec8_uint8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ocl_u_upsample(__clc_vec16_uint8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ocl_u_upsample(__clc_uint16_t, __clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ocl_u_upsample(__clc_vec2_uint16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ocl_u_upsample(__clc_vec3_uint16_t, __clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ocl_u_upsample(__clc_vec4_uint16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ocl_u_upsample(__clc_vec8_uint16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ocl_u_upsample(__clc_vec16_uint16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ocl_u_upsample(__clc_uint32_t, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ocl_u_upsample(__clc_vec2_uint32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ocl_u_upsample(__clc_vec3_uint32_t, __clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ocl_u_upsample(__clc_vec4_uint32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ocl_u_upsample(__clc_vec8_uint32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ocl_u_upsample(__clc_vec16_uint32_t, __clc_vec16_uint32_t);

#endif
