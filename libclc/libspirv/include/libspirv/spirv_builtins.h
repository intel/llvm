//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//
// Automatically generated file, do not edit!
//

#include <clc/clcfunc.h>
#include <libspirv/spirv_types.h>

#ifndef CLC_SPIRV_BINDING
#define CLC_SPIRV_BINDING

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT void
    __spirv_ControlBarrier(__clc_int32_t, __clc_int32_t, __clc_int32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t, __clc_char_t __local *, __clc_char_t const __global *,
    __clc_size_t, __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t, __clc_char_t __global *, __clc_char_t const __local *,
    __clc_size_t, __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec2_char_t __local *,
                       __clc_vec2_char_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec2_char_t __global *,
                       __clc_vec2_char_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec3_char_t __local *,
                       __clc_vec3_char_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec3_char_t __global *,
                       __clc_vec3_char_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec4_char_t __local *,
                       __clc_vec4_char_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec4_char_t __global *,
                       __clc_vec4_char_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec8_char_t __local *,
                       __clc_vec8_char_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec8_char_t __global *,
                       __clc_vec8_char_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec16_char_t __local *,
                       __clc_vec16_char_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec16_char_t __global *,
                       __clc_vec16_char_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_uint32_t, __clc_int8_t __local *, __clc_int8_t const __global *,
    __clc_size_t, __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_uint32_t, __clc_int8_t __global *, __clc_int8_t const __local *,
    __clc_size_t, __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec2_int8_t __local *,
                       __clc_vec2_int8_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec2_int8_t __global *,
                       __clc_vec2_int8_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec3_int8_t __local *,
                       __clc_vec3_int8_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec3_int8_t __global *,
                       __clc_vec3_int8_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec4_int8_t __local *,
                       __clc_vec4_int8_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec4_int8_t __global *,
                       __clc_vec4_int8_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec8_int8_t __local *,
                       __clc_vec8_int8_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec8_int8_t __global *,
                       __clc_vec8_int8_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec16_int8_t __local *,
                       __clc_vec16_int8_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec16_int8_t __global *,
                       __clc_vec16_int8_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t, __clc_int16_t __local *, __clc_int16_t const __global *,
    __clc_size_t, __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t, __clc_int16_t __global *, __clc_int16_t const __local *,
    __clc_size_t, __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec2_int16_t __local *,
                       __clc_vec2_int16_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec2_int16_t __global *,
                       __clc_vec2_int16_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec3_int16_t __local *,
                       __clc_vec3_int16_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec3_int16_t __global *,
                       __clc_vec3_int16_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec4_int16_t __local *,
                       __clc_vec4_int16_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec4_int16_t __global *,
                       __clc_vec4_int16_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec8_int16_t __local *,
                       __clc_vec8_int16_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec8_int16_t __global *,
                       __clc_vec8_int16_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec16_int16_t __local *,
                       __clc_vec16_int16_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec16_int16_t __global *,
                       __clc_vec16_int16_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t, __clc_int32_t __local *, __clc_int32_t const __global *,
    __clc_size_t, __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_uint32_t, __clc_int32_t __global *, __clc_int32_t const __local *,
    __clc_size_t, __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec2_int32_t __local *,
                       __clc_vec2_int32_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec2_int32_t __global *,
                       __clc_vec2_int32_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec3_int32_t __local *,
                       __clc_vec3_int32_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec3_int32_t __global *,
                       __clc_vec3_int32_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec4_int32_t __local *,
                       __clc_vec4_int32_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec4_int32_t __global *,
                       __clc_vec4_int32_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec8_int32_t __local *,
                       __clc_vec8_int32_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec8_int32_t __global *,
                       __clc_vec8_int32_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec16_int32_t __local *,
                       __clc_vec16_int32_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec16_int32_t __global *,
                       __clc_vec16_int32_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_uint32_t, __clc_int64_t __local *, __clc_int64_t const __global *,
    __clc_size_t, __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_uint32_t, __clc_int64_t __global *, __clc_int64_t const __local *,
    __clc_size_t, __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec2_int64_t __local *,
                       __clc_vec2_int64_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec2_int64_t __global *,
                       __clc_vec2_int64_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec3_int64_t __local *,
                       __clc_vec3_int64_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec3_int64_t __global *,
                       __clc_vec3_int64_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec4_int64_t __local *,
                       __clc_vec4_int64_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec4_int64_t __global *,
                       __clc_vec4_int64_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec8_int64_t __local *,
                       __clc_vec8_int64_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec8_int64_t __global *,
                       __clc_vec8_int64_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec16_int64_t __local *,
                       __clc_vec16_int64_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec16_int64_t __global *,
                       __clc_vec16_int64_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t, __clc_uint8_t __local *, __clc_uint8_t const __global *,
    __clc_size_t, __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t, __clc_uint8_t __global *, __clc_uint8_t const __local *,
    __clc_size_t, __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec2_uint8_t __local *,
                       __clc_vec2_uint8_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec2_uint8_t __global *,
                       __clc_vec2_uint8_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec3_uint8_t __local *,
                       __clc_vec3_uint8_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec3_uint8_t __global *,
                       __clc_vec3_uint8_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec4_uint8_t __local *,
                       __clc_vec4_uint8_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec4_uint8_t __global *,
                       __clc_vec4_uint8_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec8_uint8_t __local *,
                       __clc_vec8_uint8_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec8_uint8_t __global *,
                       __clc_vec8_uint8_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec16_uint8_t __local *,
                       __clc_vec16_uint8_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec16_uint8_t __global *,
                       __clc_vec16_uint8_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t, __clc_uint16_t __local *, __clc_uint16_t const __global *,
    __clc_size_t, __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t, __clc_uint16_t __global *, __clc_uint16_t const __local *,
    __clc_size_t, __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec2_uint16_t __local *,
                       __clc_vec2_uint16_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec2_uint16_t __global *,
                       __clc_vec2_uint16_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec3_uint16_t __local *,
                       __clc_vec3_uint16_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec3_uint16_t __global *,
                       __clc_vec3_uint16_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec4_uint16_t __local *,
                       __clc_vec4_uint16_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec4_uint16_t __global *,
                       __clc_vec4_uint16_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec8_uint16_t __local *,
                       __clc_vec8_uint16_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec8_uint16_t __global *,
                       __clc_vec8_uint16_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec16_uint16_t __local *,
                       __clc_vec16_uint16_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec16_uint16_t __global *,
                       __clc_vec16_uint16_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t, __clc_uint32_t __local *, __clc_uint32_t const __global *,
    __clc_size_t, __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t, __clc_uint32_t __global *, __clc_uint32_t const __local *,
    __clc_size_t, __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec2_uint32_t __local *,
                       __clc_vec2_uint32_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec2_uint32_t __global *,
                       __clc_vec2_uint32_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec3_uint32_t __local *,
                       __clc_vec3_uint32_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec3_uint32_t __global *,
                       __clc_vec3_uint32_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec4_uint32_t __local *,
                       __clc_vec4_uint32_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec4_uint32_t __global *,
                       __clc_vec4_uint32_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec8_uint32_t __local *,
                       __clc_vec8_uint32_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec8_uint32_t __global *,
                       __clc_vec8_uint32_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec16_uint32_t __local *,
                       __clc_vec16_uint32_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec16_uint32_t __global *,
                       __clc_vec16_uint32_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t, __clc_uint64_t __local *, __clc_uint64_t const __global *,
    __clc_size_t, __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t, __clc_uint64_t __global *, __clc_uint64_t const __local *,
    __clc_size_t, __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec2_uint64_t __local *,
                       __clc_vec2_uint64_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec2_uint64_t __global *,
                       __clc_vec2_uint64_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec3_uint64_t __local *,
                       __clc_vec3_uint64_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec3_uint64_t __global *,
                       __clc_vec3_uint64_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec4_uint64_t __local *,
                       __clc_vec4_uint64_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec4_uint64_t __global *,
                       __clc_vec4_uint64_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec8_uint64_t __local *,
                       __clc_vec8_uint64_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec8_uint64_t __global *,
                       __clc_vec8_uint64_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec16_uint64_t __local *,
                       __clc_vec16_uint64_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec16_uint64_t __global *,
                       __clc_vec16_uint64_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t, __clc_fp32_t __local *, __clc_fp32_t const __global *,
    __clc_size_t, __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t, __clc_fp32_t __global *, __clc_fp32_t const __local *,
    __clc_size_t, __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec2_fp32_t __local *,
                       __clc_vec2_fp32_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec2_fp32_t __global *,
                       __clc_vec2_fp32_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec3_fp32_t __local *,
                       __clc_vec3_fp32_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec3_fp32_t __global *,
                       __clc_vec3_fp32_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec4_fp32_t __local *,
                       __clc_vec4_fp32_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec4_fp32_t __global *,
                       __clc_vec4_fp32_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec8_fp32_t __local *,
                       __clc_vec8_fp32_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec8_fp32_t __global *,
                       __clc_vec8_fp32_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec16_fp32_t __local *,
                       __clc_vec16_fp32_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec16_fp32_t __global *,
                       __clc_vec16_fp32_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t, __clc_fp64_t __local *, __clc_fp64_t const __global *,
    __clc_size_t, __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t, __clc_fp64_t __global *, __clc_fp64_t const __local *,
    __clc_size_t, __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec2_fp64_t __local *,
                       __clc_vec2_fp64_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec2_fp64_t __global *,
                       __clc_vec2_fp64_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec3_fp64_t __local *,
                       __clc_vec3_fp64_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec3_fp64_t __global *,
                       __clc_vec3_fp64_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec4_fp64_t __local *,
                       __clc_vec4_fp64_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec4_fp64_t __global *,
                       __clc_vec4_fp64_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec8_fp64_t __local *,
                       __clc_vec8_fp64_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec8_fp64_t __global *,
                       __clc_vec8_fp64_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec16_fp64_t __local *,
                       __clc_vec16_fp64_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec16_fp64_t __global *,
                       __clc_vec16_fp64_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t, __clc_fp16_t __local *, __clc_fp16_t const __global *,
    __clc_size_t, __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t __spirv_GroupAsyncCopy(
    __clc_int32_t, __clc_fp16_t __global *, __clc_fp16_t const __local *,
    __clc_size_t, __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec2_fp16_t __local *,
                       __clc_vec2_fp16_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec2_fp16_t __global *,
                       __clc_vec2_fp16_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec3_fp16_t __local *,
                       __clc_vec3_fp16_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec3_fp16_t __global *,
                       __clc_vec3_fp16_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec4_fp16_t __local *,
                       __clc_vec4_fp16_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec4_fp16_t __global *,
                       __clc_vec4_fp16_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec8_fp16_t __local *,
                       __clc_vec8_fp16_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec8_fp16_t __global *,
                       __clc_vec8_fp16_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec16_fp16_t __local *,
                       __clc_vec16_fp16_t const __global *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT __clc_event_t
__spirv_GroupAsyncCopy(__clc_int32_t, __clc_vec16_fp16_t __global *,
                       __clc_vec16_fp16_t const __local *, __clc_size_t,
                       __clc_size_t, __clc_event_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONVERGENT void
__spirv_GroupWaitEvents(__clc_int32_t, __clc_int32_t, __clc_event_t *);

_CLC_OVERLOAD _CLC_DECL void __spirv_MemoryBarrier(__clc_int32_t,
                                                   __clc_int32_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_acos(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_acos(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_acos(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_acos(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_acos(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_acos(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_acos(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_acos(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_acos(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_acos(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_acos(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_acos(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_acos(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_acos(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_acos(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_acos(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_acos(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_acos(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_acosh(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_acosh(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_acosh(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_acosh(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_acosh(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_acosh(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_acosh(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_acosh(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_acosh(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_acosh(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_acosh(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_acosh(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_acosh(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_acosh(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_acosh(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_acosh(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_acosh(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_acosh(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_acospi(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_acospi(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_acospi(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_acospi(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_acospi(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_acospi(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_acospi(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_acospi(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_acospi(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_acospi(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_acospi(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_acospi(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_acospi(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_acospi(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_acospi(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_acospi(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_acospi(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_acospi(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_asin(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_asin(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_asin(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_asin(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_asin(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_asin(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_asin(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_asin(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_asin(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_asin(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_asin(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_asin(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_asin(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_asin(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_asin(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_asin(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_asin(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_asin(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_asinh(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_asinh(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_asinh(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_asinh(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_asinh(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_asinh(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_asinh(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_asinh(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_asinh(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_asinh(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_asinh(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_asinh(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_asinh(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_asinh(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_asinh(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_asinh(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_asinh(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_asinh(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_asinpi(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_asinpi(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_asinpi(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_asinpi(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_asinpi(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_asinpi(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_asinpi(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_asinpi(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_asinpi(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_asinpi(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_asinpi(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_asinpi(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_asinpi(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_asinpi(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_asinpi(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_asinpi(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_asinpi(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_asinpi(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_atan(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_atan(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_atan(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_atan(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_atan(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_atan(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_atan(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_atan(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_atan(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_atan(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_atan(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_atan(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_atan(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_atan(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_atan(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_atan(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_atan(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_atan(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_atan2(__clc_fp32_t, __clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_atan2(__clc_vec2_fp32_t, __clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_atan2(__clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_atan2(__clc_vec4_fp32_t, __clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_atan2(__clc_vec8_fp32_t, __clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_atan2(__clc_vec16_fp32_t, __clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ocl_atan2(__clc_fp64_t, __clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_atan2(__clc_vec2_fp64_t, __clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_atan2(__clc_vec3_fp64_t, __clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_atan2(__clc_vec4_fp64_t, __clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_atan2(__clc_vec8_fp64_t, __clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_atan2(__clc_vec16_fp64_t, __clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ocl_atan2(__clc_fp16_t, __clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_atan2(__clc_vec2_fp16_t, __clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_atan2(__clc_vec3_fp16_t, __clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_atan2(__clc_vec4_fp16_t, __clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_atan2(__clc_vec8_fp16_t, __clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_atan2(__clc_vec16_fp16_t, __clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_atan2pi(__clc_fp32_t, __clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_atan2pi(__clc_vec2_fp32_t, __clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_atan2pi(__clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_atan2pi(__clc_vec4_fp32_t, __clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_atan2pi(__clc_vec8_fp32_t, __clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_atan2pi(__clc_vec16_fp32_t, __clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ocl_atan2pi(__clc_fp64_t, __clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_atan2pi(__clc_vec2_fp64_t, __clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_atan2pi(__clc_vec3_fp64_t, __clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_atan2pi(__clc_vec4_fp64_t, __clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_atan2pi(__clc_vec8_fp64_t, __clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_atan2pi(__clc_vec16_fp64_t, __clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ocl_atan2pi(__clc_fp16_t, __clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_atan2pi(__clc_vec2_fp16_t, __clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_atan2pi(__clc_vec3_fp16_t, __clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_atan2pi(__clc_vec4_fp16_t, __clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_atan2pi(__clc_vec8_fp16_t, __clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_atan2pi(__clc_vec16_fp16_t, __clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_atanh(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_atanh(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_atanh(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_atanh(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_atanh(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_atanh(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_atanh(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_atanh(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_atanh(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_atanh(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_atanh(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_atanh(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_atanh(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_atanh(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_atanh(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_atanh(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_atanh(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_atanh(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_atanpi(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_atanpi(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_atanpi(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_atanpi(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_atanpi(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_atanpi(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_atanpi(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_atanpi(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_atanpi(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_atanpi(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_atanpi(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_atanpi(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_atanpi(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_atanpi(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_atanpi(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_atanpi(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_atanpi(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_atanpi(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_char_t
    __spirv_ocl_bitselect(__clc_char_t, __clc_char_t, __clc_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_char_t __spirv_ocl_bitselect(
    __clc_vec2_char_t, __clc_vec2_char_t, __clc_vec2_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_char_t __spirv_ocl_bitselect(
    __clc_vec3_char_t, __clc_vec3_char_t, __clc_vec3_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_char_t __spirv_ocl_bitselect(
    __clc_vec4_char_t, __clc_vec4_char_t, __clc_vec4_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_char_t __spirv_ocl_bitselect(
    __clc_vec8_char_t, __clc_vec8_char_t, __clc_vec8_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_char_t __spirv_ocl_bitselect(
    __clc_vec16_char_t, __clc_vec16_char_t, __clc_vec16_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ocl_bitselect(__clc_int8_t, __clc_int8_t, __clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t __spirv_ocl_bitselect(
    __clc_vec2_int8_t, __clc_vec2_int8_t, __clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t __spirv_ocl_bitselect(
    __clc_vec3_int8_t, __clc_vec3_int8_t, __clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t __spirv_ocl_bitselect(
    __clc_vec4_int8_t, __clc_vec4_int8_t, __clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t __spirv_ocl_bitselect(
    __clc_vec8_int8_t, __clc_vec8_int8_t, __clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t __spirv_ocl_bitselect(
    __clc_vec16_int8_t, __clc_vec16_int8_t, __clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ocl_bitselect(__clc_int16_t, __clc_int16_t, __clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t __spirv_ocl_bitselect(
    __clc_vec2_int16_t, __clc_vec2_int16_t, __clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t __spirv_ocl_bitselect(
    __clc_vec3_int16_t, __clc_vec3_int16_t, __clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t __spirv_ocl_bitselect(
    __clc_vec4_int16_t, __clc_vec4_int16_t, __clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t __spirv_ocl_bitselect(
    __clc_vec8_int16_t, __clc_vec8_int16_t, __clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t __spirv_ocl_bitselect(
    __clc_vec16_int16_t, __clc_vec16_int16_t, __clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ocl_bitselect(__clc_int32_t, __clc_int32_t, __clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t __spirv_ocl_bitselect(
    __clc_vec2_int32_t, __clc_vec2_int32_t, __clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t __spirv_ocl_bitselect(
    __clc_vec3_int32_t, __clc_vec3_int32_t, __clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t __spirv_ocl_bitselect(
    __clc_vec4_int32_t, __clc_vec4_int32_t, __clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t __spirv_ocl_bitselect(
    __clc_vec8_int32_t, __clc_vec8_int32_t, __clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t __spirv_ocl_bitselect(
    __clc_vec16_int32_t, __clc_vec16_int32_t, __clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ocl_bitselect(__clc_int64_t, __clc_int64_t, __clc_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t __spirv_ocl_bitselect(
    __clc_vec2_int64_t, __clc_vec2_int64_t, __clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t __spirv_ocl_bitselect(
    __clc_vec3_int64_t, __clc_vec3_int64_t, __clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t __spirv_ocl_bitselect(
    __clc_vec4_int64_t, __clc_vec4_int64_t, __clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t __spirv_ocl_bitselect(
    __clc_vec8_int64_t, __clc_vec8_int64_t, __clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t __spirv_ocl_bitselect(
    __clc_vec16_int64_t, __clc_vec16_int64_t, __clc_vec16_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ocl_bitselect(__clc_uint8_t, __clc_uint8_t, __clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t __spirv_ocl_bitselect(
    __clc_vec2_uint8_t, __clc_vec2_uint8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t __spirv_ocl_bitselect(
    __clc_vec3_uint8_t, __clc_vec3_uint8_t, __clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t __spirv_ocl_bitselect(
    __clc_vec4_uint8_t, __clc_vec4_uint8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t __spirv_ocl_bitselect(
    __clc_vec8_uint8_t, __clc_vec8_uint8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t __spirv_ocl_bitselect(
    __clc_vec16_uint8_t, __clc_vec16_uint8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ocl_bitselect(__clc_uint16_t, __clc_uint16_t, __clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t __spirv_ocl_bitselect(
    __clc_vec2_uint16_t, __clc_vec2_uint16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t __spirv_ocl_bitselect(
    __clc_vec3_uint16_t, __clc_vec3_uint16_t, __clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t __spirv_ocl_bitselect(
    __clc_vec4_uint16_t, __clc_vec4_uint16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t __spirv_ocl_bitselect(
    __clc_vec8_uint16_t, __clc_vec8_uint16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t __spirv_ocl_bitselect(
    __clc_vec16_uint16_t, __clc_vec16_uint16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ocl_bitselect(__clc_uint32_t, __clc_uint32_t, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t __spirv_ocl_bitselect(
    __clc_vec2_uint32_t, __clc_vec2_uint32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t __spirv_ocl_bitselect(
    __clc_vec3_uint32_t, __clc_vec3_uint32_t, __clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t __spirv_ocl_bitselect(
    __clc_vec4_uint32_t, __clc_vec4_uint32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t __spirv_ocl_bitselect(
    __clc_vec8_uint32_t, __clc_vec8_uint32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t __spirv_ocl_bitselect(
    __clc_vec16_uint32_t, __clc_vec16_uint32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ocl_bitselect(__clc_uint64_t, __clc_uint64_t, __clc_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t __spirv_ocl_bitselect(
    __clc_vec2_uint64_t, __clc_vec2_uint64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t __spirv_ocl_bitselect(
    __clc_vec3_uint64_t, __clc_vec3_uint64_t, __clc_vec3_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t __spirv_ocl_bitselect(
    __clc_vec4_uint64_t, __clc_vec4_uint64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t __spirv_ocl_bitselect(
    __clc_vec8_uint64_t, __clc_vec8_uint64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t __spirv_ocl_bitselect(
    __clc_vec16_uint64_t, __clc_vec16_uint64_t, __clc_vec16_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_bitselect(__clc_fp32_t, __clc_fp32_t, __clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t __spirv_ocl_bitselect(
    __clc_vec2_fp32_t, __clc_vec2_fp32_t, __clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t __spirv_ocl_bitselect(
    __clc_vec3_fp32_t, __clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t __spirv_ocl_bitselect(
    __clc_vec4_fp32_t, __clc_vec4_fp32_t, __clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t __spirv_ocl_bitselect(
    __clc_vec8_fp32_t, __clc_vec8_fp32_t, __clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t __spirv_ocl_bitselect(
    __clc_vec16_fp32_t, __clc_vec16_fp32_t, __clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ocl_bitselect(__clc_fp64_t, __clc_fp64_t, __clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t __spirv_ocl_bitselect(
    __clc_vec2_fp64_t, __clc_vec2_fp64_t, __clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t __spirv_ocl_bitselect(
    __clc_vec3_fp64_t, __clc_vec3_fp64_t, __clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t __spirv_ocl_bitselect(
    __clc_vec4_fp64_t, __clc_vec4_fp64_t, __clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t __spirv_ocl_bitselect(
    __clc_vec8_fp64_t, __clc_vec8_fp64_t, __clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t __spirv_ocl_bitselect(
    __clc_vec16_fp64_t, __clc_vec16_fp64_t, __clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ocl_bitselect(__clc_fp16_t, __clc_fp16_t, __clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t __spirv_ocl_bitselect(
    __clc_vec2_fp16_t, __clc_vec2_fp16_t, __clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t __spirv_ocl_bitselect(
    __clc_vec3_fp16_t, __clc_vec3_fp16_t, __clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t __spirv_ocl_bitselect(
    __clc_vec4_fp16_t, __clc_vec4_fp16_t, __clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t __spirv_ocl_bitselect(
    __clc_vec8_fp16_t, __clc_vec8_fp16_t, __clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t __spirv_ocl_bitselect(
    __clc_vec16_fp16_t, __clc_vec16_fp16_t, __clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_cbrt(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_cbrt(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_cbrt(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_cbrt(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_cbrt(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_cbrt(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_cbrt(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_cbrt(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_cbrt(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_cbrt(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_cbrt(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_cbrt(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_cbrt(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_cbrt(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_cbrt(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_cbrt(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_cbrt(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_cbrt(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_ceil(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_ceil(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_ceil(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_ceil(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_ceil(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_ceil(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_ceil(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_ceil(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_ceil(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_ceil(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_ceil(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_ceil(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_ceil(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_ceil(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_ceil(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_ceil(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_ceil(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_ceil(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_char_t __spirv_ocl_clz(__clc_char_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec2_char_t __spirv_ocl_clz(__clc_vec2_char_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec3_char_t __spirv_ocl_clz(__clc_vec3_char_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec4_char_t __spirv_ocl_clz(__clc_vec4_char_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec8_char_t __spirv_ocl_clz(__clc_vec8_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_char_t
    __spirv_ocl_clz(__clc_vec16_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t __spirv_ocl_clz(__clc_int8_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t __spirv_ocl_clz(__clc_vec2_int8_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t __spirv_ocl_clz(__clc_vec3_int8_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t __spirv_ocl_clz(__clc_vec4_int8_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t __spirv_ocl_clz(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ocl_clz(__clc_vec16_int8_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int16_t __spirv_ocl_clz(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ocl_clz(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ocl_clz(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ocl_clz(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ocl_clz(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ocl_clz(__clc_vec16_int16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int32_t __spirv_ocl_clz(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ocl_clz(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ocl_clz(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ocl_clz(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ocl_clz(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ocl_clz(__clc_vec16_int32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int64_t __spirv_ocl_clz(__clc_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ocl_clz(__clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ocl_clz(__clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ocl_clz(__clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ocl_clz(__clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ocl_clz(__clc_vec16_int64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_uint8_t __spirv_ocl_clz(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ocl_clz(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ocl_clz(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ocl_clz(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ocl_clz(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ocl_clz(__clc_vec16_uint8_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_uint16_t __spirv_ocl_clz(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ocl_clz(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ocl_clz(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ocl_clz(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ocl_clz(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ocl_clz(__clc_vec16_uint16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_uint32_t __spirv_ocl_clz(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ocl_clz(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ocl_clz(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ocl_clz(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ocl_clz(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ocl_clz(__clc_vec16_uint32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_uint64_t __spirv_ocl_clz(__clc_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ocl_clz(__clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ocl_clz(__clc_vec3_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ocl_clz(__clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ocl_clz(__clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ocl_clz(__clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_copysign(__clc_fp32_t, __clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_copysign(__clc_vec2_fp32_t, __clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_copysign(__clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_copysign(__clc_vec4_fp32_t, __clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_copysign(__clc_vec8_fp32_t, __clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_copysign(__clc_vec16_fp32_t, __clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ocl_copysign(__clc_fp64_t, __clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_copysign(__clc_vec2_fp64_t, __clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_copysign(__clc_vec3_fp64_t, __clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_copysign(__clc_vec4_fp64_t, __clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_copysign(__clc_vec8_fp64_t, __clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_copysign(__clc_vec16_fp64_t, __clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ocl_copysign(__clc_fp16_t, __clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_copysign(__clc_vec2_fp16_t, __clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_copysign(__clc_vec3_fp16_t, __clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_copysign(__clc_vec4_fp16_t, __clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_copysign(__clc_vec8_fp16_t, __clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_copysign(__clc_vec16_fp16_t, __clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_cos(__clc_fp32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t __spirv_ocl_cos(__clc_vec2_fp32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t __spirv_ocl_cos(__clc_vec3_fp32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t __spirv_ocl_cos(__clc_vec4_fp32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t __spirv_ocl_cos(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_cos(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_cos(__clc_fp64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t __spirv_ocl_cos(__clc_vec2_fp64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t __spirv_ocl_cos(__clc_vec3_fp64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t __spirv_ocl_cos(__clc_vec4_fp64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t __spirv_ocl_cos(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_cos(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_cos(__clc_fp16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t __spirv_ocl_cos(__clc_vec2_fp16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t __spirv_ocl_cos(__clc_vec3_fp16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t __spirv_ocl_cos(__clc_vec4_fp16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t __spirv_ocl_cos(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_cos(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_cosh(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_cosh(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_cosh(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_cosh(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_cosh(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_cosh(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_cosh(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_cosh(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_cosh(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_cosh(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_cosh(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_cosh(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_cosh(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_cosh(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_cosh(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_cosh(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_cosh(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_cosh(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_cospi(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_cospi(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_cospi(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_cospi(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_cospi(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_cospi(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_cospi(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_cospi(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_cospi(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_cospi(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_cospi(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_cospi(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_cospi(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_cospi(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_cospi(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_cospi(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_cospi(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_cospi(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_cross(__clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_cross(__clc_vec4_fp32_t, __clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_cross(__clc_vec3_fp64_t, __clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_cross(__clc_vec4_fp64_t, __clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_cross(__clc_vec3_fp16_t, __clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_cross(__clc_vec4_fp16_t, __clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_char_t __spirv_ocl_ctz(__clc_char_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec2_char_t __spirv_ocl_ctz(__clc_vec2_char_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec3_char_t __spirv_ocl_ctz(__clc_vec3_char_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec4_char_t __spirv_ocl_ctz(__clc_vec4_char_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec8_char_t __spirv_ocl_ctz(__clc_vec8_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_char_t
    __spirv_ocl_ctz(__clc_vec16_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t __spirv_ocl_ctz(__clc_int8_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t __spirv_ocl_ctz(__clc_vec2_int8_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t __spirv_ocl_ctz(__clc_vec3_int8_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t __spirv_ocl_ctz(__clc_vec4_int8_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t __spirv_ocl_ctz(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ocl_ctz(__clc_vec16_int8_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int16_t __spirv_ocl_ctz(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ocl_ctz(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ocl_ctz(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ocl_ctz(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ocl_ctz(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ocl_ctz(__clc_vec16_int16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int32_t __spirv_ocl_ctz(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ocl_ctz(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ocl_ctz(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ocl_ctz(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ocl_ctz(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ocl_ctz(__clc_vec16_int32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int64_t __spirv_ocl_ctz(__clc_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ocl_ctz(__clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ocl_ctz(__clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ocl_ctz(__clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ocl_ctz(__clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ocl_ctz(__clc_vec16_int64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_uint8_t __spirv_ocl_ctz(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ocl_ctz(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ocl_ctz(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ocl_ctz(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ocl_ctz(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ocl_ctz(__clc_vec16_uint8_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_uint16_t __spirv_ocl_ctz(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ocl_ctz(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ocl_ctz(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ocl_ctz(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ocl_ctz(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ocl_ctz(__clc_vec16_uint16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_uint32_t __spirv_ocl_ctz(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ocl_ctz(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ocl_ctz(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ocl_ctz(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ocl_ctz(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ocl_ctz(__clc_vec16_uint32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_uint64_t __spirv_ocl_ctz(__clc_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ocl_ctz(__clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ocl_ctz(__clc_vec3_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ocl_ctz(__clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ocl_ctz(__clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ocl_ctz(__clc_vec16_uint64_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_degrees(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_degrees(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_degrees(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_degrees(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_degrees(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_degrees(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_degrees(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_degrees(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_degrees(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_degrees(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_degrees(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_degrees(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_degrees(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_degrees(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_degrees(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_degrees(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_degrees(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_degrees(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_distance(__clc_fp32_t, __clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_distance(__clc_vec2_fp32_t, __clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_distance(__clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_distance(__clc_vec4_fp32_t, __clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ocl_distance(__clc_fp64_t, __clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ocl_distance(__clc_vec2_fp64_t, __clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ocl_distance(__clc_vec3_fp64_t, __clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ocl_distance(__clc_vec4_fp64_t, __clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ocl_distance(__clc_fp16_t, __clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ocl_distance(__clc_vec2_fp16_t, __clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ocl_distance(__clc_vec3_fp16_t, __clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ocl_distance(__clc_vec4_fp16_t, __clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_erf(__clc_fp32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t __spirv_ocl_erf(__clc_vec2_fp32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t __spirv_ocl_erf(__clc_vec3_fp32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t __spirv_ocl_erf(__clc_vec4_fp32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t __spirv_ocl_erf(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_erf(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_erf(__clc_fp64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t __spirv_ocl_erf(__clc_vec2_fp64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t __spirv_ocl_erf(__clc_vec3_fp64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t __spirv_ocl_erf(__clc_vec4_fp64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t __spirv_ocl_erf(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_erf(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_erf(__clc_fp16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t __spirv_ocl_erf(__clc_vec2_fp16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t __spirv_ocl_erf(__clc_vec3_fp16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t __spirv_ocl_erf(__clc_vec4_fp16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t __spirv_ocl_erf(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_erf(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_erfc(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_erfc(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_erfc(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_erfc(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_erfc(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_erfc(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_erfc(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_erfc(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_erfc(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_erfc(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_erfc(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_erfc(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_erfc(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_erfc(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_erfc(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_erfc(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_erfc(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_erfc(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_exp(__clc_fp32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t __spirv_ocl_exp(__clc_vec2_fp32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t __spirv_ocl_exp(__clc_vec3_fp32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t __spirv_ocl_exp(__clc_vec4_fp32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t __spirv_ocl_exp(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_exp(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_exp(__clc_fp64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t __spirv_ocl_exp(__clc_vec2_fp64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t __spirv_ocl_exp(__clc_vec3_fp64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t __spirv_ocl_exp(__clc_vec4_fp64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t __spirv_ocl_exp(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_exp(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_exp(__clc_fp16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t __spirv_ocl_exp(__clc_vec2_fp16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t __spirv_ocl_exp(__clc_vec3_fp16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t __spirv_ocl_exp(__clc_vec4_fp16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t __spirv_ocl_exp(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_exp(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_exp10(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_exp10(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_exp10(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_exp10(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_exp10(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_exp10(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_exp10(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_exp10(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_exp10(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_exp10(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_exp10(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_exp10(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_exp10(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_exp10(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_exp10(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_exp10(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_exp10(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_exp10(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_exp2(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_exp2(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_exp2(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_exp2(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_exp2(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_exp2(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_exp2(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_exp2(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_exp2(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_exp2(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_exp2(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_exp2(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_exp2(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_exp2(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_exp2(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_exp2(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_exp2(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_exp2(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_expm1(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_expm1(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_expm1(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_expm1(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_expm1(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_expm1(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_expm1(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_expm1(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_expm1(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_expm1(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_expm1(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_expm1(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_expm1(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_expm1(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_expm1(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_expm1(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_expm1(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_expm1(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_fabs(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_fabs(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_fabs(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_fabs(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_fabs(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_fabs(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_fabs(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_fabs(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_fabs(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_fabs(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_fabs(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_fabs(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_fabs(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_fabs(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_fabs(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_fabs(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_fabs(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_fabs(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_fast_distance(__clc_fp32_t, __clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_fast_distance(__clc_vec2_fp32_t, __clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_fast_distance(__clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_fast_distance(__clc_vec4_fp32_t, __clc_vec4_fp32_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_fast_length(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_fast_length(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_fast_length(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_fast_length(__clc_vec4_fp32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_fast_normalize(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_fast_normalize(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_fast_normalize(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_fast_normalize(__clc_vec4_fp32_t);

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

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_fdim(__clc_fp32_t, __clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_fdim(__clc_vec2_fp32_t, __clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_fdim(__clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_fdim(__clc_vec4_fp32_t, __clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_fdim(__clc_vec8_fp32_t, __clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_fdim(__clc_vec16_fp32_t, __clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ocl_fdim(__clc_fp64_t, __clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_fdim(__clc_vec2_fp64_t, __clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_fdim(__clc_vec3_fp64_t, __clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_fdim(__clc_vec4_fp64_t, __clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_fdim(__clc_vec8_fp64_t, __clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_fdim(__clc_vec16_fp64_t, __clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ocl_fdim(__clc_fp16_t, __clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_fdim(__clc_vec2_fp16_t, __clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_fdim(__clc_vec3_fp16_t, __clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_fdim(__clc_vec4_fp16_t, __clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_fdim(__clc_vec8_fp16_t, __clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_fdim(__clc_vec16_fp16_t, __clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_floor(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_floor(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_floor(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_floor(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_floor(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_floor(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_floor(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_floor(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_floor(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_floor(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_floor(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_floor(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_floor(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_floor(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_floor(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_floor(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_floor(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_floor(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_fma(__clc_fp32_t,
                                                                  __clc_fp32_t,
                                                                  __clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_fma(__clc_vec2_fp32_t, __clc_vec2_fp32_t, __clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_fma(__clc_vec3_fp32_t, __clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_fma(__clc_vec4_fp32_t, __clc_vec4_fp32_t, __clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_fma(__clc_vec8_fp32_t, __clc_vec8_fp32_t, __clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_fma(__clc_vec16_fp32_t, __clc_vec16_fp32_t, __clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_fma(__clc_fp64_t,
                                                                  __clc_fp64_t,
                                                                  __clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_fma(__clc_vec2_fp64_t, __clc_vec2_fp64_t, __clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_fma(__clc_vec3_fp64_t, __clc_vec3_fp64_t, __clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_fma(__clc_vec4_fp64_t, __clc_vec4_fp64_t, __clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_fma(__clc_vec8_fp64_t, __clc_vec8_fp64_t, __clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_fma(__clc_vec16_fp64_t, __clc_vec16_fp64_t, __clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_fma(__clc_fp16_t,
                                                                  __clc_fp16_t,
                                                                  __clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_fma(__clc_vec2_fp16_t, __clc_vec2_fp16_t, __clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_fma(__clc_vec3_fp16_t, __clc_vec3_fp16_t, __clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_fma(__clc_vec4_fp16_t, __clc_vec4_fp16_t, __clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_fma(__clc_vec8_fp16_t, __clc_vec8_fp16_t, __clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_fma(__clc_vec16_fp16_t, __clc_vec16_fp16_t, __clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_fmax(__clc_fp32_t, __clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_fmax(__clc_vec2_fp32_t, __clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_fmax(__clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_fmax(__clc_vec4_fp32_t, __clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_fmax(__clc_vec8_fp32_t, __clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_fmax(__clc_vec16_fp32_t, __clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ocl_fmax(__clc_fp64_t, __clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_fmax(__clc_vec2_fp64_t, __clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_fmax(__clc_vec3_fp64_t, __clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_fmax(__clc_vec4_fp64_t, __clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_fmax(__clc_vec8_fp64_t, __clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_fmax(__clc_vec16_fp64_t, __clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ocl_fmax(__clc_fp16_t, __clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_fmax(__clc_vec2_fp16_t, __clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_fmax(__clc_vec3_fp16_t, __clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_fmax(__clc_vec4_fp16_t, __clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_fmax(__clc_vec8_fp16_t, __clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_fmax(__clc_vec16_fp16_t, __clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_fmax_common(__clc_fp32_t, __clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_fmax_common(__clc_vec2_fp32_t, __clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_fmax_common(__clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_fmax_common(__clc_vec4_fp32_t, __clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_fmax_common(__clc_vec8_fp32_t, __clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_fmax_common(__clc_vec16_fp32_t, __clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ocl_fmax_common(__clc_fp64_t, __clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_fmax_common(__clc_vec2_fp64_t, __clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_fmax_common(__clc_vec3_fp64_t, __clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_fmax_common(__clc_vec4_fp64_t, __clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_fmax_common(__clc_vec8_fp64_t, __clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_fmax_common(__clc_vec16_fp64_t, __clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ocl_fmax_common(__clc_fp16_t, __clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_fmax_common(__clc_vec2_fp16_t, __clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_fmax_common(__clc_vec3_fp16_t, __clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_fmax_common(__clc_vec4_fp16_t, __clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_fmax_common(__clc_vec8_fp16_t, __clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_fmax_common(__clc_vec16_fp16_t, __clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_fmin(__clc_fp32_t, __clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_fmin(__clc_vec2_fp32_t, __clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_fmin(__clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_fmin(__clc_vec4_fp32_t, __clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_fmin(__clc_vec8_fp32_t, __clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_fmin(__clc_vec16_fp32_t, __clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ocl_fmin(__clc_fp64_t, __clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_fmin(__clc_vec2_fp64_t, __clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_fmin(__clc_vec3_fp64_t, __clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_fmin(__clc_vec4_fp64_t, __clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_fmin(__clc_vec8_fp64_t, __clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_fmin(__clc_vec16_fp64_t, __clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ocl_fmin(__clc_fp16_t, __clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_fmin(__clc_vec2_fp16_t, __clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_fmin(__clc_vec3_fp16_t, __clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_fmin(__clc_vec4_fp16_t, __clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_fmin(__clc_vec8_fp16_t, __clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_fmin(__clc_vec16_fp16_t, __clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_fmin_common(__clc_fp32_t, __clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_fmin_common(__clc_vec2_fp32_t, __clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_fmin_common(__clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_fmin_common(__clc_vec4_fp32_t, __clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_fmin_common(__clc_vec8_fp32_t, __clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_fmin_common(__clc_vec16_fp32_t, __clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ocl_fmin_common(__clc_fp64_t, __clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_fmin_common(__clc_vec2_fp64_t, __clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_fmin_common(__clc_vec3_fp64_t, __clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_fmin_common(__clc_vec4_fp64_t, __clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_fmin_common(__clc_vec8_fp64_t, __clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_fmin_common(__clc_vec16_fp64_t, __clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ocl_fmin_common(__clc_fp16_t, __clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_fmin_common(__clc_vec2_fp16_t, __clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_fmin_common(__clc_vec3_fp16_t, __clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_fmin_common(__clc_vec4_fp16_t, __clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_fmin_common(__clc_vec8_fp16_t, __clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_fmin_common(__clc_vec16_fp16_t, __clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_fmod(__clc_fp32_t, __clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_fmod(__clc_vec2_fp32_t, __clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_fmod(__clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_fmod(__clc_vec4_fp32_t, __clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_fmod(__clc_vec8_fp32_t, __clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_fmod(__clc_vec16_fp32_t, __clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ocl_fmod(__clc_fp64_t, __clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_fmod(__clc_vec2_fp64_t, __clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_fmod(__clc_vec3_fp64_t, __clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_fmod(__clc_vec4_fp64_t, __clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_fmod(__clc_vec8_fp64_t, __clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_fmod(__clc_vec16_fp64_t, __clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ocl_fmod(__clc_fp16_t, __clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_fmod(__clc_vec2_fp16_t, __clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_fmod(__clc_vec3_fp16_t, __clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_fmod(__clc_vec4_fp16_t, __clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_fmod(__clc_vec8_fp16_t, __clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_fmod(__clc_vec16_fp16_t, __clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL __clc_fp32_t
__spirv_ocl_fract(__clc_fp32_t, __clc_fp32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_fp32_t __spirv_ocl_fract(__clc_fp32_t,
                                                       __clc_fp32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_fp32_t __spirv_ocl_fract(__clc_fp32_t,
                                                       __clc_fp32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_fract(__clc_vec2_fp32_t, __clc_vec2_fp32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_fract(__clc_vec2_fp32_t, __clc_vec2_fp32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_fract(__clc_vec2_fp32_t, __clc_vec2_fp32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_fract(__clc_vec3_fp32_t, __clc_vec3_fp32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_fract(__clc_vec3_fp32_t, __clc_vec3_fp32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_fract(__clc_vec3_fp32_t, __clc_vec3_fp32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_fract(__clc_vec4_fp32_t, __clc_vec4_fp32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_fract(__clc_vec4_fp32_t, __clc_vec4_fp32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_fract(__clc_vec4_fp32_t, __clc_vec4_fp32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_fract(__clc_vec8_fp32_t, __clc_vec8_fp32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_fract(__clc_vec8_fp32_t, __clc_vec8_fp32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_fract(__clc_vec8_fp32_t, __clc_vec8_fp32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_fract(__clc_vec16_fp32_t, __clc_vec16_fp32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_fract(__clc_vec16_fp32_t, __clc_vec16_fp32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_fract(__clc_vec16_fp32_t, __clc_vec16_fp32_t __global *);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL __clc_fp64_t
__spirv_ocl_fract(__clc_fp64_t, __clc_fp64_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_fp64_t __spirv_ocl_fract(__clc_fp64_t,
                                                       __clc_fp64_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_fp64_t __spirv_ocl_fract(__clc_fp64_t,
                                                       __clc_fp64_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_fract(__clc_vec2_fp64_t, __clc_vec2_fp64_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_fract(__clc_vec2_fp64_t, __clc_vec2_fp64_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_fract(__clc_vec2_fp64_t, __clc_vec2_fp64_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_fract(__clc_vec3_fp64_t, __clc_vec3_fp64_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_fract(__clc_vec3_fp64_t, __clc_vec3_fp64_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_fract(__clc_vec3_fp64_t, __clc_vec3_fp64_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_fract(__clc_vec4_fp64_t, __clc_vec4_fp64_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_fract(__clc_vec4_fp64_t, __clc_vec4_fp64_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_fract(__clc_vec4_fp64_t, __clc_vec4_fp64_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_fract(__clc_vec8_fp64_t, __clc_vec8_fp64_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_fract(__clc_vec8_fp64_t, __clc_vec8_fp64_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_fract(__clc_vec8_fp64_t, __clc_vec8_fp64_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_fract(__clc_vec16_fp64_t, __clc_vec16_fp64_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_fract(__clc_vec16_fp64_t, __clc_vec16_fp64_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_fract(__clc_vec16_fp64_t, __clc_vec16_fp64_t __global *);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL __clc_fp16_t
__spirv_ocl_fract(__clc_fp16_t, __clc_fp16_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_fp16_t __spirv_ocl_fract(__clc_fp16_t,
                                                       __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_fp16_t __spirv_ocl_fract(__clc_fp16_t,
                                                       __clc_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_fract(__clc_vec2_fp16_t, __clc_vec2_fp16_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_fract(__clc_vec2_fp16_t, __clc_vec2_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_fract(__clc_vec2_fp16_t, __clc_vec2_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_fract(__clc_vec3_fp16_t, __clc_vec3_fp16_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_fract(__clc_vec3_fp16_t, __clc_vec3_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_fract(__clc_vec3_fp16_t, __clc_vec3_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_fract(__clc_vec4_fp16_t, __clc_vec4_fp16_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_fract(__clc_vec4_fp16_t, __clc_vec4_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_fract(__clc_vec4_fp16_t, __clc_vec4_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_fract(__clc_vec8_fp16_t, __clc_vec8_fp16_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_fract(__clc_vec8_fp16_t, __clc_vec8_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_fract(__clc_vec8_fp16_t, __clc_vec8_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_fract(__clc_vec16_fp16_t, __clc_vec16_fp16_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_fract(__clc_vec16_fp16_t, __clc_vec16_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_fract(__clc_vec16_fp16_t, __clc_vec16_fp16_t __global *);
#endif

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DECL __clc_fp32_t
__spirv_ocl_fract(__clc_fp32_t, __clc_fp32_t __generic *);

_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_fract(__clc_vec2_fp32_t, __clc_vec2_fp32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_fract(__clc_vec3_fp32_t, __clc_vec3_fp32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_fract(__clc_vec4_fp32_t, __clc_vec4_fp32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_fract(__clc_vec8_fp32_t, __clc_vec8_fp32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_fract(__clc_vec16_fp32_t, __clc_vec16_fp32_t __generic *);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL __clc_fp64_t
__spirv_ocl_fract(__clc_fp64_t, __clc_fp64_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_fract(__clc_vec2_fp64_t, __clc_vec2_fp64_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_fract(__clc_vec3_fp64_t, __clc_vec3_fp64_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_fract(__clc_vec4_fp64_t, __clc_vec4_fp64_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_fract(__clc_vec8_fp64_t, __clc_vec8_fp64_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_fract(__clc_vec16_fp64_t, __clc_vec16_fp64_t __generic *);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL __clc_fp16_t
__spirv_ocl_fract(__clc_fp16_t, __clc_fp16_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_fract(__clc_vec2_fp16_t, __clc_vec2_fp16_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_fract(__clc_vec3_fp16_t, __clc_vec3_fp16_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_fract(__clc_vec4_fp16_t, __clc_vec4_fp16_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_fract(__clc_vec8_fp16_t, __clc_vec8_fp16_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_fract(__clc_vec16_fp16_t, __clc_vec16_fp16_t __generic *);
#endif
#endif

_CLC_OVERLOAD _CLC_DECL __clc_fp32_t
__spirv_ocl_frexp(__clc_fp32_t, __clc_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_fp32_t __spirv_ocl_frexp(__clc_fp32_t,
                                                       __clc_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_fp32_t
__spirv_ocl_frexp(__clc_fp32_t, __clc_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_frexp(__clc_vec2_fp32_t, __clc_vec2_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_frexp(__clc_vec2_fp32_t, __clc_vec2_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_frexp(__clc_vec2_fp32_t, __clc_vec2_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_frexp(__clc_vec3_fp32_t, __clc_vec3_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_frexp(__clc_vec3_fp32_t, __clc_vec3_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_frexp(__clc_vec3_fp32_t, __clc_vec3_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_frexp(__clc_vec4_fp32_t, __clc_vec4_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_frexp(__clc_vec4_fp32_t, __clc_vec4_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_frexp(__clc_vec4_fp32_t, __clc_vec4_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_frexp(__clc_vec8_fp32_t, __clc_vec8_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_frexp(__clc_vec8_fp32_t, __clc_vec8_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_frexp(__clc_vec8_fp32_t, __clc_vec8_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_frexp(__clc_vec16_fp32_t, __clc_vec16_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_frexp(__clc_vec16_fp32_t, __clc_vec16_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_frexp(__clc_vec16_fp32_t, __clc_vec16_int32_t __global *);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL __clc_fp64_t
__spirv_ocl_frexp(__clc_fp64_t, __clc_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_fp64_t __spirv_ocl_frexp(__clc_fp64_t,
                                                       __clc_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_fp64_t
__spirv_ocl_frexp(__clc_fp64_t, __clc_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_frexp(__clc_vec2_fp64_t, __clc_vec2_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_frexp(__clc_vec2_fp64_t, __clc_vec2_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_frexp(__clc_vec2_fp64_t, __clc_vec2_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_frexp(__clc_vec3_fp64_t, __clc_vec3_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_frexp(__clc_vec3_fp64_t, __clc_vec3_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_frexp(__clc_vec3_fp64_t, __clc_vec3_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_frexp(__clc_vec4_fp64_t, __clc_vec4_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_frexp(__clc_vec4_fp64_t, __clc_vec4_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_frexp(__clc_vec4_fp64_t, __clc_vec4_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_frexp(__clc_vec8_fp64_t, __clc_vec8_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_frexp(__clc_vec8_fp64_t, __clc_vec8_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_frexp(__clc_vec8_fp64_t, __clc_vec8_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_frexp(__clc_vec16_fp64_t, __clc_vec16_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_frexp(__clc_vec16_fp64_t, __clc_vec16_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_frexp(__clc_vec16_fp64_t, __clc_vec16_int32_t __global *);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL __clc_fp16_t
__spirv_ocl_frexp(__clc_fp16_t, __clc_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_fp16_t __spirv_ocl_frexp(__clc_fp16_t,
                                                       __clc_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_fp16_t
__spirv_ocl_frexp(__clc_fp16_t, __clc_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_frexp(__clc_vec2_fp16_t, __clc_vec2_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_frexp(__clc_vec2_fp16_t, __clc_vec2_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_frexp(__clc_vec2_fp16_t, __clc_vec2_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_frexp(__clc_vec3_fp16_t, __clc_vec3_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_frexp(__clc_vec3_fp16_t, __clc_vec3_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_frexp(__clc_vec3_fp16_t, __clc_vec3_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_frexp(__clc_vec4_fp16_t, __clc_vec4_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_frexp(__clc_vec4_fp16_t, __clc_vec4_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_frexp(__clc_vec4_fp16_t, __clc_vec4_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_frexp(__clc_vec8_fp16_t, __clc_vec8_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_frexp(__clc_vec8_fp16_t, __clc_vec8_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_frexp(__clc_vec8_fp16_t, __clc_vec8_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_frexp(__clc_vec16_fp16_t, __clc_vec16_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_frexp(__clc_vec16_fp16_t, __clc_vec16_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_frexp(__clc_vec16_fp16_t, __clc_vec16_int32_t __global *);
#endif

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DECL __clc_fp32_t
__spirv_ocl_frexp(__clc_fp32_t, __clc_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_frexp(__clc_vec2_fp32_t, __clc_vec2_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_frexp(__clc_vec3_fp32_t, __clc_vec3_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_frexp(__clc_vec4_fp32_t, __clc_vec4_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_frexp(__clc_vec8_fp32_t, __clc_vec8_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_frexp(__clc_vec16_fp32_t, __clc_vec16_int32_t __generic *);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL __clc_fp64_t
__spirv_ocl_frexp(__clc_fp64_t, __clc_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_frexp(__clc_vec2_fp64_t, __clc_vec2_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_frexp(__clc_vec3_fp64_t, __clc_vec3_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_frexp(__clc_vec4_fp64_t, __clc_vec4_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_frexp(__clc_vec8_fp64_t, __clc_vec8_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_frexp(__clc_vec16_fp64_t, __clc_vec16_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_fp16_t
#endif

#ifdef cl_khr_fp16
__spirv_ocl_frexp(__clc_fp16_t, __clc_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_frexp(__clc_vec2_fp16_t, __clc_vec2_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_frexp(__clc_vec3_fp16_t, __clc_vec3_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_frexp(__clc_vec4_fp16_t, __clc_vec4_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_frexp(__clc_vec8_fp16_t, __clc_vec8_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_frexp(__clc_vec16_fp16_t, __clc_vec16_int32_t __generic *);
#endif
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_half_cos(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_half_cos(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_half_cos(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_half_cos(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_half_cos(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_half_cos(__clc_vec16_fp32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_half_divide(__clc_fp32_t, __clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_half_divide(__clc_vec2_fp32_t, __clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_half_divide(__clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_half_divide(__clc_vec4_fp32_t, __clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_half_divide(__clc_vec8_fp32_t, __clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_half_divide(__clc_vec16_fp32_t, __clc_vec16_fp32_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_half_exp(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_half_exp(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_half_exp(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_half_exp(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_half_exp(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_half_exp(__clc_vec16_fp32_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_half_exp10(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_half_exp10(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_half_exp10(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_half_exp10(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_half_exp10(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_half_exp10(__clc_vec16_fp32_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_half_exp2(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_half_exp2(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_half_exp2(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_half_exp2(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_half_exp2(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_half_exp2(__clc_vec16_fp32_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_half_log(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_half_log(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_half_log(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_half_log(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_half_log(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_half_log(__clc_vec16_fp32_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_half_log10(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_half_log10(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_half_log10(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_half_log10(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_half_log10(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_half_log10(__clc_vec16_fp32_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_half_log2(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_half_log2(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_half_log2(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_half_log2(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_half_log2(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_half_log2(__clc_vec16_fp32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_half_powr(__clc_fp32_t, __clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_half_powr(__clc_vec2_fp32_t, __clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_half_powr(__clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_half_powr(__clc_vec4_fp32_t, __clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_half_powr(__clc_vec8_fp32_t, __clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_half_powr(__clc_vec16_fp32_t, __clc_vec16_fp32_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_half_recip(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_half_recip(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_half_recip(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_half_recip(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_half_recip(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_half_recip(__clc_vec16_fp32_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_half_rsqrt(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_half_rsqrt(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_half_rsqrt(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_half_rsqrt(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_half_rsqrt(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_half_rsqrt(__clc_vec16_fp32_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_half_sin(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_half_sin(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_half_sin(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_half_sin(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_half_sin(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_half_sin(__clc_vec16_fp32_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_half_sqrt(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_half_sqrt(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_half_sqrt(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_half_sqrt(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_half_sqrt(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_half_sqrt(__clc_vec16_fp32_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_half_tan(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_half_tan(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_half_tan(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_half_tan(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_half_tan(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_half_tan(__clc_vec16_fp32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_hypot(__clc_fp32_t, __clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_hypot(__clc_vec2_fp32_t, __clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_hypot(__clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_hypot(__clc_vec4_fp32_t, __clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_hypot(__clc_vec8_fp32_t, __clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_hypot(__clc_vec16_fp32_t, __clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ocl_hypot(__clc_fp64_t, __clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_hypot(__clc_vec2_fp64_t, __clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_hypot(__clc_vec3_fp64_t, __clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_hypot(__clc_vec4_fp64_t, __clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_hypot(__clc_vec8_fp64_t, __clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_hypot(__clc_vec16_fp64_t, __clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ocl_hypot(__clc_fp16_t, __clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_hypot(__clc_vec2_fp16_t, __clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_hypot(__clc_vec3_fp16_t, __clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_hypot(__clc_vec4_fp16_t, __clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_hypot(__clc_vec8_fp16_t, __clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_hypot(__clc_vec16_fp16_t, __clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int32_t __spirv_ocl_ilogb(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ocl_ilogb(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ocl_ilogb(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ocl_ilogb(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ocl_ilogb(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ocl_ilogb(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int32_t __spirv_ocl_ilogb(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ocl_ilogb(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ocl_ilogb(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ocl_ilogb(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ocl_ilogb(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ocl_ilogb(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int32_t __spirv_ocl_ilogb(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ocl_ilogb(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ocl_ilogb(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ocl_ilogb(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ocl_ilogb(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ocl_ilogb(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_ldexp(__clc_fp32_t, __clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_ldexp(__clc_fp32_t, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_ldexp(__clc_vec2_fp32_t, __clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_ldexp(__clc_vec2_fp32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_ldexp(__clc_vec3_fp32_t, __clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_ldexp(__clc_vec3_fp32_t, __clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_ldexp(__clc_vec4_fp32_t, __clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_ldexp(__clc_vec4_fp32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_ldexp(__clc_vec8_fp32_t, __clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_ldexp(__clc_vec8_fp32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_ldexp(__clc_vec16_fp32_t, __clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_ldexp(__clc_vec16_fp32_t, __clc_vec16_uint32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ocl_ldexp(__clc_fp64_t, __clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ocl_ldexp(__clc_fp64_t, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_ldexp(__clc_vec2_fp64_t, __clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_ldexp(__clc_vec2_fp64_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_ldexp(__clc_vec3_fp64_t, __clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_ldexp(__clc_vec3_fp64_t, __clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_ldexp(__clc_vec4_fp64_t, __clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_ldexp(__clc_vec4_fp64_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_ldexp(__clc_vec8_fp64_t, __clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_ldexp(__clc_vec8_fp64_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_ldexp(__clc_vec16_fp64_t, __clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_ldexp(__clc_vec16_fp64_t, __clc_vec16_uint32_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ocl_ldexp(__clc_fp16_t, __clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ocl_ldexp(__clc_fp16_t, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_ldexp(__clc_vec2_fp16_t, __clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_ldexp(__clc_vec2_fp16_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_ldexp(__clc_vec3_fp16_t, __clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_ldexp(__clc_vec3_fp16_t, __clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_ldexp(__clc_vec4_fp16_t, __clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_ldexp(__clc_vec4_fp16_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_ldexp(__clc_vec8_fp16_t, __clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_ldexp(__clc_vec8_fp16_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_ldexp(__clc_vec16_fp16_t, __clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_ldexp(__clc_vec16_fp16_t, __clc_vec16_uint32_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_length(__clc_fp32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_length(__clc_vec2_fp32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_length(__clc_vec3_fp32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_length(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_length(__clc_fp64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_length(__clc_vec2_fp64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_length(__clc_vec3_fp64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_length(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_length(__clc_fp16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_length(__clc_vec2_fp16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_length(__clc_vec3_fp16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_length(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_lgamma(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_lgamma(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_lgamma(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_lgamma(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_lgamma(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_lgamma(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_lgamma(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_lgamma(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_lgamma(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_lgamma(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_lgamma(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_lgamma(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_lgamma(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_lgamma(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_lgamma(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_lgamma(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_lgamma(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_lgamma(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL __clc_fp32_t
__spirv_ocl_lgamma_r(__clc_fp32_t, __clc_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_fp32_t
__spirv_ocl_lgamma_r(__clc_fp32_t, __clc_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_fp32_t
__spirv_ocl_lgamma_r(__clc_fp32_t, __clc_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_lgamma_r(__clc_vec2_fp32_t, __clc_vec2_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_lgamma_r(__clc_vec2_fp32_t, __clc_vec2_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_lgamma_r(__clc_vec2_fp32_t, __clc_vec2_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_lgamma_r(__clc_vec3_fp32_t, __clc_vec3_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_lgamma_r(__clc_vec3_fp32_t, __clc_vec3_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_lgamma_r(__clc_vec3_fp32_t, __clc_vec3_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_lgamma_r(__clc_vec4_fp32_t, __clc_vec4_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_lgamma_r(__clc_vec4_fp32_t, __clc_vec4_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_lgamma_r(__clc_vec4_fp32_t, __clc_vec4_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_lgamma_r(__clc_vec8_fp32_t, __clc_vec8_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_lgamma_r(__clc_vec8_fp32_t, __clc_vec8_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_lgamma_r(__clc_vec8_fp32_t, __clc_vec8_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_lgamma_r(__clc_vec16_fp32_t, __clc_vec16_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_lgamma_r(__clc_vec16_fp32_t, __clc_vec16_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_lgamma_r(__clc_vec16_fp32_t, __clc_vec16_int32_t __global *);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL __clc_fp64_t
__spirv_ocl_lgamma_r(__clc_fp64_t, __clc_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_fp64_t
__spirv_ocl_lgamma_r(__clc_fp64_t, __clc_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_fp64_t
__spirv_ocl_lgamma_r(__clc_fp64_t, __clc_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_lgamma_r(__clc_vec2_fp64_t, __clc_vec2_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_lgamma_r(__clc_vec2_fp64_t, __clc_vec2_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_lgamma_r(__clc_vec2_fp64_t, __clc_vec2_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_lgamma_r(__clc_vec3_fp64_t, __clc_vec3_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_lgamma_r(__clc_vec3_fp64_t, __clc_vec3_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_lgamma_r(__clc_vec3_fp64_t, __clc_vec3_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_lgamma_r(__clc_vec4_fp64_t, __clc_vec4_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_lgamma_r(__clc_vec4_fp64_t, __clc_vec4_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_lgamma_r(__clc_vec4_fp64_t, __clc_vec4_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_lgamma_r(__clc_vec8_fp64_t, __clc_vec8_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_lgamma_r(__clc_vec8_fp64_t, __clc_vec8_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_lgamma_r(__clc_vec8_fp64_t, __clc_vec8_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_lgamma_r(__clc_vec16_fp64_t, __clc_vec16_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_lgamma_r(__clc_vec16_fp64_t, __clc_vec16_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_lgamma_r(__clc_vec16_fp64_t, __clc_vec16_int32_t __global *);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL __clc_fp16_t
__spirv_ocl_lgamma_r(__clc_fp16_t, __clc_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_fp16_t
__spirv_ocl_lgamma_r(__clc_fp16_t, __clc_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_fp16_t
__spirv_ocl_lgamma_r(__clc_fp16_t, __clc_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_lgamma_r(__clc_vec2_fp16_t, __clc_vec2_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_lgamma_r(__clc_vec2_fp16_t, __clc_vec2_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_lgamma_r(__clc_vec2_fp16_t, __clc_vec2_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_lgamma_r(__clc_vec3_fp16_t, __clc_vec3_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_lgamma_r(__clc_vec3_fp16_t, __clc_vec3_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_lgamma_r(__clc_vec3_fp16_t, __clc_vec3_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_lgamma_r(__clc_vec4_fp16_t, __clc_vec4_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_lgamma_r(__clc_vec4_fp16_t, __clc_vec4_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_lgamma_r(__clc_vec4_fp16_t, __clc_vec4_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_lgamma_r(__clc_vec8_fp16_t, __clc_vec8_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_lgamma_r(__clc_vec8_fp16_t, __clc_vec8_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_lgamma_r(__clc_vec8_fp16_t, __clc_vec8_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_lgamma_r(__clc_vec16_fp16_t, __clc_vec16_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_lgamma_r(__clc_vec16_fp16_t, __clc_vec16_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_lgamma_r(__clc_vec16_fp16_t, __clc_vec16_int32_t __global *);
#endif

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DECL __clc_fp32_t
__spirv_ocl_lgamma_r(__clc_fp32_t, __clc_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_lgamma_r(__clc_vec2_fp32_t, __clc_vec2_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_lgamma_r(__clc_vec3_fp32_t, __clc_vec3_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_lgamma_r(__clc_vec4_fp32_t, __clc_vec4_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_lgamma_r(__clc_vec8_fp32_t, __clc_vec8_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_lgamma_r(__clc_vec16_fp32_t, __clc_vec16_int32_t __generic *);
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL __clc_fp64_t
__spirv_ocl_lgamma_r(__clc_fp64_t, __clc_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_lgamma_r(__clc_vec2_fp64_t, __clc_vec2_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_lgamma_r(__clc_vec3_fp64_t, __clc_vec3_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_lgamma_r(__clc_vec4_fp64_t, __clc_vec4_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_lgamma_r(__clc_vec8_fp64_t, __clc_vec8_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_lgamma_r(__clc_vec16_fp64_t, __clc_vec16_int32_t __generic *);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL __clc_fp16_t
__spirv_ocl_lgamma_r(__clc_fp16_t, __clc_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_lgamma_r(__clc_vec2_fp16_t, __clc_vec2_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_lgamma_r(__clc_vec3_fp16_t, __clc_vec3_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_lgamma_r(__clc_vec4_fp16_t, __clc_vec4_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_lgamma_r(__clc_vec8_fp16_t, __clc_vec8_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_lgamma_r(__clc_vec16_fp16_t, __clc_vec16_int32_t __generic *);
#endif
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_log(__clc_fp32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t __spirv_ocl_log(__clc_vec2_fp32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t __spirv_ocl_log(__clc_vec3_fp32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t __spirv_ocl_log(__clc_vec4_fp32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t __spirv_ocl_log(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_log(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_log(__clc_fp64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t __spirv_ocl_log(__clc_vec2_fp64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t __spirv_ocl_log(__clc_vec3_fp64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t __spirv_ocl_log(__clc_vec4_fp64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t __spirv_ocl_log(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_log(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_log(__clc_fp16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t __spirv_ocl_log(__clc_vec2_fp16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t __spirv_ocl_log(__clc_vec3_fp16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t __spirv_ocl_log(__clc_vec4_fp16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t __spirv_ocl_log(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_log(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_log10(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_log10(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_log10(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_log10(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_log10(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_log10(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_log10(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_log10(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_log10(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_log10(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_log10(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_log10(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_log10(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_log10(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_log10(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_log10(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_log10(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_log10(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_log1p(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_log1p(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_log1p(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_log1p(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_log1p(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_log1p(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_log1p(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_log1p(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_log1p(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_log1p(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_log1p(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_log1p(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_log1p(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_log1p(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_log1p(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_log1p(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_log1p(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_log1p(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_log2(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_log2(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_log2(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_log2(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_log2(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_log2(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_log2(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_log2(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_log2(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_log2(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_log2(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_log2(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_log2(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_log2(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_log2(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_log2(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_log2(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_log2(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_logb(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_logb(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_logb(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_logb(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_logb(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_logb(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_logb(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_logb(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_logb(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_logb(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_logb(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_logb(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_logb(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_logb(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_logb(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_logb(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_logb(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_logb(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_mad(__clc_fp32_t,
                                                                  __clc_fp32_t,
                                                                  __clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_mad(__clc_vec2_fp32_t, __clc_vec2_fp32_t, __clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_mad(__clc_vec3_fp32_t, __clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_mad(__clc_vec4_fp32_t, __clc_vec4_fp32_t, __clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_mad(__clc_vec8_fp32_t, __clc_vec8_fp32_t, __clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_mad(__clc_vec16_fp32_t, __clc_vec16_fp32_t, __clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_mad(__clc_fp64_t,
                                                                  __clc_fp64_t,
                                                                  __clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_mad(__clc_vec2_fp64_t, __clc_vec2_fp64_t, __clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_mad(__clc_vec3_fp64_t, __clc_vec3_fp64_t, __clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_mad(__clc_vec4_fp64_t, __clc_vec4_fp64_t, __clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_mad(__clc_vec8_fp64_t, __clc_vec8_fp64_t, __clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_mad(__clc_vec16_fp64_t, __clc_vec16_fp64_t, __clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_mad(__clc_fp16_t,
                                                                  __clc_fp16_t,
                                                                  __clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_mad(__clc_vec2_fp16_t, __clc_vec2_fp16_t, __clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_mad(__clc_vec3_fp16_t, __clc_vec3_fp16_t, __clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_mad(__clc_vec4_fp16_t, __clc_vec4_fp16_t, __clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_mad(__clc_vec8_fp16_t, __clc_vec8_fp16_t, __clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_mad(__clc_vec16_fp16_t, __clc_vec16_fp16_t, __clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_maxmag(__clc_fp32_t, __clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_maxmag(__clc_vec2_fp32_t, __clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_maxmag(__clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_maxmag(__clc_vec4_fp32_t, __clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_maxmag(__clc_vec8_fp32_t, __clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_maxmag(__clc_vec16_fp32_t, __clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ocl_maxmag(__clc_fp64_t, __clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_maxmag(__clc_vec2_fp64_t, __clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_maxmag(__clc_vec3_fp64_t, __clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_maxmag(__clc_vec4_fp64_t, __clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_maxmag(__clc_vec8_fp64_t, __clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_maxmag(__clc_vec16_fp64_t, __clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ocl_maxmag(__clc_fp16_t, __clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_maxmag(__clc_vec2_fp16_t, __clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_maxmag(__clc_vec3_fp16_t, __clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_maxmag(__clc_vec4_fp16_t, __clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_maxmag(__clc_vec8_fp16_t, __clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_maxmag(__clc_vec16_fp16_t, __clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_minmag(__clc_fp32_t, __clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_minmag(__clc_vec2_fp32_t, __clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_minmag(__clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_minmag(__clc_vec4_fp32_t, __clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_minmag(__clc_vec8_fp32_t, __clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_minmag(__clc_vec16_fp32_t, __clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ocl_minmag(__clc_fp64_t, __clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_minmag(__clc_vec2_fp64_t, __clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_minmag(__clc_vec3_fp64_t, __clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_minmag(__clc_vec4_fp64_t, __clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_minmag(__clc_vec8_fp64_t, __clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_minmag(__clc_vec16_fp64_t, __clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ocl_minmag(__clc_fp16_t, __clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_minmag(__clc_vec2_fp16_t, __clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_minmag(__clc_vec3_fp16_t, __clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_minmag(__clc_vec4_fp16_t, __clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_minmag(__clc_vec8_fp16_t, __clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_minmag(__clc_vec16_fp16_t, __clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_mix(__clc_fp32_t,
                                                                  __clc_fp32_t,
                                                                  __clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_mix(__clc_vec2_fp32_t, __clc_vec2_fp32_t, __clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_mix(__clc_vec3_fp32_t, __clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_mix(__clc_vec4_fp32_t, __clc_vec4_fp32_t, __clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_mix(__clc_vec8_fp32_t, __clc_vec8_fp32_t, __clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_mix(__clc_vec16_fp32_t, __clc_vec16_fp32_t, __clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_mix(__clc_fp64_t,
                                                                  __clc_fp64_t,
                                                                  __clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_mix(__clc_vec2_fp64_t, __clc_vec2_fp64_t, __clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_mix(__clc_vec3_fp64_t, __clc_vec3_fp64_t, __clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_mix(__clc_vec4_fp64_t, __clc_vec4_fp64_t, __clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_mix(__clc_vec8_fp64_t, __clc_vec8_fp64_t, __clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_mix(__clc_vec16_fp64_t, __clc_vec16_fp64_t, __clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_mix(__clc_fp16_t,
                                                                  __clc_fp16_t,
                                                                  __clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_mix(__clc_vec2_fp16_t, __clc_vec2_fp16_t, __clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_mix(__clc_vec3_fp16_t, __clc_vec3_fp16_t, __clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_mix(__clc_vec4_fp16_t, __clc_vec4_fp16_t, __clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_mix(__clc_vec8_fp16_t, __clc_vec8_fp16_t, __clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_mix(__clc_vec16_fp16_t, __clc_vec16_fp16_t, __clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL __clc_fp32_t __spirv_ocl_modf(__clc_fp32_t,
                                                      __clc_fp32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_fp32_t __spirv_ocl_modf(__clc_fp32_t,
                                                      __clc_fp32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_fp32_t __spirv_ocl_modf(__clc_fp32_t,
                                                      __clc_fp32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_modf(__clc_vec2_fp32_t, __clc_vec2_fp32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_modf(__clc_vec2_fp32_t, __clc_vec2_fp32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_modf(__clc_vec2_fp32_t, __clc_vec2_fp32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_modf(__clc_vec3_fp32_t, __clc_vec3_fp32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_modf(__clc_vec3_fp32_t, __clc_vec3_fp32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_modf(__clc_vec3_fp32_t, __clc_vec3_fp32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_modf(__clc_vec4_fp32_t, __clc_vec4_fp32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_modf(__clc_vec4_fp32_t, __clc_vec4_fp32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_modf(__clc_vec4_fp32_t, __clc_vec4_fp32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_modf(__clc_vec8_fp32_t, __clc_vec8_fp32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_modf(__clc_vec8_fp32_t, __clc_vec8_fp32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_modf(__clc_vec8_fp32_t, __clc_vec8_fp32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_modf(__clc_vec16_fp32_t, __clc_vec16_fp32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_modf(__clc_vec16_fp32_t, __clc_vec16_fp32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_modf(__clc_vec16_fp32_t, __clc_vec16_fp32_t __global *);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL __clc_fp64_t __spirv_ocl_modf(__clc_fp64_t,
                                                      __clc_fp64_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_fp64_t __spirv_ocl_modf(__clc_fp64_t,
                                                      __clc_fp64_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_fp64_t __spirv_ocl_modf(__clc_fp64_t,
                                                      __clc_fp64_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_modf(__clc_vec2_fp64_t, __clc_vec2_fp64_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_modf(__clc_vec2_fp64_t, __clc_vec2_fp64_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_modf(__clc_vec2_fp64_t, __clc_vec2_fp64_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_modf(__clc_vec3_fp64_t, __clc_vec3_fp64_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_modf(__clc_vec3_fp64_t, __clc_vec3_fp64_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_modf(__clc_vec3_fp64_t, __clc_vec3_fp64_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_modf(__clc_vec4_fp64_t, __clc_vec4_fp64_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_modf(__clc_vec4_fp64_t, __clc_vec4_fp64_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_modf(__clc_vec4_fp64_t, __clc_vec4_fp64_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_modf(__clc_vec8_fp64_t, __clc_vec8_fp64_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_modf(__clc_vec8_fp64_t, __clc_vec8_fp64_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_modf(__clc_vec8_fp64_t, __clc_vec8_fp64_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_modf(__clc_vec16_fp64_t, __clc_vec16_fp64_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_modf(__clc_vec16_fp64_t, __clc_vec16_fp64_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_modf(__clc_vec16_fp64_t, __clc_vec16_fp64_t __global *);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL __clc_fp16_t __spirv_ocl_modf(__clc_fp16_t,
                                                      __clc_fp16_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_fp16_t __spirv_ocl_modf(__clc_fp16_t,
                                                      __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_fp16_t __spirv_ocl_modf(__clc_fp16_t,
                                                      __clc_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_modf(__clc_vec2_fp16_t, __clc_vec2_fp16_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_modf(__clc_vec2_fp16_t, __clc_vec2_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_modf(__clc_vec2_fp16_t, __clc_vec2_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_modf(__clc_vec3_fp16_t, __clc_vec3_fp16_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_modf(__clc_vec3_fp16_t, __clc_vec3_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_modf(__clc_vec3_fp16_t, __clc_vec3_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_modf(__clc_vec4_fp16_t, __clc_vec4_fp16_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_modf(__clc_vec4_fp16_t, __clc_vec4_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_modf(__clc_vec4_fp16_t, __clc_vec4_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_modf(__clc_vec8_fp16_t, __clc_vec8_fp16_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_modf(__clc_vec8_fp16_t, __clc_vec8_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_modf(__clc_vec8_fp16_t, __clc_vec8_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_modf(__clc_vec16_fp16_t, __clc_vec16_fp16_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_modf(__clc_vec16_fp16_t, __clc_vec16_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_modf(__clc_vec16_fp16_t, __clc_vec16_fp16_t __global *);
#endif

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DECL __clc_fp32_t __spirv_ocl_modf(__clc_fp32_t,
                                                      __clc_fp32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_modf(__clc_vec2_fp32_t, __clc_vec2_fp32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_modf(__clc_vec3_fp32_t, __clc_vec3_fp32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_modf(__clc_vec4_fp32_t, __clc_vec4_fp32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_modf(__clc_vec8_fp32_t, __clc_vec8_fp32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_modf(__clc_vec16_fp32_t, __clc_vec16_fp32_t __generic *);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL __clc_fp64_t __spirv_ocl_modf(__clc_fp64_t,
                                                      __clc_fp64_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_modf(__clc_vec2_fp64_t, __clc_vec2_fp64_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_modf(__clc_vec3_fp64_t, __clc_vec3_fp64_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_modf(__clc_vec4_fp64_t, __clc_vec4_fp64_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_modf(__clc_vec8_fp64_t, __clc_vec8_fp64_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_modf(__clc_vec16_fp64_t, __clc_vec16_fp64_t __generic *);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL __clc_fp16_t __spirv_ocl_modf(__clc_fp16_t,
                                                      __clc_fp16_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_modf(__clc_vec2_fp16_t, __clc_vec2_fp16_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_modf(__clc_vec3_fp16_t, __clc_vec3_fp16_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_modf(__clc_vec4_fp16_t, __clc_vec4_fp16_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_modf(__clc_vec8_fp16_t, __clc_vec8_fp16_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_modf(__clc_vec16_fp16_t, __clc_vec16_fp16_t __generic *);
#endif
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_nan(__clc_int32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_nan(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_nan(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_nan(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_nan(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_nan(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_nan(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_nan(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_nan(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_nan(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_nan(__clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_nan(__clc_vec16_uint32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_nan(__clc_int64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_nan(__clc_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_nan(__clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_nan(__clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_nan(__clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_nan(__clc_vec3_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_nan(__clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_nan(__clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_nan(__clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_nan(__clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_nan(__clc_vec16_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_nan(__clc_vec16_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_nan(__clc_int16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_nan(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_nan(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_nan(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_nan(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_nan(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_nan(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_nan(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_nan(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_nan(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_nan(__clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_nan(__clc_vec16_uint16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_native_cos(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_native_cos(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_native_cos(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_native_cos(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_native_cos(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_native_cos(__clc_vec16_fp32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_native_divide(__clc_fp32_t, __clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_native_divide(__clc_vec2_fp32_t, __clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_native_divide(__clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_native_divide(__clc_vec4_fp32_t, __clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_native_divide(__clc_vec8_fp32_t, __clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_native_divide(__clc_vec16_fp32_t, __clc_vec16_fp32_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_native_exp(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_native_exp(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_native_exp(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_native_exp(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_native_exp(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_native_exp(__clc_vec16_fp32_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_native_exp10(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_native_exp10(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_native_exp10(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_native_exp10(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_native_exp10(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_native_exp10(__clc_vec16_fp32_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_native_exp2(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_native_exp2(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_native_exp2(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_native_exp2(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_native_exp2(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_native_exp2(__clc_vec16_fp32_t);

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __clc_native_exp2(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __clc_native_exp2(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __clc_native_exp2(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __clc_native_exp2(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __clc_native_exp2(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __clc_native_exp2(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_native_log(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_native_log(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_native_log(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_native_log(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_native_log(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_native_log(__clc_vec16_fp32_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_native_log10(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_native_log10(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_native_log10(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_native_log10(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_native_log10(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_native_log10(__clc_vec16_fp32_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_native_log2(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_native_log2(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_native_log2(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_native_log2(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_native_log2(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_native_log2(__clc_vec16_fp32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_native_powr(__clc_fp32_t, __clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_native_powr(__clc_vec2_fp32_t, __clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_native_powr(__clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_native_powr(__clc_vec4_fp32_t, __clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_native_powr(__clc_vec8_fp32_t, __clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_native_powr(__clc_vec16_fp32_t, __clc_vec16_fp32_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_native_recip(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_native_recip(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_native_recip(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_native_recip(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_native_recip(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_native_recip(__clc_vec16_fp32_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_native_rsqrt(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_native_rsqrt(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_native_rsqrt(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_native_rsqrt(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_native_rsqrt(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_native_rsqrt(__clc_vec16_fp32_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_native_sin(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_native_sin(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_native_sin(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_native_sin(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_native_sin(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_native_sin(__clc_vec16_fp32_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_native_sqrt(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_native_sqrt(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_native_sqrt(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_native_sqrt(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_native_sqrt(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_native_sqrt(__clc_vec16_fp32_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_native_tan(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_native_tan(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_native_tan(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_native_tan(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_native_tan(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_native_tan(__clc_vec16_fp32_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_nextafter(__clc_fp32_t, __clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_nextafter(__clc_vec2_fp32_t, __clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_nextafter(__clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_nextafter(__clc_vec4_fp32_t, __clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_nextafter(__clc_vec8_fp32_t, __clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_nextafter(__clc_vec16_fp32_t, __clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ocl_nextafter(__clc_fp64_t, __clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_nextafter(__clc_vec2_fp64_t, __clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_nextafter(__clc_vec3_fp64_t, __clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_nextafter(__clc_vec4_fp64_t, __clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_nextafter(__clc_vec8_fp64_t, __clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_nextafter(__clc_vec16_fp64_t, __clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ocl_nextafter(__clc_fp16_t, __clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_nextafter(__clc_vec2_fp16_t, __clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_nextafter(__clc_vec3_fp16_t, __clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_nextafter(__clc_vec4_fp16_t, __clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_nextafter(__clc_vec8_fp16_t, __clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_nextafter(__clc_vec16_fp16_t, __clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_normalize(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_normalize(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_normalize(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_normalize(__clc_vec4_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_normalize(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_normalize(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_normalize(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_normalize(__clc_vec4_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_normalize(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_normalize(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_normalize(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_normalize(__clc_vec4_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_char_t __spirv_ocl_popcount(__clc_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_char_t
    __spirv_ocl_popcount(__clc_vec2_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_char_t
    __spirv_ocl_popcount(__clc_vec3_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_char_t
    __spirv_ocl_popcount(__clc_vec4_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_char_t
    __spirv_ocl_popcount(__clc_vec8_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_char_t
    __spirv_ocl_popcount(__clc_vec16_char_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int8_t __spirv_ocl_popcount(__clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ocl_popcount(__clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ocl_popcount(__clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ocl_popcount(__clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ocl_popcount(__clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ocl_popcount(__clc_vec16_int8_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int16_t __spirv_ocl_popcount(__clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ocl_popcount(__clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ocl_popcount(__clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ocl_popcount(__clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ocl_popcount(__clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ocl_popcount(__clc_vec16_int16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int32_t __spirv_ocl_popcount(__clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ocl_popcount(__clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ocl_popcount(__clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ocl_popcount(__clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ocl_popcount(__clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ocl_popcount(__clc_vec16_int32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_int64_t __spirv_ocl_popcount(__clc_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ocl_popcount(__clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ocl_popcount(__clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ocl_popcount(__clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ocl_popcount(__clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ocl_popcount(__clc_vec16_int64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_uint8_t __spirv_ocl_popcount(__clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ocl_popcount(__clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ocl_popcount(__clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ocl_popcount(__clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ocl_popcount(__clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ocl_popcount(__clc_vec16_uint8_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_uint16_t __spirv_ocl_popcount(__clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ocl_popcount(__clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ocl_popcount(__clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ocl_popcount(__clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ocl_popcount(__clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ocl_popcount(__clc_vec16_uint16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_uint32_t __spirv_ocl_popcount(__clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ocl_popcount(__clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ocl_popcount(__clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ocl_popcount(__clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ocl_popcount(__clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ocl_popcount(__clc_vec16_uint32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_uint64_t __spirv_ocl_popcount(__clc_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ocl_popcount(__clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ocl_popcount(__clc_vec3_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ocl_popcount(__clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ocl_popcount(__clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ocl_popcount(__clc_vec16_uint64_t);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_pow(__clc_fp32_t,
                                                                  __clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_pow(__clc_vec2_fp32_t, __clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_pow(__clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_pow(__clc_vec4_fp32_t, __clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_pow(__clc_vec8_fp32_t, __clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_pow(__clc_vec16_fp32_t, __clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_pow(__clc_fp64_t,
                                                                  __clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_pow(__clc_vec2_fp64_t, __clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_pow(__clc_vec3_fp64_t, __clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_pow(__clc_vec4_fp64_t, __clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_pow(__clc_vec8_fp64_t, __clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_pow(__clc_vec16_fp64_t, __clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_pow(__clc_fp16_t,
                                                                  __clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_pow(__clc_vec2_fp16_t, __clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_pow(__clc_vec3_fp16_t, __clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_pow(__clc_vec4_fp16_t, __clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_pow(__clc_vec8_fp16_t, __clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_pow(__clc_vec16_fp16_t, __clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_pown(__clc_fp32_t, __clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_pown(__clc_vec2_fp32_t, __clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_pown(__clc_vec3_fp32_t, __clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_pown(__clc_vec4_fp32_t, __clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_pown(__clc_vec8_fp32_t, __clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_pown(__clc_vec16_fp32_t, __clc_vec16_int32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ocl_pown(__clc_fp64_t, __clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_pown(__clc_vec2_fp64_t, __clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_pown(__clc_vec3_fp64_t, __clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_pown(__clc_vec4_fp64_t, __clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_pown(__clc_vec8_fp64_t, __clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_pown(__clc_vec16_fp64_t, __clc_vec16_int32_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ocl_pown(__clc_fp16_t, __clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_pown(__clc_vec2_fp16_t, __clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_pown(__clc_vec3_fp16_t, __clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_pown(__clc_vec4_fp16_t, __clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_pown(__clc_vec8_fp16_t, __clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_pown(__clc_vec16_fp16_t, __clc_vec16_int32_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_powr(__clc_fp32_t, __clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_powr(__clc_vec2_fp32_t, __clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_powr(__clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_powr(__clc_vec4_fp32_t, __clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_powr(__clc_vec8_fp32_t, __clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_powr(__clc_vec16_fp32_t, __clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ocl_powr(__clc_fp64_t, __clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_powr(__clc_vec2_fp64_t, __clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_powr(__clc_vec3_fp64_t, __clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_powr(__clc_vec4_fp64_t, __clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_powr(__clc_vec8_fp64_t, __clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_powr(__clc_vec16_fp64_t, __clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ocl_powr(__clc_fp16_t, __clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_powr(__clc_vec2_fp16_t, __clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_powr(__clc_vec3_fp16_t, __clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_powr(__clc_vec4_fp16_t, __clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_powr(__clc_vec8_fp16_t, __clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_powr(__clc_vec16_fp16_t, __clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_prefetch(__clc_char_t const __global *,
                                                  __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec2_char_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec3_char_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec4_char_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec8_char_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec16_char_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_prefetch(__clc_int8_t const __global *,
                                                  __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec2_int8_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec3_int8_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec4_int8_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec8_int8_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec16_int8_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_int16_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec2_int16_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec3_int16_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec4_int16_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec8_int16_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec16_int16_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_int32_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec2_int32_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec3_int32_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec4_int32_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec8_int32_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec16_int32_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_int64_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec2_int64_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec3_int64_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec4_int64_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec8_int64_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec16_int64_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_uint8_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec2_uint8_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec3_uint8_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec4_uint8_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec8_uint8_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec16_uint8_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_uint16_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec2_uint16_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec3_uint16_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec4_uint16_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec8_uint16_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec16_uint16_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_uint32_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec2_uint32_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec3_uint32_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec4_uint32_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec8_uint32_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec16_uint32_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_uint64_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec2_uint64_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec3_uint64_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec4_uint64_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec8_uint64_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec16_uint64_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_prefetch(__clc_fp32_t const __global *,
                                                  __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec2_fp32_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec3_fp32_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec4_fp32_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec8_fp32_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec16_fp32_t const __global *, __clc_size_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_prefetch(__clc_fp64_t const __global *,
                                                  __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec2_fp64_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec3_fp64_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec4_fp64_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec8_fp64_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec16_fp64_t const __global *, __clc_size_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_prefetch(__clc_fp16_t const __global *,
                                                  __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec2_fp16_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec3_fp16_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec4_fp16_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec8_fp16_t const __global *, __clc_size_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_prefetch(__clc_vec16_fp16_t const __global *, __clc_size_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_radians(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_radians(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_radians(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_radians(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_radians(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_radians(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_radians(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_radians(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_radians(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_radians(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_radians(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_radians(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_radians(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_radians(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_radians(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_radians(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_radians(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_radians(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_remainder(__clc_fp32_t, __clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_remainder(__clc_vec2_fp32_t, __clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_remainder(__clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_remainder(__clc_vec4_fp32_t, __clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_remainder(__clc_vec8_fp32_t, __clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_remainder(__clc_vec16_fp32_t, __clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ocl_remainder(__clc_fp64_t, __clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_remainder(__clc_vec2_fp64_t, __clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_remainder(__clc_vec3_fp64_t, __clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_remainder(__clc_vec4_fp64_t, __clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_remainder(__clc_vec8_fp64_t, __clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_remainder(__clc_vec16_fp64_t, __clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ocl_remainder(__clc_fp16_t, __clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_remainder(__clc_vec2_fp16_t, __clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_remainder(__clc_vec3_fp16_t, __clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_remainder(__clc_vec4_fp16_t, __clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_remainder(__clc_vec8_fp16_t, __clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_remainder(__clc_vec16_fp16_t, __clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL __clc_fp32_t
__spirv_ocl_remquo(__clc_fp32_t, __clc_fp32_t, __clc_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_fp32_t
__spirv_ocl_remquo(__clc_fp32_t, __clc_fp32_t, __clc_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_fp32_t
__spirv_ocl_remquo(__clc_fp32_t, __clc_fp32_t, __clc_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t __spirv_ocl_remquo(
    __clc_vec2_fp32_t, __clc_vec2_fp32_t, __clc_vec2_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t __spirv_ocl_remquo(
    __clc_vec2_fp32_t, __clc_vec2_fp32_t, __clc_vec2_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t __spirv_ocl_remquo(
    __clc_vec2_fp32_t, __clc_vec2_fp32_t, __clc_vec2_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t __spirv_ocl_remquo(
    __clc_vec3_fp32_t, __clc_vec3_fp32_t, __clc_vec3_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t __spirv_ocl_remquo(
    __clc_vec3_fp32_t, __clc_vec3_fp32_t, __clc_vec3_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t __spirv_ocl_remquo(
    __clc_vec3_fp32_t, __clc_vec3_fp32_t, __clc_vec3_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t __spirv_ocl_remquo(
    __clc_vec4_fp32_t, __clc_vec4_fp32_t, __clc_vec4_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t __spirv_ocl_remquo(
    __clc_vec4_fp32_t, __clc_vec4_fp32_t, __clc_vec4_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t __spirv_ocl_remquo(
    __clc_vec4_fp32_t, __clc_vec4_fp32_t, __clc_vec4_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t __spirv_ocl_remquo(
    __clc_vec8_fp32_t, __clc_vec8_fp32_t, __clc_vec8_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t __spirv_ocl_remquo(
    __clc_vec8_fp32_t, __clc_vec8_fp32_t, __clc_vec8_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t __spirv_ocl_remquo(
    __clc_vec8_fp32_t, __clc_vec8_fp32_t, __clc_vec8_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t __spirv_ocl_remquo(
    __clc_vec16_fp32_t, __clc_vec16_fp32_t, __clc_vec16_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t __spirv_ocl_remquo(
    __clc_vec16_fp32_t, __clc_vec16_fp32_t, __clc_vec16_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t __spirv_ocl_remquo(
    __clc_vec16_fp32_t, __clc_vec16_fp32_t, __clc_vec16_int32_t __global *);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL __clc_fp64_t
__spirv_ocl_remquo(__clc_fp64_t, __clc_fp64_t, __clc_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_fp64_t
__spirv_ocl_remquo(__clc_fp64_t, __clc_fp64_t, __clc_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_fp64_t
__spirv_ocl_remquo(__clc_fp64_t, __clc_fp64_t, __clc_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t __spirv_ocl_remquo(
    __clc_vec2_fp64_t, __clc_vec2_fp64_t, __clc_vec2_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t __spirv_ocl_remquo(
    __clc_vec2_fp64_t, __clc_vec2_fp64_t, __clc_vec2_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t __spirv_ocl_remquo(
    __clc_vec2_fp64_t, __clc_vec2_fp64_t, __clc_vec2_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t __spirv_ocl_remquo(
    __clc_vec3_fp64_t, __clc_vec3_fp64_t, __clc_vec3_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t __spirv_ocl_remquo(
    __clc_vec3_fp64_t, __clc_vec3_fp64_t, __clc_vec3_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t __spirv_ocl_remquo(
    __clc_vec3_fp64_t, __clc_vec3_fp64_t, __clc_vec3_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t __spirv_ocl_remquo(
    __clc_vec4_fp64_t, __clc_vec4_fp64_t, __clc_vec4_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t __spirv_ocl_remquo(
    __clc_vec4_fp64_t, __clc_vec4_fp64_t, __clc_vec4_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t __spirv_ocl_remquo(
    __clc_vec4_fp64_t, __clc_vec4_fp64_t, __clc_vec4_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t __spirv_ocl_remquo(
    __clc_vec8_fp64_t, __clc_vec8_fp64_t, __clc_vec8_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t __spirv_ocl_remquo(
    __clc_vec8_fp64_t, __clc_vec8_fp64_t, __clc_vec8_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t __spirv_ocl_remquo(
    __clc_vec8_fp64_t, __clc_vec8_fp64_t, __clc_vec8_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t __spirv_ocl_remquo(
    __clc_vec16_fp64_t, __clc_vec16_fp64_t, __clc_vec16_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t __spirv_ocl_remquo(
    __clc_vec16_fp64_t, __clc_vec16_fp64_t, __clc_vec16_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t __spirv_ocl_remquo(
    __clc_vec16_fp64_t, __clc_vec16_fp64_t, __clc_vec16_int32_t __global *);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL __clc_fp16_t
__spirv_ocl_remquo(__clc_fp16_t, __clc_fp16_t, __clc_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_fp16_t
__spirv_ocl_remquo(__clc_fp16_t, __clc_fp16_t, __clc_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_fp16_t
__spirv_ocl_remquo(__clc_fp16_t, __clc_fp16_t, __clc_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t __spirv_ocl_remquo(
    __clc_vec2_fp16_t, __clc_vec2_fp16_t, __clc_vec2_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t __spirv_ocl_remquo(
    __clc_vec2_fp16_t, __clc_vec2_fp16_t, __clc_vec2_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t __spirv_ocl_remquo(
    __clc_vec2_fp16_t, __clc_vec2_fp16_t, __clc_vec2_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t __spirv_ocl_remquo(
    __clc_vec3_fp16_t, __clc_vec3_fp16_t, __clc_vec3_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t __spirv_ocl_remquo(
    __clc_vec3_fp16_t, __clc_vec3_fp16_t, __clc_vec3_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t __spirv_ocl_remquo(
    __clc_vec3_fp16_t, __clc_vec3_fp16_t, __clc_vec3_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t __spirv_ocl_remquo(
    __clc_vec4_fp16_t, __clc_vec4_fp16_t, __clc_vec4_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t __spirv_ocl_remquo(
    __clc_vec4_fp16_t, __clc_vec4_fp16_t, __clc_vec4_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t __spirv_ocl_remquo(
    __clc_vec4_fp16_t, __clc_vec4_fp16_t, __clc_vec4_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t __spirv_ocl_remquo(
    __clc_vec8_fp16_t, __clc_vec8_fp16_t, __clc_vec8_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t __spirv_ocl_remquo(
    __clc_vec8_fp16_t, __clc_vec8_fp16_t, __clc_vec8_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t __spirv_ocl_remquo(
    __clc_vec8_fp16_t, __clc_vec8_fp16_t, __clc_vec8_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t __spirv_ocl_remquo(
    __clc_vec16_fp16_t, __clc_vec16_fp16_t, __clc_vec16_int32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t __spirv_ocl_remquo(
    __clc_vec16_fp16_t, __clc_vec16_fp16_t, __clc_vec16_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t __spirv_ocl_remquo(
    __clc_vec16_fp16_t, __clc_vec16_fp16_t, __clc_vec16_int32_t __global *);
#endif

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DECL __clc_fp32_t
__spirv_ocl_remquo(__clc_fp32_t, __clc_fp32_t, __clc_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t __spirv_ocl_remquo(
    __clc_vec2_fp32_t, __clc_vec2_fp32_t, __clc_vec2_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t __spirv_ocl_remquo(
    __clc_vec3_fp32_t, __clc_vec3_fp32_t, __clc_vec3_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t __spirv_ocl_remquo(
    __clc_vec4_fp32_t, __clc_vec4_fp32_t, __clc_vec4_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t __spirv_ocl_remquo(
    __clc_vec8_fp32_t, __clc_vec8_fp32_t, __clc_vec8_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t __spirv_ocl_remquo(
    __clc_vec16_fp32_t, __clc_vec16_fp32_t, __clc_vec16_int32_t __generic *);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL __clc_fp64_t
__spirv_ocl_remquo(__clc_fp64_t, __clc_fp64_t, __clc_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t __spirv_ocl_remquo(
    __clc_vec2_fp64_t, __clc_vec2_fp64_t, __clc_vec2_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t __spirv_ocl_remquo(
    __clc_vec3_fp64_t, __clc_vec3_fp64_t, __clc_vec3_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t __spirv_ocl_remquo(
    __clc_vec4_fp64_t, __clc_vec4_fp64_t, __clc_vec4_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t __spirv_ocl_remquo(
    __clc_vec8_fp64_t, __clc_vec8_fp64_t, __clc_vec8_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t __spirv_ocl_remquo(
    __clc_vec16_fp64_t, __clc_vec16_fp64_t, __clc_vec16_int32_t __generic *);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL __clc_fp16_t
__spirv_ocl_remquo(__clc_fp16_t, __clc_fp16_t, __clc_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t __spirv_ocl_remquo(
    __clc_vec2_fp16_t, __clc_vec2_fp16_t, __clc_vec2_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t __spirv_ocl_remquo(
    __clc_vec3_fp16_t, __clc_vec3_fp16_t, __clc_vec3_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t __spirv_ocl_remquo(
    __clc_vec4_fp16_t, __clc_vec4_fp16_t, __clc_vec4_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t __spirv_ocl_remquo(
    __clc_vec8_fp16_t, __clc_vec8_fp16_t, __clc_vec8_int32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t __spirv_ocl_remquo(
    __clc_vec16_fp16_t, __clc_vec16_fp16_t, __clc_vec16_int32_t __generic *);
#endif
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_rint(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_rint(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_rint(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_rint(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_rint(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_rint(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_rint(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_rint(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_rint(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_rint(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_rint(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_rint(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_rint(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_rint(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_rint(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_rint(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_rint(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_rint(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_rootn(__clc_fp32_t, __clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_rootn(__clc_vec2_fp32_t, __clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_rootn(__clc_vec3_fp32_t, __clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_rootn(__clc_vec4_fp32_t, __clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_rootn(__clc_vec8_fp32_t, __clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_rootn(__clc_vec16_fp32_t, __clc_vec16_int32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ocl_rootn(__clc_fp64_t, __clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_rootn(__clc_vec2_fp64_t, __clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_rootn(__clc_vec3_fp64_t, __clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_rootn(__clc_vec4_fp64_t, __clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_rootn(__clc_vec8_fp64_t, __clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_rootn(__clc_vec16_fp64_t, __clc_vec16_int32_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ocl_rootn(__clc_fp16_t, __clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_rootn(__clc_vec2_fp16_t, __clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_rootn(__clc_vec3_fp16_t, __clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_rootn(__clc_vec4_fp16_t, __clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_rootn(__clc_vec8_fp16_t, __clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_rootn(__clc_vec16_fp16_t, __clc_vec16_int32_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_char_t
    __spirv_ocl_rotate(__clc_char_t, __clc_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_char_t
    __spirv_ocl_rotate(__clc_vec2_char_t, __clc_vec2_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_char_t
    __spirv_ocl_rotate(__clc_vec3_char_t, __clc_vec3_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_char_t
    __spirv_ocl_rotate(__clc_vec4_char_t, __clc_vec4_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_char_t
    __spirv_ocl_rotate(__clc_vec8_char_t, __clc_vec8_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_char_t
    __spirv_ocl_rotate(__clc_vec16_char_t, __clc_vec16_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ocl_rotate(__clc_int8_t, __clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ocl_rotate(__clc_vec2_int8_t, __clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ocl_rotate(__clc_vec3_int8_t, __clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ocl_rotate(__clc_vec4_int8_t, __clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ocl_rotate(__clc_vec8_int8_t, __clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ocl_rotate(__clc_vec16_int8_t, __clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ocl_rotate(__clc_int16_t, __clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ocl_rotate(__clc_vec2_int16_t, __clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t
    __spirv_ocl_rotate(__clc_vec3_int16_t, __clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ocl_rotate(__clc_vec4_int16_t, __clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ocl_rotate(__clc_vec8_int16_t, __clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ocl_rotate(__clc_vec16_int16_t, __clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ocl_rotate(__clc_int32_t, __clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ocl_rotate(__clc_vec2_int32_t, __clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t
    __spirv_ocl_rotate(__clc_vec3_int32_t, __clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ocl_rotate(__clc_vec4_int32_t, __clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ocl_rotate(__clc_vec8_int32_t, __clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ocl_rotate(__clc_vec16_int32_t, __clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ocl_rotate(__clc_int64_t, __clc_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ocl_rotate(__clc_vec2_int64_t, __clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t
    __spirv_ocl_rotate(__clc_vec3_int64_t, __clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ocl_rotate(__clc_vec4_int64_t, __clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ocl_rotate(__clc_vec8_int64_t, __clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ocl_rotate(__clc_vec16_int64_t, __clc_vec16_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ocl_rotate(__clc_uint8_t, __clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ocl_rotate(__clc_vec2_uint8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t
    __spirv_ocl_rotate(__clc_vec3_uint8_t, __clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ocl_rotate(__clc_vec4_uint8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ocl_rotate(__clc_vec8_uint8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ocl_rotate(__clc_vec16_uint8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ocl_rotate(__clc_uint16_t, __clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ocl_rotate(__clc_vec2_uint16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t
    __spirv_ocl_rotate(__clc_vec3_uint16_t, __clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ocl_rotate(__clc_vec4_uint16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ocl_rotate(__clc_vec8_uint16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ocl_rotate(__clc_vec16_uint16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ocl_rotate(__clc_uint32_t, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ocl_rotate(__clc_vec2_uint32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t
    __spirv_ocl_rotate(__clc_vec3_uint32_t, __clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ocl_rotate(__clc_vec4_uint32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ocl_rotate(__clc_vec8_uint32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ocl_rotate(__clc_vec16_uint32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ocl_rotate(__clc_uint64_t, __clc_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ocl_rotate(__clc_vec2_uint64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t
    __spirv_ocl_rotate(__clc_vec3_uint64_t, __clc_vec3_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ocl_rotate(__clc_vec4_uint64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ocl_rotate(__clc_vec8_uint64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ocl_rotate(__clc_vec16_uint64_t, __clc_vec16_uint64_t);

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_round(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_round(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_round(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_round(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_round(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_round(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_round(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_round(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_round(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_round(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_round(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_round(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_round(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_round(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_round(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_round(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_round(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_round(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_rsqrt(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_rsqrt(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_rsqrt(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_rsqrt(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_rsqrt(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_rsqrt(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_rsqrt(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_rsqrt(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_rsqrt(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_rsqrt(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_rsqrt(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_rsqrt(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_rsqrt(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_rsqrt(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_rsqrt(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_rsqrt(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_rsqrt(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_rsqrt(__clc_vec16_fp16_t);
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

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_char_t
    __spirv_ocl_select(__clc_char_t, __clc_char_t, __clc_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_char_t
    __spirv_ocl_select(__clc_char_t, __clc_char_t, __clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_char_t
    __spirv_ocl_select(__clc_vec2_char_t, __clc_vec2_char_t, __clc_vec2_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_char_t __spirv_ocl_select(
    __clc_vec2_char_t, __clc_vec2_char_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_char_t
    __spirv_ocl_select(__clc_vec3_char_t, __clc_vec3_char_t, __clc_vec3_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_char_t __spirv_ocl_select(
    __clc_vec3_char_t, __clc_vec3_char_t, __clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_char_t
    __spirv_ocl_select(__clc_vec4_char_t, __clc_vec4_char_t, __clc_vec4_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_char_t __spirv_ocl_select(
    __clc_vec4_char_t, __clc_vec4_char_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_char_t
    __spirv_ocl_select(__clc_vec8_char_t, __clc_vec8_char_t, __clc_vec8_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_char_t __spirv_ocl_select(
    __clc_vec8_char_t, __clc_vec8_char_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_char_t __spirv_ocl_select(
    __clc_vec16_char_t, __clc_vec16_char_t, __clc_vec16_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_char_t __spirv_ocl_select(
    __clc_vec16_char_t, __clc_vec16_char_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ocl_select(__clc_int8_t, __clc_int8_t, __clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int8_t
    __spirv_ocl_select(__clc_int8_t, __clc_int8_t, __clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ocl_select(__clc_vec2_int8_t, __clc_vec2_int8_t, __clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t __spirv_ocl_select(
    __clc_vec2_int8_t, __clc_vec2_int8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t
    __spirv_ocl_select(__clc_vec3_int8_t, __clc_vec3_int8_t, __clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int8_t __spirv_ocl_select(
    __clc_vec3_int8_t, __clc_vec3_int8_t, __clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ocl_select(__clc_vec4_int8_t, __clc_vec4_int8_t, __clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t __spirv_ocl_select(
    __clc_vec4_int8_t, __clc_vec4_int8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ocl_select(__clc_vec8_int8_t, __clc_vec8_int8_t, __clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t __spirv_ocl_select(
    __clc_vec8_int8_t, __clc_vec8_int8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t __spirv_ocl_select(
    __clc_vec16_int8_t, __clc_vec16_int8_t, __clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t __spirv_ocl_select(
    __clc_vec16_int8_t, __clc_vec16_int8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ocl_select(__clc_int16_t, __clc_int16_t, __clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int16_t
    __spirv_ocl_select(__clc_int16_t, __clc_int16_t, __clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t __spirv_ocl_select(
    __clc_vec2_int16_t, __clc_vec2_int16_t, __clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t __spirv_ocl_select(
    __clc_vec2_int16_t, __clc_vec2_int16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t __spirv_ocl_select(
    __clc_vec3_int16_t, __clc_vec3_int16_t, __clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int16_t __spirv_ocl_select(
    __clc_vec3_int16_t, __clc_vec3_int16_t, __clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t __spirv_ocl_select(
    __clc_vec4_int16_t, __clc_vec4_int16_t, __clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t __spirv_ocl_select(
    __clc_vec4_int16_t, __clc_vec4_int16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t __spirv_ocl_select(
    __clc_vec8_int16_t, __clc_vec8_int16_t, __clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t __spirv_ocl_select(
    __clc_vec8_int16_t, __clc_vec8_int16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t __spirv_ocl_select(
    __clc_vec16_int16_t, __clc_vec16_int16_t, __clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t __spirv_ocl_select(
    __clc_vec16_int16_t, __clc_vec16_int16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ocl_select(__clc_int32_t, __clc_int32_t, __clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int32_t
    __spirv_ocl_select(__clc_int32_t, __clc_int32_t, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t __spirv_ocl_select(
    __clc_vec2_int32_t, __clc_vec2_int32_t, __clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t __spirv_ocl_select(
    __clc_vec2_int32_t, __clc_vec2_int32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t __spirv_ocl_select(
    __clc_vec3_int32_t, __clc_vec3_int32_t, __clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int32_t __spirv_ocl_select(
    __clc_vec3_int32_t, __clc_vec3_int32_t, __clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t __spirv_ocl_select(
    __clc_vec4_int32_t, __clc_vec4_int32_t, __clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t __spirv_ocl_select(
    __clc_vec4_int32_t, __clc_vec4_int32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t __spirv_ocl_select(
    __clc_vec8_int32_t, __clc_vec8_int32_t, __clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t __spirv_ocl_select(
    __clc_vec8_int32_t, __clc_vec8_int32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t __spirv_ocl_select(
    __clc_vec16_int32_t, __clc_vec16_int32_t, __clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t __spirv_ocl_select(
    __clc_vec16_int32_t, __clc_vec16_int32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ocl_select(__clc_int64_t, __clc_int64_t, __clc_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_int64_t
    __spirv_ocl_select(__clc_int64_t, __clc_int64_t, __clc_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t __spirv_ocl_select(
    __clc_vec2_int64_t, __clc_vec2_int64_t, __clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t __spirv_ocl_select(
    __clc_vec2_int64_t, __clc_vec2_int64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t __spirv_ocl_select(
    __clc_vec3_int64_t, __clc_vec3_int64_t, __clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_int64_t __spirv_ocl_select(
    __clc_vec3_int64_t, __clc_vec3_int64_t, __clc_vec3_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t __spirv_ocl_select(
    __clc_vec4_int64_t, __clc_vec4_int64_t, __clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t __spirv_ocl_select(
    __clc_vec4_int64_t, __clc_vec4_int64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t __spirv_ocl_select(
    __clc_vec8_int64_t, __clc_vec8_int64_t, __clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t __spirv_ocl_select(
    __clc_vec8_int64_t, __clc_vec8_int64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t __spirv_ocl_select(
    __clc_vec16_int64_t, __clc_vec16_int64_t, __clc_vec16_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t __spirv_ocl_select(
    __clc_vec16_int64_t, __clc_vec16_int64_t, __clc_vec16_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ocl_select(__clc_uint8_t, __clc_uint8_t, __clc_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ocl_select(__clc_uint8_t, __clc_uint8_t, __clc_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint8_t
    __spirv_ocl_select(__clc_uint8_t, __clc_uint8_t, __clc_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t __spirv_ocl_select(
    __clc_vec2_uint8_t, __clc_vec2_uint8_t, __clc_vec2_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t __spirv_ocl_select(
    __clc_vec2_uint8_t, __clc_vec2_uint8_t, __clc_vec2_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t __spirv_ocl_select(
    __clc_vec2_uint8_t, __clc_vec2_uint8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t __spirv_ocl_select(
    __clc_vec3_uint8_t, __clc_vec3_uint8_t, __clc_vec3_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t __spirv_ocl_select(
    __clc_vec3_uint8_t, __clc_vec3_uint8_t, __clc_vec3_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint8_t __spirv_ocl_select(
    __clc_vec3_uint8_t, __clc_vec3_uint8_t, __clc_vec3_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t __spirv_ocl_select(
    __clc_vec4_uint8_t, __clc_vec4_uint8_t, __clc_vec4_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t __spirv_ocl_select(
    __clc_vec4_uint8_t, __clc_vec4_uint8_t, __clc_vec4_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t __spirv_ocl_select(
    __clc_vec4_uint8_t, __clc_vec4_uint8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t __spirv_ocl_select(
    __clc_vec8_uint8_t, __clc_vec8_uint8_t, __clc_vec8_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t __spirv_ocl_select(
    __clc_vec8_uint8_t, __clc_vec8_uint8_t, __clc_vec8_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t __spirv_ocl_select(
    __clc_vec8_uint8_t, __clc_vec8_uint8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t __spirv_ocl_select(
    __clc_vec16_uint8_t, __clc_vec16_uint8_t, __clc_vec16_char_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t __spirv_ocl_select(
    __clc_vec16_uint8_t, __clc_vec16_uint8_t, __clc_vec16_int8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t __spirv_ocl_select(
    __clc_vec16_uint8_t, __clc_vec16_uint8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ocl_select(__clc_uint16_t, __clc_uint16_t, __clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint16_t
    __spirv_ocl_select(__clc_uint16_t, __clc_uint16_t, __clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t __spirv_ocl_select(
    __clc_vec2_uint16_t, __clc_vec2_uint16_t, __clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t __spirv_ocl_select(
    __clc_vec2_uint16_t, __clc_vec2_uint16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t __spirv_ocl_select(
    __clc_vec3_uint16_t, __clc_vec3_uint16_t, __clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint16_t __spirv_ocl_select(
    __clc_vec3_uint16_t, __clc_vec3_uint16_t, __clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t __spirv_ocl_select(
    __clc_vec4_uint16_t, __clc_vec4_uint16_t, __clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t __spirv_ocl_select(
    __clc_vec4_uint16_t, __clc_vec4_uint16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t __spirv_ocl_select(
    __clc_vec8_uint16_t, __clc_vec8_uint16_t, __clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t __spirv_ocl_select(
    __clc_vec8_uint16_t, __clc_vec8_uint16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t __spirv_ocl_select(
    __clc_vec16_uint16_t, __clc_vec16_uint16_t, __clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t __spirv_ocl_select(
    __clc_vec16_uint16_t, __clc_vec16_uint16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ocl_select(__clc_uint32_t, __clc_uint32_t, __clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint32_t
    __spirv_ocl_select(__clc_uint32_t, __clc_uint32_t, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t __spirv_ocl_select(
    __clc_vec2_uint32_t, __clc_vec2_uint32_t, __clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t __spirv_ocl_select(
    __clc_vec2_uint32_t, __clc_vec2_uint32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t __spirv_ocl_select(
    __clc_vec3_uint32_t, __clc_vec3_uint32_t, __clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint32_t __spirv_ocl_select(
    __clc_vec3_uint32_t, __clc_vec3_uint32_t, __clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t __spirv_ocl_select(
    __clc_vec4_uint32_t, __clc_vec4_uint32_t, __clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t __spirv_ocl_select(
    __clc_vec4_uint32_t, __clc_vec4_uint32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t __spirv_ocl_select(
    __clc_vec8_uint32_t, __clc_vec8_uint32_t, __clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t __spirv_ocl_select(
    __clc_vec8_uint32_t, __clc_vec8_uint32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t __spirv_ocl_select(
    __clc_vec16_uint32_t, __clc_vec16_uint32_t, __clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t __spirv_ocl_select(
    __clc_vec16_uint32_t, __clc_vec16_uint32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ocl_select(__clc_uint64_t, __clc_uint64_t, __clc_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_uint64_t
    __spirv_ocl_select(__clc_uint64_t, __clc_uint64_t, __clc_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t __spirv_ocl_select(
    __clc_vec2_uint64_t, __clc_vec2_uint64_t, __clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t __spirv_ocl_select(
    __clc_vec2_uint64_t, __clc_vec2_uint64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t __spirv_ocl_select(
    __clc_vec3_uint64_t, __clc_vec3_uint64_t, __clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_uint64_t __spirv_ocl_select(
    __clc_vec3_uint64_t, __clc_vec3_uint64_t, __clc_vec3_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t __spirv_ocl_select(
    __clc_vec4_uint64_t, __clc_vec4_uint64_t, __clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t __spirv_ocl_select(
    __clc_vec4_uint64_t, __clc_vec4_uint64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t __spirv_ocl_select(
    __clc_vec8_uint64_t, __clc_vec8_uint64_t, __clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t __spirv_ocl_select(
    __clc_vec8_uint64_t, __clc_vec8_uint64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t __spirv_ocl_select(
    __clc_vec16_uint64_t, __clc_vec16_uint64_t, __clc_vec16_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t __spirv_ocl_select(
    __clc_vec16_uint64_t, __clc_vec16_uint64_t, __clc_vec16_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_select(__clc_fp32_t, __clc_fp32_t, __clc_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_select(__clc_fp32_t, __clc_fp32_t, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t __spirv_ocl_select(
    __clc_vec2_fp32_t, __clc_vec2_fp32_t, __clc_vec2_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t __spirv_ocl_select(
    __clc_vec2_fp32_t, __clc_vec2_fp32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t __spirv_ocl_select(
    __clc_vec3_fp32_t, __clc_vec3_fp32_t, __clc_vec3_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t __spirv_ocl_select(
    __clc_vec3_fp32_t, __clc_vec3_fp32_t, __clc_vec3_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t __spirv_ocl_select(
    __clc_vec4_fp32_t, __clc_vec4_fp32_t, __clc_vec4_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t __spirv_ocl_select(
    __clc_vec4_fp32_t, __clc_vec4_fp32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t __spirv_ocl_select(
    __clc_vec8_fp32_t, __clc_vec8_fp32_t, __clc_vec8_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t __spirv_ocl_select(
    __clc_vec8_fp32_t, __clc_vec8_fp32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t __spirv_ocl_select(
    __clc_vec16_fp32_t, __clc_vec16_fp32_t, __clc_vec16_int32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t __spirv_ocl_select(
    __clc_vec16_fp32_t, __clc_vec16_fp32_t, __clc_vec16_uint32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ocl_select(__clc_fp64_t, __clc_fp64_t, __clc_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ocl_select(__clc_fp64_t, __clc_fp64_t, __clc_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t __spirv_ocl_select(
    __clc_vec2_fp64_t, __clc_vec2_fp64_t, __clc_vec2_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t __spirv_ocl_select(
    __clc_vec2_fp64_t, __clc_vec2_fp64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t __spirv_ocl_select(
    __clc_vec3_fp64_t, __clc_vec3_fp64_t, __clc_vec3_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t __spirv_ocl_select(
    __clc_vec3_fp64_t, __clc_vec3_fp64_t, __clc_vec3_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t __spirv_ocl_select(
    __clc_vec4_fp64_t, __clc_vec4_fp64_t, __clc_vec4_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t __spirv_ocl_select(
    __clc_vec4_fp64_t, __clc_vec4_fp64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t __spirv_ocl_select(
    __clc_vec8_fp64_t, __clc_vec8_fp64_t, __clc_vec8_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t __spirv_ocl_select(
    __clc_vec8_fp64_t, __clc_vec8_fp64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t __spirv_ocl_select(
    __clc_vec16_fp64_t, __clc_vec16_fp64_t, __clc_vec16_int64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t __spirv_ocl_select(
    __clc_vec16_fp64_t, __clc_vec16_fp64_t, __clc_vec16_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ocl_select(__clc_fp16_t, __clc_fp16_t, __clc_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ocl_select(__clc_fp16_t, __clc_fp16_t, __clc_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t __spirv_ocl_select(
    __clc_vec2_fp16_t, __clc_vec2_fp16_t, __clc_vec2_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t __spirv_ocl_select(
    __clc_vec2_fp16_t, __clc_vec2_fp16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t __spirv_ocl_select(
    __clc_vec3_fp16_t, __clc_vec3_fp16_t, __clc_vec3_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t __spirv_ocl_select(
    __clc_vec3_fp16_t, __clc_vec3_fp16_t, __clc_vec3_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t __spirv_ocl_select(
    __clc_vec4_fp16_t, __clc_vec4_fp16_t, __clc_vec4_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t __spirv_ocl_select(
    __clc_vec4_fp16_t, __clc_vec4_fp16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t __spirv_ocl_select(
    __clc_vec8_fp16_t, __clc_vec8_fp16_t, __clc_vec8_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t __spirv_ocl_select(
    __clc_vec8_fp16_t, __clc_vec8_fp16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t __spirv_ocl_select(
    __clc_vec16_fp16_t, __clc_vec16_fp16_t, __clc_vec16_int16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t __spirv_ocl_select(
    __clc_vec16_fp16_t, __clc_vec16_fp16_t, __clc_vec16_uint16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_char_t
    __spirv_ocl_shuffle(__clc_vec2_char_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_char_t
    __spirv_ocl_shuffle(__clc_vec4_char_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_char_t
    __spirv_ocl_shuffle(__clc_vec8_char_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_char_t
    __spirv_ocl_shuffle(__clc_vec16_char_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_char_t
    __spirv_ocl_shuffle(__clc_vec2_char_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_char_t
    __spirv_ocl_shuffle(__clc_vec4_char_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_char_t
    __spirv_ocl_shuffle(__clc_vec8_char_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_char_t
    __spirv_ocl_shuffle(__clc_vec16_char_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_char_t
    __spirv_ocl_shuffle(__clc_vec2_char_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_char_t
    __spirv_ocl_shuffle(__clc_vec4_char_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_char_t
    __spirv_ocl_shuffle(__clc_vec8_char_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_char_t
    __spirv_ocl_shuffle(__clc_vec16_char_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_char_t
    __spirv_ocl_shuffle(__clc_vec2_char_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_char_t
    __spirv_ocl_shuffle(__clc_vec4_char_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_char_t
    __spirv_ocl_shuffle(__clc_vec8_char_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_char_t
    __spirv_ocl_shuffle(__clc_vec16_char_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ocl_shuffle(__clc_vec2_int8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ocl_shuffle(__clc_vec4_int8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ocl_shuffle(__clc_vec8_int8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t
    __spirv_ocl_shuffle(__clc_vec16_int8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ocl_shuffle(__clc_vec2_int8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ocl_shuffle(__clc_vec4_int8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ocl_shuffle(__clc_vec8_int8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t
    __spirv_ocl_shuffle(__clc_vec16_int8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ocl_shuffle(__clc_vec2_int8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ocl_shuffle(__clc_vec4_int8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ocl_shuffle(__clc_vec8_int8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t
    __spirv_ocl_shuffle(__clc_vec16_int8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ocl_shuffle(__clc_vec2_int8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ocl_shuffle(__clc_vec4_int8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ocl_shuffle(__clc_vec8_int8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t
    __spirv_ocl_shuffle(__clc_vec16_int8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ocl_shuffle(__clc_vec2_int16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ocl_shuffle(__clc_vec4_int16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ocl_shuffle(__clc_vec8_int16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t
    __spirv_ocl_shuffle(__clc_vec16_int16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ocl_shuffle(__clc_vec2_int16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ocl_shuffle(__clc_vec4_int16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ocl_shuffle(__clc_vec8_int16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t
    __spirv_ocl_shuffle(__clc_vec16_int16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ocl_shuffle(__clc_vec2_int16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ocl_shuffle(__clc_vec4_int16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ocl_shuffle(__clc_vec8_int16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t
    __spirv_ocl_shuffle(__clc_vec16_int16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ocl_shuffle(__clc_vec2_int16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ocl_shuffle(__clc_vec4_int16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ocl_shuffle(__clc_vec8_int16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t
    __spirv_ocl_shuffle(__clc_vec16_int16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ocl_shuffle(__clc_vec2_int32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ocl_shuffle(__clc_vec4_int32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ocl_shuffle(__clc_vec8_int32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t
    __spirv_ocl_shuffle(__clc_vec16_int32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ocl_shuffle(__clc_vec2_int32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ocl_shuffle(__clc_vec4_int32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ocl_shuffle(__clc_vec8_int32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t
    __spirv_ocl_shuffle(__clc_vec16_int32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ocl_shuffle(__clc_vec2_int32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ocl_shuffle(__clc_vec4_int32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ocl_shuffle(__clc_vec8_int32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t
    __spirv_ocl_shuffle(__clc_vec16_int32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ocl_shuffle(__clc_vec2_int32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ocl_shuffle(__clc_vec4_int32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ocl_shuffle(__clc_vec8_int32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t
    __spirv_ocl_shuffle(__clc_vec16_int32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ocl_shuffle(__clc_vec2_int64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ocl_shuffle(__clc_vec4_int64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ocl_shuffle(__clc_vec8_int64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t
    __spirv_ocl_shuffle(__clc_vec16_int64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ocl_shuffle(__clc_vec2_int64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ocl_shuffle(__clc_vec4_int64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ocl_shuffle(__clc_vec8_int64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t
    __spirv_ocl_shuffle(__clc_vec16_int64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ocl_shuffle(__clc_vec2_int64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ocl_shuffle(__clc_vec4_int64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ocl_shuffle(__clc_vec8_int64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t
    __spirv_ocl_shuffle(__clc_vec16_int64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ocl_shuffle(__clc_vec2_int64_t, __clc_vec16_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ocl_shuffle(__clc_vec4_int64_t, __clc_vec16_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ocl_shuffle(__clc_vec8_int64_t, __clc_vec16_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t
    __spirv_ocl_shuffle(__clc_vec16_int64_t, __clc_vec16_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ocl_shuffle(__clc_vec2_uint8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ocl_shuffle(__clc_vec4_uint8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ocl_shuffle(__clc_vec8_uint8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t
    __spirv_ocl_shuffle(__clc_vec16_uint8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ocl_shuffle(__clc_vec2_uint8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ocl_shuffle(__clc_vec4_uint8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ocl_shuffle(__clc_vec8_uint8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t
    __spirv_ocl_shuffle(__clc_vec16_uint8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ocl_shuffle(__clc_vec2_uint8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ocl_shuffle(__clc_vec4_uint8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ocl_shuffle(__clc_vec8_uint8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t
    __spirv_ocl_shuffle(__clc_vec16_uint8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ocl_shuffle(__clc_vec2_uint8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ocl_shuffle(__clc_vec4_uint8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ocl_shuffle(__clc_vec8_uint8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t
    __spirv_ocl_shuffle(__clc_vec16_uint8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ocl_shuffle(__clc_vec2_uint16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ocl_shuffle(__clc_vec4_uint16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ocl_shuffle(__clc_vec8_uint16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t
    __spirv_ocl_shuffle(__clc_vec16_uint16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ocl_shuffle(__clc_vec2_uint16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ocl_shuffle(__clc_vec4_uint16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ocl_shuffle(__clc_vec8_uint16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t
    __spirv_ocl_shuffle(__clc_vec16_uint16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ocl_shuffle(__clc_vec2_uint16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ocl_shuffle(__clc_vec4_uint16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ocl_shuffle(__clc_vec8_uint16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t
    __spirv_ocl_shuffle(__clc_vec16_uint16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ocl_shuffle(__clc_vec2_uint16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ocl_shuffle(__clc_vec4_uint16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ocl_shuffle(__clc_vec8_uint16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t
    __spirv_ocl_shuffle(__clc_vec16_uint16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ocl_shuffle(__clc_vec2_uint32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ocl_shuffle(__clc_vec4_uint32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ocl_shuffle(__clc_vec8_uint32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t
    __spirv_ocl_shuffle(__clc_vec16_uint32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ocl_shuffle(__clc_vec2_uint32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ocl_shuffle(__clc_vec4_uint32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ocl_shuffle(__clc_vec8_uint32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t
    __spirv_ocl_shuffle(__clc_vec16_uint32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ocl_shuffle(__clc_vec2_uint32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ocl_shuffle(__clc_vec4_uint32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ocl_shuffle(__clc_vec8_uint32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t
    __spirv_ocl_shuffle(__clc_vec16_uint32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ocl_shuffle(__clc_vec2_uint32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ocl_shuffle(__clc_vec4_uint32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ocl_shuffle(__clc_vec8_uint32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t
    __spirv_ocl_shuffle(__clc_vec16_uint32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ocl_shuffle(__clc_vec2_uint64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ocl_shuffle(__clc_vec4_uint64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ocl_shuffle(__clc_vec8_uint64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t
    __spirv_ocl_shuffle(__clc_vec16_uint64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ocl_shuffle(__clc_vec2_uint64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ocl_shuffle(__clc_vec4_uint64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ocl_shuffle(__clc_vec8_uint64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t
    __spirv_ocl_shuffle(__clc_vec16_uint64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ocl_shuffle(__clc_vec2_uint64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ocl_shuffle(__clc_vec4_uint64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ocl_shuffle(__clc_vec8_uint64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t
    __spirv_ocl_shuffle(__clc_vec16_uint64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ocl_shuffle(__clc_vec2_uint64_t, __clc_vec16_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ocl_shuffle(__clc_vec4_uint64_t, __clc_vec16_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ocl_shuffle(__clc_vec8_uint64_t, __clc_vec16_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t
    __spirv_ocl_shuffle(__clc_vec16_uint64_t, __clc_vec16_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_shuffle(__clc_vec2_fp32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_shuffle(__clc_vec4_fp32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_shuffle(__clc_vec8_fp32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_shuffle(__clc_vec16_fp32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_shuffle(__clc_vec2_fp32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_shuffle(__clc_vec4_fp32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_shuffle(__clc_vec8_fp32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_shuffle(__clc_vec16_fp32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_shuffle(__clc_vec2_fp32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_shuffle(__clc_vec4_fp32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_shuffle(__clc_vec8_fp32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_shuffle(__clc_vec16_fp32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_shuffle(__clc_vec2_fp32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_shuffle(__clc_vec4_fp32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_shuffle(__clc_vec8_fp32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_shuffle(__clc_vec16_fp32_t, __clc_vec16_uint32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_shuffle(__clc_vec2_fp64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_shuffle(__clc_vec4_fp64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_shuffle(__clc_vec8_fp64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_shuffle(__clc_vec16_fp64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_shuffle(__clc_vec2_fp64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_shuffle(__clc_vec4_fp64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_shuffle(__clc_vec8_fp64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_shuffle(__clc_vec16_fp64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_shuffle(__clc_vec2_fp64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_shuffle(__clc_vec4_fp64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_shuffle(__clc_vec8_fp64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_shuffle(__clc_vec16_fp64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_shuffle(__clc_vec2_fp64_t, __clc_vec16_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_shuffle(__clc_vec4_fp64_t, __clc_vec16_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_shuffle(__clc_vec8_fp64_t, __clc_vec16_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_shuffle(__clc_vec16_fp64_t, __clc_vec16_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_shuffle(__clc_vec2_fp16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_shuffle(__clc_vec4_fp16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_shuffle(__clc_vec8_fp16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_shuffle(__clc_vec16_fp16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_shuffle(__clc_vec2_fp16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_shuffle(__clc_vec4_fp16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_shuffle(__clc_vec8_fp16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_shuffle(__clc_vec16_fp16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_shuffle(__clc_vec2_fp16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_shuffle(__clc_vec4_fp16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_shuffle(__clc_vec8_fp16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_shuffle(__clc_vec16_fp16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_shuffle(__clc_vec2_fp16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_shuffle(__clc_vec4_fp16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_shuffle(__clc_vec8_fp16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_shuffle(__clc_vec16_fp16_t, __clc_vec16_uint16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_char_t __spirv_ocl_shuffle2(
    __clc_vec2_char_t, __clc_vec2_char_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_char_t __spirv_ocl_shuffle2(
    __clc_vec4_char_t, __clc_vec4_char_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_char_t __spirv_ocl_shuffle2(
    __clc_vec8_char_t, __clc_vec8_char_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_char_t __spirv_ocl_shuffle2(
    __clc_vec16_char_t, __clc_vec16_char_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_char_t __spirv_ocl_shuffle2(
    __clc_vec2_char_t, __clc_vec2_char_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_char_t __spirv_ocl_shuffle2(
    __clc_vec4_char_t, __clc_vec4_char_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_char_t __spirv_ocl_shuffle2(
    __clc_vec8_char_t, __clc_vec8_char_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_char_t __spirv_ocl_shuffle2(
    __clc_vec16_char_t, __clc_vec16_char_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_char_t __spirv_ocl_shuffle2(
    __clc_vec2_char_t, __clc_vec2_char_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_char_t __spirv_ocl_shuffle2(
    __clc_vec4_char_t, __clc_vec4_char_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_char_t __spirv_ocl_shuffle2(
    __clc_vec8_char_t, __clc_vec8_char_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_char_t __spirv_ocl_shuffle2(
    __clc_vec16_char_t, __clc_vec16_char_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_char_t __spirv_ocl_shuffle2(
    __clc_vec2_char_t, __clc_vec2_char_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_char_t __spirv_ocl_shuffle2(
    __clc_vec4_char_t, __clc_vec4_char_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_char_t __spirv_ocl_shuffle2(
    __clc_vec8_char_t, __clc_vec8_char_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_char_t __spirv_ocl_shuffle2(
    __clc_vec16_char_t, __clc_vec16_char_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t __spirv_ocl_shuffle2(
    __clc_vec2_int8_t, __clc_vec2_int8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t __spirv_ocl_shuffle2(
    __clc_vec4_int8_t, __clc_vec4_int8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t __spirv_ocl_shuffle2(
    __clc_vec8_int8_t, __clc_vec8_int8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int8_t __spirv_ocl_shuffle2(
    __clc_vec16_int8_t, __clc_vec16_int8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t __spirv_ocl_shuffle2(
    __clc_vec2_int8_t, __clc_vec2_int8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t __spirv_ocl_shuffle2(
    __clc_vec4_int8_t, __clc_vec4_int8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t __spirv_ocl_shuffle2(
    __clc_vec8_int8_t, __clc_vec8_int8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int8_t __spirv_ocl_shuffle2(
    __clc_vec16_int8_t, __clc_vec16_int8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t __spirv_ocl_shuffle2(
    __clc_vec2_int8_t, __clc_vec2_int8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t __spirv_ocl_shuffle2(
    __clc_vec4_int8_t, __clc_vec4_int8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t __spirv_ocl_shuffle2(
    __clc_vec8_int8_t, __clc_vec8_int8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int8_t __spirv_ocl_shuffle2(
    __clc_vec16_int8_t, __clc_vec16_int8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t __spirv_ocl_shuffle2(
    __clc_vec2_int8_t, __clc_vec2_int8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t __spirv_ocl_shuffle2(
    __clc_vec4_int8_t, __clc_vec4_int8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t __spirv_ocl_shuffle2(
    __clc_vec8_int8_t, __clc_vec8_int8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int8_t __spirv_ocl_shuffle2(
    __clc_vec16_int8_t, __clc_vec16_int8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t __spirv_ocl_shuffle2(
    __clc_vec2_int16_t, __clc_vec2_int16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t __spirv_ocl_shuffle2(
    __clc_vec4_int16_t, __clc_vec4_int16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t __spirv_ocl_shuffle2(
    __clc_vec8_int16_t, __clc_vec8_int16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int16_t __spirv_ocl_shuffle2(
    __clc_vec16_int16_t, __clc_vec16_int16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t __spirv_ocl_shuffle2(
    __clc_vec2_int16_t, __clc_vec2_int16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t __spirv_ocl_shuffle2(
    __clc_vec4_int16_t, __clc_vec4_int16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t __spirv_ocl_shuffle2(
    __clc_vec8_int16_t, __clc_vec8_int16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int16_t __spirv_ocl_shuffle2(
    __clc_vec16_int16_t, __clc_vec16_int16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t __spirv_ocl_shuffle2(
    __clc_vec2_int16_t, __clc_vec2_int16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t __spirv_ocl_shuffle2(
    __clc_vec4_int16_t, __clc_vec4_int16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t __spirv_ocl_shuffle2(
    __clc_vec8_int16_t, __clc_vec8_int16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int16_t __spirv_ocl_shuffle2(
    __clc_vec16_int16_t, __clc_vec16_int16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t __spirv_ocl_shuffle2(
    __clc_vec2_int16_t, __clc_vec2_int16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t __spirv_ocl_shuffle2(
    __clc_vec4_int16_t, __clc_vec4_int16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t __spirv_ocl_shuffle2(
    __clc_vec8_int16_t, __clc_vec8_int16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int16_t __spirv_ocl_shuffle2(
    __clc_vec16_int16_t, __clc_vec16_int16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t __spirv_ocl_shuffle2(
    __clc_vec2_int32_t, __clc_vec2_int32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t __spirv_ocl_shuffle2(
    __clc_vec4_int32_t, __clc_vec4_int32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t __spirv_ocl_shuffle2(
    __clc_vec8_int32_t, __clc_vec8_int32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int32_t __spirv_ocl_shuffle2(
    __clc_vec16_int32_t, __clc_vec16_int32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t __spirv_ocl_shuffle2(
    __clc_vec2_int32_t, __clc_vec2_int32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t __spirv_ocl_shuffle2(
    __clc_vec4_int32_t, __clc_vec4_int32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t __spirv_ocl_shuffle2(
    __clc_vec8_int32_t, __clc_vec8_int32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int32_t __spirv_ocl_shuffle2(
    __clc_vec16_int32_t, __clc_vec16_int32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t __spirv_ocl_shuffle2(
    __clc_vec2_int32_t, __clc_vec2_int32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t __spirv_ocl_shuffle2(
    __clc_vec4_int32_t, __clc_vec4_int32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t __spirv_ocl_shuffle2(
    __clc_vec8_int32_t, __clc_vec8_int32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int32_t __spirv_ocl_shuffle2(
    __clc_vec16_int32_t, __clc_vec16_int32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t __spirv_ocl_shuffle2(
    __clc_vec2_int32_t, __clc_vec2_int32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t __spirv_ocl_shuffle2(
    __clc_vec4_int32_t, __clc_vec4_int32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t __spirv_ocl_shuffle2(
    __clc_vec8_int32_t, __clc_vec8_int32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int32_t __spirv_ocl_shuffle2(
    __clc_vec16_int32_t, __clc_vec16_int32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t __spirv_ocl_shuffle2(
    __clc_vec2_int64_t, __clc_vec2_int64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t __spirv_ocl_shuffle2(
    __clc_vec4_int64_t, __clc_vec4_int64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t __spirv_ocl_shuffle2(
    __clc_vec8_int64_t, __clc_vec8_int64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_int64_t __spirv_ocl_shuffle2(
    __clc_vec16_int64_t, __clc_vec16_int64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t __spirv_ocl_shuffle2(
    __clc_vec2_int64_t, __clc_vec2_int64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t __spirv_ocl_shuffle2(
    __clc_vec4_int64_t, __clc_vec4_int64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t __spirv_ocl_shuffle2(
    __clc_vec8_int64_t, __clc_vec8_int64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_int64_t __spirv_ocl_shuffle2(
    __clc_vec16_int64_t, __clc_vec16_int64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t __spirv_ocl_shuffle2(
    __clc_vec2_int64_t, __clc_vec2_int64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t __spirv_ocl_shuffle2(
    __clc_vec4_int64_t, __clc_vec4_int64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t __spirv_ocl_shuffle2(
    __clc_vec8_int64_t, __clc_vec8_int64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_int64_t __spirv_ocl_shuffle2(
    __clc_vec16_int64_t, __clc_vec16_int64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t __spirv_ocl_shuffle2(
    __clc_vec2_int64_t, __clc_vec2_int64_t, __clc_vec16_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t __spirv_ocl_shuffle2(
    __clc_vec4_int64_t, __clc_vec4_int64_t, __clc_vec16_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t __spirv_ocl_shuffle2(
    __clc_vec8_int64_t, __clc_vec8_int64_t, __clc_vec16_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_int64_t __spirv_ocl_shuffle2(
    __clc_vec16_int64_t, __clc_vec16_int64_t, __clc_vec16_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t __spirv_ocl_shuffle2(
    __clc_vec2_uint8_t, __clc_vec2_uint8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t __spirv_ocl_shuffle2(
    __clc_vec4_uint8_t, __clc_vec4_uint8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t __spirv_ocl_shuffle2(
    __clc_vec8_uint8_t, __clc_vec8_uint8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint8_t __spirv_ocl_shuffle2(
    __clc_vec16_uint8_t, __clc_vec16_uint8_t, __clc_vec2_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t __spirv_ocl_shuffle2(
    __clc_vec2_uint8_t, __clc_vec2_uint8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t __spirv_ocl_shuffle2(
    __clc_vec4_uint8_t, __clc_vec4_uint8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t __spirv_ocl_shuffle2(
    __clc_vec8_uint8_t, __clc_vec8_uint8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint8_t __spirv_ocl_shuffle2(
    __clc_vec16_uint8_t, __clc_vec16_uint8_t, __clc_vec4_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t __spirv_ocl_shuffle2(
    __clc_vec2_uint8_t, __clc_vec2_uint8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t __spirv_ocl_shuffle2(
    __clc_vec4_uint8_t, __clc_vec4_uint8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t __spirv_ocl_shuffle2(
    __clc_vec8_uint8_t, __clc_vec8_uint8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint8_t __spirv_ocl_shuffle2(
    __clc_vec16_uint8_t, __clc_vec16_uint8_t, __clc_vec8_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t __spirv_ocl_shuffle2(
    __clc_vec2_uint8_t, __clc_vec2_uint8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t __spirv_ocl_shuffle2(
    __clc_vec4_uint8_t, __clc_vec4_uint8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t __spirv_ocl_shuffle2(
    __clc_vec8_uint8_t, __clc_vec8_uint8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint8_t __spirv_ocl_shuffle2(
    __clc_vec16_uint8_t, __clc_vec16_uint8_t, __clc_vec16_uint8_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t __spirv_ocl_shuffle2(
    __clc_vec2_uint16_t, __clc_vec2_uint16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t __spirv_ocl_shuffle2(
    __clc_vec4_uint16_t, __clc_vec4_uint16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t __spirv_ocl_shuffle2(
    __clc_vec8_uint16_t, __clc_vec8_uint16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint16_t __spirv_ocl_shuffle2(
    __clc_vec16_uint16_t, __clc_vec16_uint16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t __spirv_ocl_shuffle2(
    __clc_vec2_uint16_t, __clc_vec2_uint16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t __spirv_ocl_shuffle2(
    __clc_vec4_uint16_t, __clc_vec4_uint16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t __spirv_ocl_shuffle2(
    __clc_vec8_uint16_t, __clc_vec8_uint16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint16_t __spirv_ocl_shuffle2(
    __clc_vec16_uint16_t, __clc_vec16_uint16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t __spirv_ocl_shuffle2(
    __clc_vec2_uint16_t, __clc_vec2_uint16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t __spirv_ocl_shuffle2(
    __clc_vec4_uint16_t, __clc_vec4_uint16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t __spirv_ocl_shuffle2(
    __clc_vec8_uint16_t, __clc_vec8_uint16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint16_t __spirv_ocl_shuffle2(
    __clc_vec16_uint16_t, __clc_vec16_uint16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t __spirv_ocl_shuffle2(
    __clc_vec2_uint16_t, __clc_vec2_uint16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t __spirv_ocl_shuffle2(
    __clc_vec4_uint16_t, __clc_vec4_uint16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t __spirv_ocl_shuffle2(
    __clc_vec8_uint16_t, __clc_vec8_uint16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint16_t __spirv_ocl_shuffle2(
    __clc_vec16_uint16_t, __clc_vec16_uint16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t __spirv_ocl_shuffle2(
    __clc_vec2_uint32_t, __clc_vec2_uint32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t __spirv_ocl_shuffle2(
    __clc_vec4_uint32_t, __clc_vec4_uint32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t __spirv_ocl_shuffle2(
    __clc_vec8_uint32_t, __clc_vec8_uint32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint32_t __spirv_ocl_shuffle2(
    __clc_vec16_uint32_t, __clc_vec16_uint32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t __spirv_ocl_shuffle2(
    __clc_vec2_uint32_t, __clc_vec2_uint32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t __spirv_ocl_shuffle2(
    __clc_vec4_uint32_t, __clc_vec4_uint32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t __spirv_ocl_shuffle2(
    __clc_vec8_uint32_t, __clc_vec8_uint32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint32_t __spirv_ocl_shuffle2(
    __clc_vec16_uint32_t, __clc_vec16_uint32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t __spirv_ocl_shuffle2(
    __clc_vec2_uint32_t, __clc_vec2_uint32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t __spirv_ocl_shuffle2(
    __clc_vec4_uint32_t, __clc_vec4_uint32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t __spirv_ocl_shuffle2(
    __clc_vec8_uint32_t, __clc_vec8_uint32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint32_t __spirv_ocl_shuffle2(
    __clc_vec16_uint32_t, __clc_vec16_uint32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t __spirv_ocl_shuffle2(
    __clc_vec2_uint32_t, __clc_vec2_uint32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t __spirv_ocl_shuffle2(
    __clc_vec4_uint32_t, __clc_vec4_uint32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t __spirv_ocl_shuffle2(
    __clc_vec8_uint32_t, __clc_vec8_uint32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint32_t __spirv_ocl_shuffle2(
    __clc_vec16_uint32_t, __clc_vec16_uint32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t __spirv_ocl_shuffle2(
    __clc_vec2_uint64_t, __clc_vec2_uint64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t __spirv_ocl_shuffle2(
    __clc_vec4_uint64_t, __clc_vec4_uint64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t __spirv_ocl_shuffle2(
    __clc_vec8_uint64_t, __clc_vec8_uint64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_uint64_t __spirv_ocl_shuffle2(
    __clc_vec16_uint64_t, __clc_vec16_uint64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t __spirv_ocl_shuffle2(
    __clc_vec2_uint64_t, __clc_vec2_uint64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t __spirv_ocl_shuffle2(
    __clc_vec4_uint64_t, __clc_vec4_uint64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t __spirv_ocl_shuffle2(
    __clc_vec8_uint64_t, __clc_vec8_uint64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_uint64_t __spirv_ocl_shuffle2(
    __clc_vec16_uint64_t, __clc_vec16_uint64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t __spirv_ocl_shuffle2(
    __clc_vec2_uint64_t, __clc_vec2_uint64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t __spirv_ocl_shuffle2(
    __clc_vec4_uint64_t, __clc_vec4_uint64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t __spirv_ocl_shuffle2(
    __clc_vec8_uint64_t, __clc_vec8_uint64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_uint64_t __spirv_ocl_shuffle2(
    __clc_vec16_uint64_t, __clc_vec16_uint64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t __spirv_ocl_shuffle2(
    __clc_vec2_uint64_t, __clc_vec2_uint64_t, __clc_vec16_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t __spirv_ocl_shuffle2(
    __clc_vec4_uint64_t, __clc_vec4_uint64_t, __clc_vec16_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t __spirv_ocl_shuffle2(
    __clc_vec8_uint64_t, __clc_vec8_uint64_t, __clc_vec16_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_uint64_t __spirv_ocl_shuffle2(
    __clc_vec16_uint64_t, __clc_vec16_uint64_t, __clc_vec16_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t __spirv_ocl_shuffle2(
    __clc_vec2_fp32_t, __clc_vec2_fp32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t __spirv_ocl_shuffle2(
    __clc_vec4_fp32_t, __clc_vec4_fp32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t __spirv_ocl_shuffle2(
    __clc_vec8_fp32_t, __clc_vec8_fp32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t __spirv_ocl_shuffle2(
    __clc_vec16_fp32_t, __clc_vec16_fp32_t, __clc_vec2_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t __spirv_ocl_shuffle2(
    __clc_vec2_fp32_t, __clc_vec2_fp32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t __spirv_ocl_shuffle2(
    __clc_vec4_fp32_t, __clc_vec4_fp32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t __spirv_ocl_shuffle2(
    __clc_vec8_fp32_t, __clc_vec8_fp32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t __spirv_ocl_shuffle2(
    __clc_vec16_fp32_t, __clc_vec16_fp32_t, __clc_vec4_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t __spirv_ocl_shuffle2(
    __clc_vec2_fp32_t, __clc_vec2_fp32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t __spirv_ocl_shuffle2(
    __clc_vec4_fp32_t, __clc_vec4_fp32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t __spirv_ocl_shuffle2(
    __clc_vec8_fp32_t, __clc_vec8_fp32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t __spirv_ocl_shuffle2(
    __clc_vec16_fp32_t, __clc_vec16_fp32_t, __clc_vec8_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t __spirv_ocl_shuffle2(
    __clc_vec2_fp32_t, __clc_vec2_fp32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t __spirv_ocl_shuffle2(
    __clc_vec4_fp32_t, __clc_vec4_fp32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t __spirv_ocl_shuffle2(
    __clc_vec8_fp32_t, __clc_vec8_fp32_t, __clc_vec16_uint32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t __spirv_ocl_shuffle2(
    __clc_vec16_fp32_t, __clc_vec16_fp32_t, __clc_vec16_uint32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t __spirv_ocl_shuffle2(
    __clc_vec2_fp64_t, __clc_vec2_fp64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t __spirv_ocl_shuffle2(
    __clc_vec4_fp64_t, __clc_vec4_fp64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t __spirv_ocl_shuffle2(
    __clc_vec8_fp64_t, __clc_vec8_fp64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t __spirv_ocl_shuffle2(
    __clc_vec16_fp64_t, __clc_vec16_fp64_t, __clc_vec2_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t __spirv_ocl_shuffle2(
    __clc_vec2_fp64_t, __clc_vec2_fp64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t __spirv_ocl_shuffle2(
    __clc_vec4_fp64_t, __clc_vec4_fp64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t __spirv_ocl_shuffle2(
    __clc_vec8_fp64_t, __clc_vec8_fp64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t __spirv_ocl_shuffle2(
    __clc_vec16_fp64_t, __clc_vec16_fp64_t, __clc_vec4_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t __spirv_ocl_shuffle2(
    __clc_vec2_fp64_t, __clc_vec2_fp64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t __spirv_ocl_shuffle2(
    __clc_vec4_fp64_t, __clc_vec4_fp64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t __spirv_ocl_shuffle2(
    __clc_vec8_fp64_t, __clc_vec8_fp64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t __spirv_ocl_shuffle2(
    __clc_vec16_fp64_t, __clc_vec16_fp64_t, __clc_vec8_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t __spirv_ocl_shuffle2(
    __clc_vec2_fp64_t, __clc_vec2_fp64_t, __clc_vec16_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t __spirv_ocl_shuffle2(
    __clc_vec4_fp64_t, __clc_vec4_fp64_t, __clc_vec16_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t __spirv_ocl_shuffle2(
    __clc_vec8_fp64_t, __clc_vec8_fp64_t, __clc_vec16_uint64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t __spirv_ocl_shuffle2(
    __clc_vec16_fp64_t, __clc_vec16_fp64_t, __clc_vec16_uint64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t __spirv_ocl_shuffle2(
    __clc_vec2_fp16_t, __clc_vec2_fp16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t __spirv_ocl_shuffle2(
    __clc_vec4_fp16_t, __clc_vec4_fp16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t __spirv_ocl_shuffle2(
    __clc_vec8_fp16_t, __clc_vec8_fp16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t __spirv_ocl_shuffle2(
    __clc_vec16_fp16_t, __clc_vec16_fp16_t, __clc_vec2_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t __spirv_ocl_shuffle2(
    __clc_vec2_fp16_t, __clc_vec2_fp16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t __spirv_ocl_shuffle2(
    __clc_vec4_fp16_t, __clc_vec4_fp16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t __spirv_ocl_shuffle2(
    __clc_vec8_fp16_t, __clc_vec8_fp16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t __spirv_ocl_shuffle2(
    __clc_vec16_fp16_t, __clc_vec16_fp16_t, __clc_vec4_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t __spirv_ocl_shuffle2(
    __clc_vec2_fp16_t, __clc_vec2_fp16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t __spirv_ocl_shuffle2(
    __clc_vec4_fp16_t, __clc_vec4_fp16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t __spirv_ocl_shuffle2(
    __clc_vec8_fp16_t, __clc_vec8_fp16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t __spirv_ocl_shuffle2(
    __clc_vec16_fp16_t, __clc_vec16_fp16_t, __clc_vec8_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t __spirv_ocl_shuffle2(
    __clc_vec2_fp16_t, __clc_vec2_fp16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t __spirv_ocl_shuffle2(
    __clc_vec4_fp16_t, __clc_vec4_fp16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t __spirv_ocl_shuffle2(
    __clc_vec8_fp16_t, __clc_vec8_fp16_t, __clc_vec16_uint16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t __spirv_ocl_shuffle2(
    __clc_vec16_fp16_t, __clc_vec16_fp16_t, __clc_vec16_uint16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_sign(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_sign(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_sign(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_sign(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_sign(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_sign(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_sign(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_sign(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_sign(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_sign(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_sign(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_sign(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_sign(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_sign(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_sign(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_sign(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_sign(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_sign(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_sin(__clc_fp32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t __spirv_ocl_sin(__clc_vec2_fp32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t __spirv_ocl_sin(__clc_vec3_fp32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t __spirv_ocl_sin(__clc_vec4_fp32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t __spirv_ocl_sin(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_sin(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_sin(__clc_fp64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t __spirv_ocl_sin(__clc_vec2_fp64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t __spirv_ocl_sin(__clc_vec3_fp64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t __spirv_ocl_sin(__clc_vec4_fp64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t __spirv_ocl_sin(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_sin(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_sin(__clc_fp16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t __spirv_ocl_sin(__clc_vec2_fp16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t __spirv_ocl_sin(__clc_vec3_fp16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t __spirv_ocl_sin(__clc_vec4_fp16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t __spirv_ocl_sin(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_sin(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL __clc_fp32_t
__spirv_ocl_sincos(__clc_fp32_t, __clc_fp32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_fp32_t __spirv_ocl_sincos(__clc_fp32_t,
                                                        __clc_fp32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_fp32_t
__spirv_ocl_sincos(__clc_fp32_t, __clc_fp32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_sincos(__clc_vec2_fp32_t, __clc_vec2_fp32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_sincos(__clc_vec2_fp32_t, __clc_vec2_fp32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_sincos(__clc_vec2_fp32_t, __clc_vec2_fp32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_sincos(__clc_vec3_fp32_t, __clc_vec3_fp32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_sincos(__clc_vec3_fp32_t, __clc_vec3_fp32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_sincos(__clc_vec3_fp32_t, __clc_vec3_fp32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_sincos(__clc_vec4_fp32_t, __clc_vec4_fp32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_sincos(__clc_vec4_fp32_t, __clc_vec4_fp32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_sincos(__clc_vec4_fp32_t, __clc_vec4_fp32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_sincos(__clc_vec8_fp32_t, __clc_vec8_fp32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_sincos(__clc_vec8_fp32_t, __clc_vec8_fp32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_sincos(__clc_vec8_fp32_t, __clc_vec8_fp32_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_sincos(__clc_vec16_fp32_t, __clc_vec16_fp32_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_sincos(__clc_vec16_fp32_t, __clc_vec16_fp32_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_sincos(__clc_vec16_fp32_t, __clc_vec16_fp32_t __global *);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL __clc_fp64_t
__spirv_ocl_sincos(__clc_fp64_t, __clc_fp64_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_fp64_t __spirv_ocl_sincos(__clc_fp64_t,
                                                        __clc_fp64_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_fp64_t
__spirv_ocl_sincos(__clc_fp64_t, __clc_fp64_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_sincos(__clc_vec2_fp64_t, __clc_vec2_fp64_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_sincos(__clc_vec2_fp64_t, __clc_vec2_fp64_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_sincos(__clc_vec2_fp64_t, __clc_vec2_fp64_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_sincos(__clc_vec3_fp64_t, __clc_vec3_fp64_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_sincos(__clc_vec3_fp64_t, __clc_vec3_fp64_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_sincos(__clc_vec3_fp64_t, __clc_vec3_fp64_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_sincos(__clc_vec4_fp64_t, __clc_vec4_fp64_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_sincos(__clc_vec4_fp64_t, __clc_vec4_fp64_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_sincos(__clc_vec4_fp64_t, __clc_vec4_fp64_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_sincos(__clc_vec8_fp64_t, __clc_vec8_fp64_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_sincos(__clc_vec8_fp64_t, __clc_vec8_fp64_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_sincos(__clc_vec8_fp64_t, __clc_vec8_fp64_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_sincos(__clc_vec16_fp64_t, __clc_vec16_fp64_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_sincos(__clc_vec16_fp64_t, __clc_vec16_fp64_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_sincos(__clc_vec16_fp64_t, __clc_vec16_fp64_t __global *);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL __clc_fp16_t
__spirv_ocl_sincos(__clc_fp16_t, __clc_fp16_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_fp16_t __spirv_ocl_sincos(__clc_fp16_t,
                                                        __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_fp16_t
__spirv_ocl_sincos(__clc_fp16_t, __clc_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_sincos(__clc_vec2_fp16_t, __clc_vec2_fp16_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_sincos(__clc_vec2_fp16_t, __clc_vec2_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_sincos(__clc_vec2_fp16_t, __clc_vec2_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_sincos(__clc_vec3_fp16_t, __clc_vec3_fp16_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_sincos(__clc_vec3_fp16_t, __clc_vec3_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_sincos(__clc_vec3_fp16_t, __clc_vec3_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_sincos(__clc_vec4_fp16_t, __clc_vec4_fp16_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_sincos(__clc_vec4_fp16_t, __clc_vec4_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_sincos(__clc_vec4_fp16_t, __clc_vec4_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_sincos(__clc_vec8_fp16_t, __clc_vec8_fp16_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_sincos(__clc_vec8_fp16_t, __clc_vec8_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_sincos(__clc_vec8_fp16_t, __clc_vec8_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_sincos(__clc_vec16_fp16_t, __clc_vec16_fp16_t __private *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_sincos(__clc_vec16_fp16_t, __clc_vec16_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_sincos(__clc_vec16_fp16_t, __clc_vec16_fp16_t __global *);
#endif

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DECL __clc_fp32_t
__spirv_ocl_sincos(__clc_fp32_t, __clc_fp32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_sincos(__clc_vec2_fp32_t, __clc_vec2_fp32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_sincos(__clc_vec3_fp32_t, __clc_vec3_fp32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_sincos(__clc_vec4_fp32_t, __clc_vec4_fp32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_sincos(__clc_vec8_fp32_t, __clc_vec8_fp32_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_sincos(__clc_vec16_fp32_t, __clc_vec16_fp32_t __generic *);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL __clc_fp64_t
__spirv_ocl_sincos(__clc_fp64_t, __clc_fp64_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_sincos(__clc_vec2_fp64_t, __clc_vec2_fp64_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_sincos(__clc_vec3_fp64_t, __clc_vec3_fp64_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_sincos(__clc_vec4_fp64_t, __clc_vec4_fp64_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_sincos(__clc_vec8_fp64_t, __clc_vec8_fp64_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_sincos(__clc_vec16_fp64_t, __clc_vec16_fp64_t __generic *);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL __clc_fp16_t
__spirv_ocl_sincos(__clc_fp16_t, __clc_fp16_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_sincos(__clc_vec2_fp16_t, __clc_vec2_fp16_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_sincos(__clc_vec3_fp16_t, __clc_vec3_fp16_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_sincos(__clc_vec4_fp16_t, __clc_vec4_fp16_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_sincos(__clc_vec8_fp16_t, __clc_vec8_fp16_t __generic *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_sincos(__clc_vec16_fp16_t, __clc_vec16_fp16_t __generic *);
#endif
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_sinh(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_sinh(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_sinh(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_sinh(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_sinh(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_sinh(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_sinh(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_sinh(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_sinh(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_sinh(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_sinh(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_sinh(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_sinh(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_sinh(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_sinh(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_sinh(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_sinh(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_sinh(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_sinpi(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_sinpi(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_sinpi(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_sinpi(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_sinpi(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_sinpi(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_sinpi(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_sinpi(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_sinpi(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_sinpi(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_sinpi(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_sinpi(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_sinpi(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_sinpi(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_sinpi(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_sinpi(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_sinpi(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_sinpi(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_smoothstep(__clc_fp32_t, __clc_fp32_t, __clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t __spirv_ocl_smoothstep(
    __clc_vec2_fp32_t, __clc_vec2_fp32_t, __clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t __spirv_ocl_smoothstep(
    __clc_vec3_fp32_t, __clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t __spirv_ocl_smoothstep(
    __clc_vec4_fp32_t, __clc_vec4_fp32_t, __clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t __spirv_ocl_smoothstep(
    __clc_vec8_fp32_t, __clc_vec8_fp32_t, __clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t __spirv_ocl_smoothstep(
    __clc_vec16_fp32_t, __clc_vec16_fp32_t, __clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ocl_smoothstep(__clc_fp64_t, __clc_fp64_t, __clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t __spirv_ocl_smoothstep(
    __clc_vec2_fp64_t, __clc_vec2_fp64_t, __clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t __spirv_ocl_smoothstep(
    __clc_vec3_fp64_t, __clc_vec3_fp64_t, __clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t __spirv_ocl_smoothstep(
    __clc_vec4_fp64_t, __clc_vec4_fp64_t, __clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t __spirv_ocl_smoothstep(
    __clc_vec8_fp64_t, __clc_vec8_fp64_t, __clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t __spirv_ocl_smoothstep(
    __clc_vec16_fp64_t, __clc_vec16_fp64_t, __clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ocl_smoothstep(__clc_fp16_t, __clc_fp16_t, __clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t __spirv_ocl_smoothstep(
    __clc_vec2_fp16_t, __clc_vec2_fp16_t, __clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t __spirv_ocl_smoothstep(
    __clc_vec3_fp16_t, __clc_vec3_fp16_t, __clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t __spirv_ocl_smoothstep(
    __clc_vec4_fp16_t, __clc_vec4_fp16_t, __clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t __spirv_ocl_smoothstep(
    __clc_vec8_fp16_t, __clc_vec8_fp16_t, __clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t __spirv_ocl_smoothstep(
    __clc_vec16_fp16_t, __clc_vec16_fp16_t, __clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_sqrt(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_sqrt(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_sqrt(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_sqrt(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_sqrt(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_sqrt(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_sqrt(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_sqrt(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_sqrt(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_sqrt(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_sqrt(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_sqrt(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_sqrt(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_sqrt(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_sqrt(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_sqrt(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_sqrt(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_sqrt(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
    __spirv_ocl_step(__clc_fp32_t, __clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_step(__clc_vec2_fp32_t, __clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_step(__clc_vec3_fp32_t, __clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_step(__clc_vec4_fp32_t, __clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_step(__clc_vec8_fp32_t, __clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_step(__clc_vec16_fp32_t, __clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
    __spirv_ocl_step(__clc_fp64_t, __clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_step(__clc_vec2_fp64_t, __clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_step(__clc_vec3_fp64_t, __clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_step(__clc_vec4_fp64_t, __clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_step(__clc_vec8_fp64_t, __clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_step(__clc_vec16_fp64_t, __clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t
    __spirv_ocl_step(__clc_fp16_t, __clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_step(__clc_vec2_fp16_t, __clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_step(__clc_vec3_fp16_t, __clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_step(__clc_vec4_fp16_t, __clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_step(__clc_vec8_fp16_t, __clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_step(__clc_vec16_fp16_t, __clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_tan(__clc_fp32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t __spirv_ocl_tan(__clc_vec2_fp32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t __spirv_ocl_tan(__clc_vec3_fp32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t __spirv_ocl_tan(__clc_vec4_fp32_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t __spirv_ocl_tan(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_tan(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_tan(__clc_fp64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t __spirv_ocl_tan(__clc_vec2_fp64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t __spirv_ocl_tan(__clc_vec3_fp64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t __spirv_ocl_tan(__clc_vec4_fp64_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t __spirv_ocl_tan(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_tan(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_tan(__clc_fp16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t __spirv_ocl_tan(__clc_vec2_fp16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t __spirv_ocl_tan(__clc_vec3_fp16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t __spirv_ocl_tan(__clc_vec4_fp16_t);
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t __spirv_ocl_tan(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_tan(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_tanh(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_tanh(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_tanh(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_tanh(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_tanh(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_tanh(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_tanh(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_tanh(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_tanh(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_tanh(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_tanh(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_tanh(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_tanh(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_tanh(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_tanh(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_tanh(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_tanh(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_tanh(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __clc_native_tanh(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __clc_native_tanh(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __clc_native_tanh(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __clc_native_tanh(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __clc_native_tanh(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __clc_native_tanh(__clc_vec16_fp32_t);

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __clc_native_tanh(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __clc_native_tanh(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __clc_native_tanh(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __clc_native_tanh(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __clc_native_tanh(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __clc_native_tanh(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_tanpi(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_tanpi(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_tanpi(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_tanpi(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_tanpi(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_tanpi(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_tanpi(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_tanpi(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_tanpi(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_tanpi(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_tanpi(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_tanpi(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_tanpi(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_tanpi(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_tanpi(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_tanpi(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_tanpi(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_tanpi(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_tgamma(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_tgamma(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_tgamma(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_tgamma(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_tgamma(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_tgamma(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_tgamma(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_tgamma(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_tgamma(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_tgamma(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_tgamma(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_tgamma(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_tgamma(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_tgamma(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_tgamma(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_tgamma(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_tgamma(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_tgamma(__clc_vec16_fp16_t);
#endif

_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp32_t __spirv_ocl_trunc(__clc_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp32_t
    __spirv_ocl_trunc(__clc_vec2_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp32_t
    __spirv_ocl_trunc(__clc_vec3_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp32_t
    __spirv_ocl_trunc(__clc_vec4_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp32_t
    __spirv_ocl_trunc(__clc_vec8_fp32_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp32_t
    __spirv_ocl_trunc(__clc_vec16_fp32_t);

#ifdef cl_khr_fp64
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp64_t __spirv_ocl_trunc(__clc_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp64_t
    __spirv_ocl_trunc(__clc_vec2_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp64_t
    __spirv_ocl_trunc(__clc_vec3_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp64_t
    __spirv_ocl_trunc(__clc_vec4_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp64_t
    __spirv_ocl_trunc(__clc_vec8_fp64_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp64_t
    __spirv_ocl_trunc(__clc_vec16_fp64_t);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD
_CLC_DECL _CLC_CONSTFN __clc_fp16_t __spirv_ocl_trunc(__clc_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec2_fp16_t
    __spirv_ocl_trunc(__clc_vec2_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec3_fp16_t
    __spirv_ocl_trunc(__clc_vec3_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec4_fp16_t
    __spirv_ocl_trunc(__clc_vec4_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec8_fp16_t
    __spirv_ocl_trunc(__clc_vec8_fp16_t);
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_vec16_fp16_t
    __spirv_ocl_trunc(__clc_vec16_fp16_t);
#endif

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

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL __clc_fp32_t
__spirv_ocl_vload_half(__clc_size_t, __clc_fp16_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_fp32_t
__spirv_ocl_vload_half(__clc_size_t, __clc_fp16_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_fp32_t
__spirv_ocl_vload_half(__clc_size_t, __clc_fp16_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_fp32_t
__spirv_ocl_vload_half(__clc_size_t, __clc_fp16_t const __constant *);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_vload_halfn_Rfloat16(__clc_size_t, __clc_fp16_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_vload_halfn_Rfloat16(__clc_size_t, __clc_fp16_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_vload_halfn_Rfloat16(__clc_size_t, __clc_fp16_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_vload_halfn_Rfloat16(__clc_size_t, __clc_fp16_t const __constant *);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_vload_halfn_Rfloat2(__clc_size_t, __clc_fp16_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_vload_halfn_Rfloat2(__clc_size_t, __clc_fp16_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_vload_halfn_Rfloat2(__clc_size_t, __clc_fp16_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_vload_halfn_Rfloat2(__clc_size_t, __clc_fp16_t const __constant *);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_vload_halfn_Rfloat3(__clc_size_t, __clc_fp16_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_vload_halfn_Rfloat3(__clc_size_t, __clc_fp16_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_vload_halfn_Rfloat3(__clc_size_t, __clc_fp16_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_vload_halfn_Rfloat3(__clc_size_t, __clc_fp16_t const __constant *);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_vload_halfn_Rfloat4(__clc_size_t, __clc_fp16_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_vload_halfn_Rfloat4(__clc_size_t, __clc_fp16_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_vload_halfn_Rfloat4(__clc_size_t, __clc_fp16_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_vload_halfn_Rfloat4(__clc_size_t, __clc_fp16_t const __constant *);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_vload_halfn_Rfloat8(__clc_size_t, __clc_fp16_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_vload_halfn_Rfloat8(__clc_size_t, __clc_fp16_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_vload_halfn_Rfloat8(__clc_size_t, __clc_fp16_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_vload_halfn_Rfloat8(__clc_size_t, __clc_fp16_t const __constant *);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_vloada_halfn_Rfloat16(__clc_size_t, __clc_fp16_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_vloada_halfn_Rfloat16(__clc_size_t, __clc_fp16_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_vloada_halfn_Rfloat16(__clc_size_t, __clc_fp16_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t __spirv_ocl_vloada_halfn_Rfloat16(
    __clc_size_t, __clc_fp16_t const __constant *);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_vloada_halfn_Rfloat2(__clc_size_t, __clc_fp16_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_vloada_halfn_Rfloat2(__clc_size_t, __clc_fp16_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_vloada_halfn_Rfloat2(__clc_size_t, __clc_fp16_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_vloada_halfn_Rfloat2(__clc_size_t, __clc_fp16_t const __constant *);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_vloada_halfn_Rfloat3(__clc_size_t, __clc_fp16_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_vloada_halfn_Rfloat3(__clc_size_t, __clc_fp16_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_vloada_halfn_Rfloat3(__clc_size_t, __clc_fp16_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_vloada_halfn_Rfloat3(__clc_size_t, __clc_fp16_t const __constant *);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_vloada_halfn_Rfloat4(__clc_size_t, __clc_fp16_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_vloada_halfn_Rfloat4(__clc_size_t, __clc_fp16_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_vloada_halfn_Rfloat4(__clc_size_t, __clc_fp16_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_vloada_halfn_Rfloat4(__clc_size_t, __clc_fp16_t const __constant *);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_vloada_halfn_Rfloat8(__clc_size_t, __clc_fp16_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_vloada_halfn_Rfloat8(__clc_size_t, __clc_fp16_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_vloada_halfn_Rfloat8(__clc_size_t, __clc_fp16_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_vloada_halfn_Rfloat8(__clc_size_t, __clc_fp16_t const __constant *);
#endif

_CLC_OVERLOAD _CLC_DECL __clc_vec16_char_t
__spirv_ocl_vloadn_Rchar16(__clc_size_t, __clc_char_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_char_t
__spirv_ocl_vloadn_Rchar16(__clc_size_t, __clc_char_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_char_t
__spirv_ocl_vloadn_Rchar16(__clc_size_t, __clc_char_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_char_t
__spirv_ocl_vloadn_Rchar16(__clc_size_t, __clc_char_t const __constant *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_int8_t
__spirv_ocl_vloadn_Rchar16(__clc_size_t, __clc_int8_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_int8_t
__spirv_ocl_vloadn_Rchar16(__clc_size_t, __clc_int8_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_int8_t
__spirv_ocl_vloadn_Rchar16(__clc_size_t, __clc_int8_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_int8_t
__spirv_ocl_vloadn_Rchar16(__clc_size_t, __clc_int8_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec2_char_t
__spirv_ocl_vloadn_Rchar2(__clc_size_t, __clc_char_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_char_t
__spirv_ocl_vloadn_Rchar2(__clc_size_t, __clc_char_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_char_t
__spirv_ocl_vloadn_Rchar2(__clc_size_t, __clc_char_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_char_t
__spirv_ocl_vloadn_Rchar2(__clc_size_t, __clc_char_t const __constant *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_int8_t
__spirv_ocl_vloadn_Rchar2(__clc_size_t, __clc_int8_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_int8_t
__spirv_ocl_vloadn_Rchar2(__clc_size_t, __clc_int8_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_int8_t
__spirv_ocl_vloadn_Rchar2(__clc_size_t, __clc_int8_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_int8_t
__spirv_ocl_vloadn_Rchar2(__clc_size_t, __clc_int8_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec3_char_t
__spirv_ocl_vloadn_Rchar3(__clc_size_t, __clc_char_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_char_t
__spirv_ocl_vloadn_Rchar3(__clc_size_t, __clc_char_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_char_t
__spirv_ocl_vloadn_Rchar3(__clc_size_t, __clc_char_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_char_t
__spirv_ocl_vloadn_Rchar3(__clc_size_t, __clc_char_t const __constant *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_int8_t
__spirv_ocl_vloadn_Rchar3(__clc_size_t, __clc_int8_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_int8_t
__spirv_ocl_vloadn_Rchar3(__clc_size_t, __clc_int8_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_int8_t
__spirv_ocl_vloadn_Rchar3(__clc_size_t, __clc_int8_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_int8_t
__spirv_ocl_vloadn_Rchar3(__clc_size_t, __clc_int8_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec4_char_t
__spirv_ocl_vloadn_Rchar4(__clc_size_t, __clc_char_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_char_t
__spirv_ocl_vloadn_Rchar4(__clc_size_t, __clc_char_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_char_t
__spirv_ocl_vloadn_Rchar4(__clc_size_t, __clc_char_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_char_t
__spirv_ocl_vloadn_Rchar4(__clc_size_t, __clc_char_t const __constant *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_int8_t
__spirv_ocl_vloadn_Rchar4(__clc_size_t, __clc_int8_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_int8_t
__spirv_ocl_vloadn_Rchar4(__clc_size_t, __clc_int8_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_int8_t
__spirv_ocl_vloadn_Rchar4(__clc_size_t, __clc_int8_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_int8_t
__spirv_ocl_vloadn_Rchar4(__clc_size_t, __clc_int8_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec8_char_t
__spirv_ocl_vloadn_Rchar8(__clc_size_t, __clc_char_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_char_t
__spirv_ocl_vloadn_Rchar8(__clc_size_t, __clc_char_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_char_t
__spirv_ocl_vloadn_Rchar8(__clc_size_t, __clc_char_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_char_t
__spirv_ocl_vloadn_Rchar8(__clc_size_t, __clc_char_t const __constant *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_int8_t
__spirv_ocl_vloadn_Rchar8(__clc_size_t, __clc_int8_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_int8_t
__spirv_ocl_vloadn_Rchar8(__clc_size_t, __clc_int8_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_int8_t
__spirv_ocl_vloadn_Rchar8(__clc_size_t, __clc_int8_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_int8_t
__spirv_ocl_vloadn_Rchar8(__clc_size_t, __clc_int8_t const __constant *);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_vloadn_Rdouble16(__clc_size_t, __clc_fp64_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_vloadn_Rdouble16(__clc_size_t, __clc_fp64_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_vloadn_Rdouble16(__clc_size_t, __clc_fp64_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_vloadn_Rdouble16(__clc_size_t, __clc_fp64_t const __constant *);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_vloadn_Rdouble2(__clc_size_t, __clc_fp64_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_vloadn_Rdouble2(__clc_size_t, __clc_fp64_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_vloadn_Rdouble2(__clc_size_t, __clc_fp64_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_vloadn_Rdouble2(__clc_size_t, __clc_fp64_t const __constant *);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_vloadn_Rdouble3(__clc_size_t, __clc_fp64_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_vloadn_Rdouble3(__clc_size_t, __clc_fp64_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_vloadn_Rdouble3(__clc_size_t, __clc_fp64_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_vloadn_Rdouble3(__clc_size_t, __clc_fp64_t const __constant *);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_vloadn_Rdouble4(__clc_size_t, __clc_fp64_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_vloadn_Rdouble4(__clc_size_t, __clc_fp64_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_vloadn_Rdouble4(__clc_size_t, __clc_fp64_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_vloadn_Rdouble4(__clc_size_t, __clc_fp64_t const __constant *);
#endif

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_vloadn_Rdouble8(__clc_size_t, __clc_fp64_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_vloadn_Rdouble8(__clc_size_t, __clc_fp64_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_vloadn_Rdouble8(__clc_size_t, __clc_fp64_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_vloadn_Rdouble8(__clc_size_t, __clc_fp64_t const __constant *);
#endif

_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_vloadn_Rfloat16(__clc_size_t, __clc_fp32_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_vloadn_Rfloat16(__clc_size_t, __clc_fp32_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_vloadn_Rfloat16(__clc_size_t, __clc_fp32_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_vloadn_Rfloat16(__clc_size_t, __clc_fp32_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_vloadn_Rfloat2(__clc_size_t, __clc_fp32_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_vloadn_Rfloat2(__clc_size_t, __clc_fp32_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_vloadn_Rfloat2(__clc_size_t, __clc_fp32_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_vloadn_Rfloat2(__clc_size_t, __clc_fp32_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_vloadn_Rfloat3(__clc_size_t, __clc_fp32_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_vloadn_Rfloat3(__clc_size_t, __clc_fp32_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_vloadn_Rfloat3(__clc_size_t, __clc_fp32_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_vloadn_Rfloat3(__clc_size_t, __clc_fp32_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_vloadn_Rfloat4(__clc_size_t, __clc_fp32_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_vloadn_Rfloat4(__clc_size_t, __clc_fp32_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_vloadn_Rfloat4(__clc_size_t, __clc_fp32_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_vloadn_Rfloat4(__clc_size_t, __clc_fp32_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_vloadn_Rfloat8(__clc_size_t, __clc_fp32_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_vloadn_Rfloat8(__clc_size_t, __clc_fp32_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_vloadn_Rfloat8(__clc_size_t, __clc_fp32_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_vloadn_Rfloat8(__clc_size_t, __clc_fp32_t const __constant *);

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_vloadn_Rhalf16(__clc_size_t, __clc_fp16_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_vloadn_Rhalf16(__clc_size_t, __clc_fp16_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_vloadn_Rhalf16(__clc_size_t, __clc_fp16_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_vloadn_Rhalf16(__clc_size_t, __clc_fp16_t const __constant *);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_vloadn_Rhalf2(__clc_size_t, __clc_fp16_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_vloadn_Rhalf2(__clc_size_t, __clc_fp16_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_vloadn_Rhalf2(__clc_size_t, __clc_fp16_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_vloadn_Rhalf2(__clc_size_t, __clc_fp16_t const __constant *);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_vloadn_Rhalf3(__clc_size_t, __clc_fp16_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_vloadn_Rhalf3(__clc_size_t, __clc_fp16_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_vloadn_Rhalf3(__clc_size_t, __clc_fp16_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_vloadn_Rhalf3(__clc_size_t, __clc_fp16_t const __constant *);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_vloadn_Rhalf4(__clc_size_t, __clc_fp16_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_vloadn_Rhalf4(__clc_size_t, __clc_fp16_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_vloadn_Rhalf4(__clc_size_t, __clc_fp16_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_vloadn_Rhalf4(__clc_size_t, __clc_fp16_t const __constant *);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_vloadn_Rhalf8(__clc_size_t, __clc_fp16_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_vloadn_Rhalf8(__clc_size_t, __clc_fp16_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_vloadn_Rhalf8(__clc_size_t, __clc_fp16_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_vloadn_Rhalf8(__clc_size_t, __clc_fp16_t const __constant *);
#endif

_CLC_OVERLOAD _CLC_DECL __clc_vec16_int32_t
__spirv_ocl_vloadn_Rint16(__clc_size_t, __clc_int32_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_int32_t
__spirv_ocl_vloadn_Rint16(__clc_size_t, __clc_int32_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_int32_t
__spirv_ocl_vloadn_Rint16(__clc_size_t, __clc_int32_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_int32_t
__spirv_ocl_vloadn_Rint16(__clc_size_t, __clc_int32_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec2_int32_t
__spirv_ocl_vloadn_Rint2(__clc_size_t, __clc_int32_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_int32_t
__spirv_ocl_vloadn_Rint2(__clc_size_t, __clc_int32_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_int32_t
__spirv_ocl_vloadn_Rint2(__clc_size_t, __clc_int32_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_int32_t
__spirv_ocl_vloadn_Rint2(__clc_size_t, __clc_int32_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec3_int32_t
__spirv_ocl_vloadn_Rint3(__clc_size_t, __clc_int32_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_int32_t
__spirv_ocl_vloadn_Rint3(__clc_size_t, __clc_int32_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_int32_t
__spirv_ocl_vloadn_Rint3(__clc_size_t, __clc_int32_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_int32_t
__spirv_ocl_vloadn_Rint3(__clc_size_t, __clc_int32_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec4_int32_t
__spirv_ocl_vloadn_Rint4(__clc_size_t, __clc_int32_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_int32_t
__spirv_ocl_vloadn_Rint4(__clc_size_t, __clc_int32_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_int32_t
__spirv_ocl_vloadn_Rint4(__clc_size_t, __clc_int32_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_int32_t
__spirv_ocl_vloadn_Rint4(__clc_size_t, __clc_int32_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec8_int32_t
__spirv_ocl_vloadn_Rint8(__clc_size_t, __clc_int32_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_int32_t
__spirv_ocl_vloadn_Rint8(__clc_size_t, __clc_int32_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_int32_t
__spirv_ocl_vloadn_Rint8(__clc_size_t, __clc_int32_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_int32_t
__spirv_ocl_vloadn_Rint8(__clc_size_t, __clc_int32_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec16_int64_t
__spirv_ocl_vloadn_Rlong16(__clc_size_t, __clc_int64_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_int64_t
__spirv_ocl_vloadn_Rlong16(__clc_size_t, __clc_int64_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_int64_t
__spirv_ocl_vloadn_Rlong16(__clc_size_t, __clc_int64_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_int64_t
__spirv_ocl_vloadn_Rlong16(__clc_size_t, __clc_int64_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec2_int64_t
__spirv_ocl_vloadn_Rlong2(__clc_size_t, __clc_int64_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_int64_t
__spirv_ocl_vloadn_Rlong2(__clc_size_t, __clc_int64_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_int64_t
__spirv_ocl_vloadn_Rlong2(__clc_size_t, __clc_int64_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_int64_t
__spirv_ocl_vloadn_Rlong2(__clc_size_t, __clc_int64_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec3_int64_t
__spirv_ocl_vloadn_Rlong3(__clc_size_t, __clc_int64_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_int64_t
__spirv_ocl_vloadn_Rlong3(__clc_size_t, __clc_int64_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_int64_t
__spirv_ocl_vloadn_Rlong3(__clc_size_t, __clc_int64_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_int64_t
__spirv_ocl_vloadn_Rlong3(__clc_size_t, __clc_int64_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec4_int64_t
__spirv_ocl_vloadn_Rlong4(__clc_size_t, __clc_int64_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_int64_t
__spirv_ocl_vloadn_Rlong4(__clc_size_t, __clc_int64_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_int64_t
__spirv_ocl_vloadn_Rlong4(__clc_size_t, __clc_int64_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_int64_t
__spirv_ocl_vloadn_Rlong4(__clc_size_t, __clc_int64_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec8_int64_t
__spirv_ocl_vloadn_Rlong8(__clc_size_t, __clc_int64_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_int64_t
__spirv_ocl_vloadn_Rlong8(__clc_size_t, __clc_int64_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_int64_t
__spirv_ocl_vloadn_Rlong8(__clc_size_t, __clc_int64_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_int64_t
__spirv_ocl_vloadn_Rlong8(__clc_size_t, __clc_int64_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec16_int16_t
__spirv_ocl_vloadn_Rshort16(__clc_size_t, __clc_int16_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_int16_t
__spirv_ocl_vloadn_Rshort16(__clc_size_t, __clc_int16_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_int16_t
__spirv_ocl_vloadn_Rshort16(__clc_size_t, __clc_int16_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_int16_t
__spirv_ocl_vloadn_Rshort16(__clc_size_t, __clc_int16_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec2_int16_t
__spirv_ocl_vloadn_Rshort2(__clc_size_t, __clc_int16_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_int16_t
__spirv_ocl_vloadn_Rshort2(__clc_size_t, __clc_int16_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_int16_t
__spirv_ocl_vloadn_Rshort2(__clc_size_t, __clc_int16_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_int16_t
__spirv_ocl_vloadn_Rshort2(__clc_size_t, __clc_int16_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec3_int16_t
__spirv_ocl_vloadn_Rshort3(__clc_size_t, __clc_int16_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_int16_t
__spirv_ocl_vloadn_Rshort3(__clc_size_t, __clc_int16_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_int16_t
__spirv_ocl_vloadn_Rshort3(__clc_size_t, __clc_int16_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_int16_t
__spirv_ocl_vloadn_Rshort3(__clc_size_t, __clc_int16_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec4_int16_t
__spirv_ocl_vloadn_Rshort4(__clc_size_t, __clc_int16_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_int16_t
__spirv_ocl_vloadn_Rshort4(__clc_size_t, __clc_int16_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_int16_t
__spirv_ocl_vloadn_Rshort4(__clc_size_t, __clc_int16_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_int16_t
__spirv_ocl_vloadn_Rshort4(__clc_size_t, __clc_int16_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec8_int16_t
__spirv_ocl_vloadn_Rshort8(__clc_size_t, __clc_int16_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_int16_t
__spirv_ocl_vloadn_Rshort8(__clc_size_t, __clc_int16_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_int16_t
__spirv_ocl_vloadn_Rshort8(__clc_size_t, __clc_int16_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_int16_t
__spirv_ocl_vloadn_Rshort8(__clc_size_t, __clc_int16_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec16_uint8_t
__spirv_ocl_vloadn_Ruchar16(__clc_size_t, __clc_uint8_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_uint8_t
__spirv_ocl_vloadn_Ruchar16(__clc_size_t, __clc_uint8_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_uint8_t
__spirv_ocl_vloadn_Ruchar16(__clc_size_t, __clc_uint8_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_uint8_t
__spirv_ocl_vloadn_Ruchar16(__clc_size_t, __clc_uint8_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec2_uint8_t
__spirv_ocl_vloadn_Ruchar2(__clc_size_t, __clc_uint8_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_uint8_t
__spirv_ocl_vloadn_Ruchar2(__clc_size_t, __clc_uint8_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_uint8_t
__spirv_ocl_vloadn_Ruchar2(__clc_size_t, __clc_uint8_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_uint8_t
__spirv_ocl_vloadn_Ruchar2(__clc_size_t, __clc_uint8_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec3_uint8_t
__spirv_ocl_vloadn_Ruchar3(__clc_size_t, __clc_uint8_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_uint8_t
__spirv_ocl_vloadn_Ruchar3(__clc_size_t, __clc_uint8_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_uint8_t
__spirv_ocl_vloadn_Ruchar3(__clc_size_t, __clc_uint8_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_uint8_t
__spirv_ocl_vloadn_Ruchar3(__clc_size_t, __clc_uint8_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec4_uint8_t
__spirv_ocl_vloadn_Ruchar4(__clc_size_t, __clc_uint8_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_uint8_t
__spirv_ocl_vloadn_Ruchar4(__clc_size_t, __clc_uint8_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_uint8_t
__spirv_ocl_vloadn_Ruchar4(__clc_size_t, __clc_uint8_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_uint8_t
__spirv_ocl_vloadn_Ruchar4(__clc_size_t, __clc_uint8_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec8_uint8_t
__spirv_ocl_vloadn_Ruchar8(__clc_size_t, __clc_uint8_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_uint8_t
__spirv_ocl_vloadn_Ruchar8(__clc_size_t, __clc_uint8_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_uint8_t
__spirv_ocl_vloadn_Ruchar8(__clc_size_t, __clc_uint8_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_uint8_t
__spirv_ocl_vloadn_Ruchar8(__clc_size_t, __clc_uint8_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec16_uint32_t
__spirv_ocl_vloadn_Ruint16(__clc_size_t, __clc_uint32_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_uint32_t
__spirv_ocl_vloadn_Ruint16(__clc_size_t, __clc_uint32_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_uint32_t
__spirv_ocl_vloadn_Ruint16(__clc_size_t, __clc_uint32_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_uint32_t
__spirv_ocl_vloadn_Ruint16(__clc_size_t, __clc_uint32_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec2_uint32_t
__spirv_ocl_vloadn_Ruint2(__clc_size_t, __clc_uint32_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_uint32_t
__spirv_ocl_vloadn_Ruint2(__clc_size_t, __clc_uint32_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_uint32_t
__spirv_ocl_vloadn_Ruint2(__clc_size_t, __clc_uint32_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_uint32_t
__spirv_ocl_vloadn_Ruint2(__clc_size_t, __clc_uint32_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec3_uint32_t
__spirv_ocl_vloadn_Ruint3(__clc_size_t, __clc_uint32_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_uint32_t
__spirv_ocl_vloadn_Ruint3(__clc_size_t, __clc_uint32_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_uint32_t
__spirv_ocl_vloadn_Ruint3(__clc_size_t, __clc_uint32_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_uint32_t
__spirv_ocl_vloadn_Ruint3(__clc_size_t, __clc_uint32_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec4_uint32_t
__spirv_ocl_vloadn_Ruint4(__clc_size_t, __clc_uint32_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_uint32_t
__spirv_ocl_vloadn_Ruint4(__clc_size_t, __clc_uint32_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_uint32_t
__spirv_ocl_vloadn_Ruint4(__clc_size_t, __clc_uint32_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_uint32_t
__spirv_ocl_vloadn_Ruint4(__clc_size_t, __clc_uint32_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec8_uint32_t
__spirv_ocl_vloadn_Ruint8(__clc_size_t, __clc_uint32_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_uint32_t
__spirv_ocl_vloadn_Ruint8(__clc_size_t, __clc_uint32_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_uint32_t
__spirv_ocl_vloadn_Ruint8(__clc_size_t, __clc_uint32_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_uint32_t
__spirv_ocl_vloadn_Ruint8(__clc_size_t, __clc_uint32_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec16_uint64_t
__spirv_ocl_vloadn_Rulong16(__clc_size_t, __clc_uint64_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_uint64_t
__spirv_ocl_vloadn_Rulong16(__clc_size_t, __clc_uint64_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_uint64_t
__spirv_ocl_vloadn_Rulong16(__clc_size_t, __clc_uint64_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_uint64_t
__spirv_ocl_vloadn_Rulong16(__clc_size_t, __clc_uint64_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec2_uint64_t
__spirv_ocl_vloadn_Rulong2(__clc_size_t, __clc_uint64_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_uint64_t
__spirv_ocl_vloadn_Rulong2(__clc_size_t, __clc_uint64_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_uint64_t
__spirv_ocl_vloadn_Rulong2(__clc_size_t, __clc_uint64_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_uint64_t
__spirv_ocl_vloadn_Rulong2(__clc_size_t, __clc_uint64_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec3_uint64_t
__spirv_ocl_vloadn_Rulong3(__clc_size_t, __clc_uint64_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_uint64_t
__spirv_ocl_vloadn_Rulong3(__clc_size_t, __clc_uint64_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_uint64_t
__spirv_ocl_vloadn_Rulong3(__clc_size_t, __clc_uint64_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_uint64_t
__spirv_ocl_vloadn_Rulong3(__clc_size_t, __clc_uint64_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec4_uint64_t
__spirv_ocl_vloadn_Rulong4(__clc_size_t, __clc_uint64_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_uint64_t
__spirv_ocl_vloadn_Rulong4(__clc_size_t, __clc_uint64_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_uint64_t
__spirv_ocl_vloadn_Rulong4(__clc_size_t, __clc_uint64_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_uint64_t
__spirv_ocl_vloadn_Rulong4(__clc_size_t, __clc_uint64_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec8_uint64_t
__spirv_ocl_vloadn_Rulong8(__clc_size_t, __clc_uint64_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_uint64_t
__spirv_ocl_vloadn_Rulong8(__clc_size_t, __clc_uint64_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_uint64_t
__spirv_ocl_vloadn_Rulong8(__clc_size_t, __clc_uint64_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_uint64_t
__spirv_ocl_vloadn_Rulong8(__clc_size_t, __clc_uint64_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec16_uint16_t
__spirv_ocl_vloadn_Rushort16(__clc_size_t, __clc_uint16_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_uint16_t
__spirv_ocl_vloadn_Rushort16(__clc_size_t, __clc_uint16_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_uint16_t
__spirv_ocl_vloadn_Rushort16(__clc_size_t, __clc_uint16_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec16_uint16_t
__spirv_ocl_vloadn_Rushort16(__clc_size_t, __clc_uint16_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec2_uint16_t
__spirv_ocl_vloadn_Rushort2(__clc_size_t, __clc_uint16_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_uint16_t
__spirv_ocl_vloadn_Rushort2(__clc_size_t, __clc_uint16_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_uint16_t
__spirv_ocl_vloadn_Rushort2(__clc_size_t, __clc_uint16_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec2_uint16_t
__spirv_ocl_vloadn_Rushort2(__clc_size_t, __clc_uint16_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec3_uint16_t
__spirv_ocl_vloadn_Rushort3(__clc_size_t, __clc_uint16_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_uint16_t
__spirv_ocl_vloadn_Rushort3(__clc_size_t, __clc_uint16_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_uint16_t
__spirv_ocl_vloadn_Rushort3(__clc_size_t, __clc_uint16_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec3_uint16_t
__spirv_ocl_vloadn_Rushort3(__clc_size_t, __clc_uint16_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec4_uint16_t
__spirv_ocl_vloadn_Rushort4(__clc_size_t, __clc_uint16_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_uint16_t
__spirv_ocl_vloadn_Rushort4(__clc_size_t, __clc_uint16_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_uint16_t
__spirv_ocl_vloadn_Rushort4(__clc_size_t, __clc_uint16_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec4_uint16_t
__spirv_ocl_vloadn_Rushort4(__clc_size_t, __clc_uint16_t const __constant *);

_CLC_OVERLOAD _CLC_DECL __clc_vec8_uint16_t
__spirv_ocl_vloadn_Rushort8(__clc_size_t, __clc_uint16_t const *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_uint16_t
__spirv_ocl_vloadn_Rushort8(__clc_size_t, __clc_uint16_t const __local *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_uint16_t
__spirv_ocl_vloadn_Rushort8(__clc_size_t, __clc_uint16_t const __global *);
_CLC_OVERLOAD _CLC_DECL __clc_vec8_uint16_t
__spirv_ocl_vloadn_Rushort8(__clc_size_t, __clc_uint16_t const __constant *);

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_half(__clc_fp32_t, __clc_size_t,
                                                     __clc_fp16_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_half(__clc_fp32_t, __clc_size_t,
                                                     __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_half(__clc_fp32_t, __clc_size_t,
                                                     __clc_fp16_t __global *);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_half(__clc_fp64_t, __clc_size_t,
                                                     __clc_fp16_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_half(__clc_fp64_t, __clc_size_t,
                                                     __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_half(__clc_fp64_t, __clc_size_t,
                                                     __clc_fp16_t __global *);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_half_r(__clc_fp32_t,
                                                       __clc_size_t,
                                                       __clc_fp16_t *,
                                                       __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_half_r(__clc_fp32_t,
                                                       __clc_size_t,
                                                       __clc_fp16_t __local *,
                                                       __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_half_r(__clc_fp32_t,
                                                       __clc_size_t,
                                                       __clc_fp16_t __global *,
                                                       __clc_uint32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_half_r(__clc_fp64_t,
                                                       __clc_size_t,
                                                       __clc_fp16_t *,
                                                       __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_half_r(__clc_fp64_t,
                                                       __clc_size_t,
                                                       __clc_fp16_t __local *,
                                                       __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_half_r(__clc_fp64_t,
                                                       __clc_size_t,
                                                       __clc_fp16_t __global *,
                                                       __clc_uint32_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec2_fp32_t, __clc_size_t, __clc_fp16_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn(__clc_vec2_fp32_t,
                                                      __clc_size_t,
                                                      __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn(__clc_vec2_fp32_t,
                                                      __clc_size_t,
                                                      __clc_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec3_fp32_t, __clc_size_t, __clc_fp16_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn(__clc_vec3_fp32_t,
                                                      __clc_size_t,
                                                      __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn(__clc_vec3_fp32_t,
                                                      __clc_size_t,
                                                      __clc_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec4_fp32_t, __clc_size_t, __clc_fp16_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn(__clc_vec4_fp32_t,
                                                      __clc_size_t,
                                                      __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn(__clc_vec4_fp32_t,
                                                      __clc_size_t,
                                                      __clc_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec8_fp32_t, __clc_size_t, __clc_fp16_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn(__clc_vec8_fp32_t,
                                                      __clc_size_t,
                                                      __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn(__clc_vec8_fp32_t,
                                                      __clc_size_t,
                                                      __clc_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec16_fp32_t, __clc_size_t, __clc_fp16_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn(__clc_vec16_fp32_t,
                                                      __clc_size_t,
                                                      __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn(__clc_vec16_fp32_t,
                                                      __clc_size_t,
                                                      __clc_fp16_t __global *);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec2_fp64_t, __clc_size_t, __clc_fp16_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn(__clc_vec2_fp64_t,
                                                      __clc_size_t,
                                                      __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn(__clc_vec2_fp64_t,
                                                      __clc_size_t,
                                                      __clc_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec3_fp64_t, __clc_size_t, __clc_fp16_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn(__clc_vec3_fp64_t,
                                                      __clc_size_t,
                                                      __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn(__clc_vec3_fp64_t,
                                                      __clc_size_t,
                                                      __clc_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec4_fp64_t, __clc_size_t, __clc_fp16_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn(__clc_vec4_fp64_t,
                                                      __clc_size_t,
                                                      __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn(__clc_vec4_fp64_t,
                                                      __clc_size_t,
                                                      __clc_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec8_fp64_t, __clc_size_t, __clc_fp16_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn(__clc_vec8_fp64_t,
                                                      __clc_size_t,
                                                      __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn(__clc_vec8_fp64_t,
                                                      __clc_size_t,
                                                      __clc_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec16_fp64_t, __clc_size_t, __clc_fp16_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn(__clc_vec16_fp64_t,
                                                      __clc_size_t,
                                                      __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn(__clc_vec16_fp64_t,
                                                      __clc_size_t,
                                                      __clc_fp16_t __global *);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec2_fp32_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t *,
                                                        __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec2_fp32_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t __local *,
                                                        __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec2_fp32_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t __global *,
                                                        __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec3_fp32_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t *,
                                                        __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec3_fp32_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t __local *,
                                                        __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec3_fp32_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t __global *,
                                                        __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec4_fp32_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t *,
                                                        __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec4_fp32_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t __local *,
                                                        __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec4_fp32_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t __global *,
                                                        __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec8_fp32_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t *,
                                                        __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec8_fp32_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t __local *,
                                                        __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec8_fp32_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t __global *,
                                                        __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec16_fp32_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t *,
                                                        __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec16_fp32_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t __local *,
                                                        __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec16_fp32_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t __global *,
                                                        __clc_uint32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec2_fp64_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t *,
                                                        __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec2_fp64_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t __local *,
                                                        __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec2_fp64_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t __global *,
                                                        __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec3_fp64_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t *,
                                                        __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec3_fp64_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t __local *,
                                                        __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec3_fp64_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t __global *,
                                                        __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec4_fp64_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t *,
                                                        __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec4_fp64_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t __local *,
                                                        __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec4_fp64_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t __global *,
                                                        __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec8_fp64_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t *,
                                                        __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec8_fp64_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t __local *,
                                                        __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec8_fp64_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t __global *,
                                                        __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec16_fp64_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t *,
                                                        __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec16_fp64_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t __local *,
                                                        __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstore_halfn_r(__clc_vec16_fp64_t,
                                                        __clc_size_t,
                                                        __clc_fp16_t __global *,
                                                        __clc_uint32_t);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec2_fp32_t, __clc_size_t, __clc_fp16_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn(__clc_vec2_fp32_t,
                                                       __clc_size_t,
                                                       __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn(__clc_vec2_fp32_t,
                                                       __clc_size_t,
                                                       __clc_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec3_fp32_t, __clc_size_t, __clc_fp16_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn(__clc_vec3_fp32_t,
                                                       __clc_size_t,
                                                       __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn(__clc_vec3_fp32_t,
                                                       __clc_size_t,
                                                       __clc_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec4_fp32_t, __clc_size_t, __clc_fp16_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn(__clc_vec4_fp32_t,
                                                       __clc_size_t,
                                                       __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn(__clc_vec4_fp32_t,
                                                       __clc_size_t,
                                                       __clc_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec8_fp32_t, __clc_size_t, __clc_fp16_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn(__clc_vec8_fp32_t,
                                                       __clc_size_t,
                                                       __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn(__clc_vec8_fp32_t,
                                                       __clc_size_t,
                                                       __clc_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec16_fp32_t, __clc_size_t, __clc_fp16_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn(__clc_vec16_fp32_t,
                                                       __clc_size_t,
                                                       __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn(__clc_vec16_fp32_t,
                                                       __clc_size_t,
                                                       __clc_fp16_t __global *);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec2_fp64_t, __clc_size_t, __clc_fp16_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn(__clc_vec2_fp64_t,
                                                       __clc_size_t,
                                                       __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn(__clc_vec2_fp64_t,
                                                       __clc_size_t,
                                                       __clc_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec3_fp64_t, __clc_size_t, __clc_fp16_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn(__clc_vec3_fp64_t,
                                                       __clc_size_t,
                                                       __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn(__clc_vec3_fp64_t,
                                                       __clc_size_t,
                                                       __clc_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec4_fp64_t, __clc_size_t, __clc_fp16_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn(__clc_vec4_fp64_t,
                                                       __clc_size_t,
                                                       __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn(__clc_vec4_fp64_t,
                                                       __clc_size_t,
                                                       __clc_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec8_fp64_t, __clc_size_t, __clc_fp16_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn(__clc_vec8_fp64_t,
                                                       __clc_size_t,
                                                       __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn(__clc_vec8_fp64_t,
                                                       __clc_size_t,
                                                       __clc_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec16_fp64_t, __clc_size_t, __clc_fp16_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn(__clc_vec16_fp64_t,
                                                       __clc_size_t,
                                                       __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn(__clc_vec16_fp64_t,
                                                       __clc_size_t,
                                                       __clc_fp16_t __global *);
#endif
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn_r(__clc_vec2_fp32_t,
                                                         __clc_size_t,
                                                         __clc_fp16_t *,
                                                         __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn_r(__clc_vec2_fp32_t,
                                                         __clc_size_t,
                                                         __clc_fp16_t __local *,
                                                         __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec2_fp32_t, __clc_size_t,
                            __clc_fp16_t __global *, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn_r(__clc_vec3_fp32_t,
                                                         __clc_size_t,
                                                         __clc_fp16_t *,
                                                         __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn_r(__clc_vec3_fp32_t,
                                                         __clc_size_t,
                                                         __clc_fp16_t __local *,
                                                         __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec3_fp32_t, __clc_size_t,
                            __clc_fp16_t __global *, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn_r(__clc_vec4_fp32_t,
                                                         __clc_size_t,
                                                         __clc_fp16_t *,
                                                         __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn_r(__clc_vec4_fp32_t,
                                                         __clc_size_t,
                                                         __clc_fp16_t __local *,
                                                         __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec4_fp32_t, __clc_size_t,
                            __clc_fp16_t __global *, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn_r(__clc_vec8_fp32_t,
                                                         __clc_size_t,
                                                         __clc_fp16_t *,
                                                         __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn_r(__clc_vec8_fp32_t,
                                                         __clc_size_t,
                                                         __clc_fp16_t __local *,
                                                         __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec8_fp32_t, __clc_size_t,
                            __clc_fp16_t __global *, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn_r(__clc_vec16_fp32_t,
                                                         __clc_size_t,
                                                         __clc_fp16_t *,
                                                         __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn_r(__clc_vec16_fp32_t,
                                                         __clc_size_t,
                                                         __clc_fp16_t __local *,
                                                         __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec16_fp32_t, __clc_size_t,
                            __clc_fp16_t __global *, __clc_uint32_t);
#endif

#ifdef cl_khr_fp16
#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn_r(__clc_vec2_fp64_t,
                                                         __clc_size_t,
                                                         __clc_fp16_t *,
                                                         __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn_r(__clc_vec2_fp64_t,
                                                         __clc_size_t,
                                                         __clc_fp16_t __local *,
                                                         __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec2_fp64_t, __clc_size_t,
                            __clc_fp16_t __global *, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn_r(__clc_vec3_fp64_t,
                                                         __clc_size_t,
                                                         __clc_fp16_t *,
                                                         __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn_r(__clc_vec3_fp64_t,
                                                         __clc_size_t,
                                                         __clc_fp16_t __local *,
                                                         __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec3_fp64_t, __clc_size_t,
                            __clc_fp16_t __global *, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn_r(__clc_vec4_fp64_t,
                                                         __clc_size_t,
                                                         __clc_fp16_t *,
                                                         __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn_r(__clc_vec4_fp64_t,
                                                         __clc_size_t,
                                                         __clc_fp16_t __local *,
                                                         __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec4_fp64_t, __clc_size_t,
                            __clc_fp16_t __global *, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn_r(__clc_vec8_fp64_t,
                                                         __clc_size_t,
                                                         __clc_fp16_t *,
                                                         __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn_r(__clc_vec8_fp64_t,
                                                         __clc_size_t,
                                                         __clc_fp16_t __local *,
                                                         __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec8_fp64_t, __clc_size_t,
                            __clc_fp16_t __global *, __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn_r(__clc_vec16_fp64_t,
                                                         __clc_size_t,
                                                         __clc_fp16_t *,
                                                         __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstorea_halfn_r(__clc_vec16_fp64_t,
                                                         __clc_size_t,
                                                         __clc_fp16_t __local *,
                                                         __clc_uint32_t);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec16_fp64_t, __clc_size_t,
                            __clc_fp16_t __global *, __clc_uint32_t);
#endif
#endif

_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec2_char_t,
                                                 __clc_size_t, __clc_char_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_char_t, __clc_size_t, __clc_char_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_char_t, __clc_size_t, __clc_char_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec3_char_t,
                                                 __clc_size_t, __clc_char_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_char_t, __clc_size_t, __clc_char_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_char_t, __clc_size_t, __clc_char_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec4_char_t,
                                                 __clc_size_t, __clc_char_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_char_t, __clc_size_t, __clc_char_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_char_t, __clc_size_t, __clc_char_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec8_char_t,
                                                 __clc_size_t, __clc_char_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_char_t, __clc_size_t, __clc_char_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_char_t, __clc_size_t, __clc_char_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec16_char_t,
                                                 __clc_size_t, __clc_char_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec16_char_t, __clc_size_t, __clc_char_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec16_char_t, __clc_size_t, __clc_char_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec2_int8_t,
                                                 __clc_size_t, __clc_int8_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_int8_t, __clc_size_t, __clc_int8_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_int8_t, __clc_size_t, __clc_int8_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec3_int8_t,
                                                 __clc_size_t, __clc_int8_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_int8_t, __clc_size_t, __clc_int8_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_int8_t, __clc_size_t, __clc_int8_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec4_int8_t,
                                                 __clc_size_t, __clc_int8_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_int8_t, __clc_size_t, __clc_int8_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_int8_t, __clc_size_t, __clc_int8_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec8_int8_t,
                                                 __clc_size_t, __clc_int8_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_int8_t, __clc_size_t, __clc_int8_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_int8_t, __clc_size_t, __clc_int8_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec16_int8_t,
                                                 __clc_size_t, __clc_int8_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec16_int8_t, __clc_size_t, __clc_int8_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec16_int8_t, __clc_size_t, __clc_int8_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec2_int16_t,
                                                 __clc_size_t, __clc_int16_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_int16_t, __clc_size_t, __clc_int16_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_int16_t, __clc_size_t, __clc_int16_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec3_int16_t,
                                                 __clc_size_t, __clc_int16_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_int16_t, __clc_size_t, __clc_int16_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_int16_t, __clc_size_t, __clc_int16_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec4_int16_t,
                                                 __clc_size_t, __clc_int16_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_int16_t, __clc_size_t, __clc_int16_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_int16_t, __clc_size_t, __clc_int16_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec8_int16_t,
                                                 __clc_size_t, __clc_int16_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_int16_t, __clc_size_t, __clc_int16_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_int16_t, __clc_size_t, __clc_int16_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec16_int16_t,
                                                 __clc_size_t, __clc_int16_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec16_int16_t, __clc_size_t, __clc_int16_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec16_int16_t,
                                                 __clc_size_t,
                                                 __clc_int16_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec2_int32_t,
                                                 __clc_size_t, __clc_int32_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_int32_t, __clc_size_t, __clc_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_int32_t, __clc_size_t, __clc_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec3_int32_t,
                                                 __clc_size_t, __clc_int32_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_int32_t, __clc_size_t, __clc_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_int32_t, __clc_size_t, __clc_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec4_int32_t,
                                                 __clc_size_t, __clc_int32_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_int32_t, __clc_size_t, __clc_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_int32_t, __clc_size_t, __clc_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec8_int32_t,
                                                 __clc_size_t, __clc_int32_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_int32_t, __clc_size_t, __clc_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_int32_t, __clc_size_t, __clc_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec16_int32_t,
                                                 __clc_size_t, __clc_int32_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec16_int32_t, __clc_size_t, __clc_int32_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec16_int32_t,
                                                 __clc_size_t,
                                                 __clc_int32_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec2_int64_t,
                                                 __clc_size_t, __clc_int64_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_int64_t, __clc_size_t, __clc_int64_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_int64_t, __clc_size_t, __clc_int64_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec3_int64_t,
                                                 __clc_size_t, __clc_int64_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_int64_t, __clc_size_t, __clc_int64_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_int64_t, __clc_size_t, __clc_int64_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec4_int64_t,
                                                 __clc_size_t, __clc_int64_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_int64_t, __clc_size_t, __clc_int64_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_int64_t, __clc_size_t, __clc_int64_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec8_int64_t,
                                                 __clc_size_t, __clc_int64_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_int64_t, __clc_size_t, __clc_int64_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_int64_t, __clc_size_t, __clc_int64_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec16_int64_t,
                                                 __clc_size_t, __clc_int64_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec16_int64_t, __clc_size_t, __clc_int64_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec16_int64_t,
                                                 __clc_size_t,
                                                 __clc_int64_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec2_uint8_t,
                                                 __clc_size_t, __clc_uint8_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_uint8_t, __clc_size_t, __clc_uint8_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_uint8_t, __clc_size_t, __clc_uint8_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec3_uint8_t,
                                                 __clc_size_t, __clc_uint8_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_uint8_t, __clc_size_t, __clc_uint8_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_uint8_t, __clc_size_t, __clc_uint8_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec4_uint8_t,
                                                 __clc_size_t, __clc_uint8_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_uint8_t, __clc_size_t, __clc_uint8_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_uint8_t, __clc_size_t, __clc_uint8_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec8_uint8_t,
                                                 __clc_size_t, __clc_uint8_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_uint8_t, __clc_size_t, __clc_uint8_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_uint8_t, __clc_size_t, __clc_uint8_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec16_uint8_t,
                                                 __clc_size_t, __clc_uint8_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec16_uint8_t, __clc_size_t, __clc_uint8_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec16_uint8_t,
                                                 __clc_size_t,
                                                 __clc_uint8_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_uint16_t, __clc_size_t, __clc_uint16_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec2_uint16_t,
                                                 __clc_size_t,
                                                 __clc_uint16_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec2_uint16_t,
                                                 __clc_size_t,
                                                 __clc_uint16_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_uint16_t, __clc_size_t, __clc_uint16_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec3_uint16_t,
                                                 __clc_size_t,
                                                 __clc_uint16_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec3_uint16_t,
                                                 __clc_size_t,
                                                 __clc_uint16_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_uint16_t, __clc_size_t, __clc_uint16_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec4_uint16_t,
                                                 __clc_size_t,
                                                 __clc_uint16_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec4_uint16_t,
                                                 __clc_size_t,
                                                 __clc_uint16_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_uint16_t, __clc_size_t, __clc_uint16_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec8_uint16_t,
                                                 __clc_size_t,
                                                 __clc_uint16_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec8_uint16_t,
                                                 __clc_size_t,
                                                 __clc_uint16_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec16_uint16_t, __clc_size_t, __clc_uint16_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec16_uint16_t,
                                                 __clc_size_t,
                                                 __clc_uint16_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec16_uint16_t,
                                                 __clc_size_t,
                                                 __clc_uint16_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_uint32_t, __clc_size_t, __clc_uint32_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec2_uint32_t,
                                                 __clc_size_t,
                                                 __clc_uint32_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec2_uint32_t,
                                                 __clc_size_t,
                                                 __clc_uint32_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_uint32_t, __clc_size_t, __clc_uint32_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec3_uint32_t,
                                                 __clc_size_t,
                                                 __clc_uint32_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec3_uint32_t,
                                                 __clc_size_t,
                                                 __clc_uint32_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_uint32_t, __clc_size_t, __clc_uint32_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec4_uint32_t,
                                                 __clc_size_t,
                                                 __clc_uint32_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec4_uint32_t,
                                                 __clc_size_t,
                                                 __clc_uint32_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_uint32_t, __clc_size_t, __clc_uint32_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec8_uint32_t,
                                                 __clc_size_t,
                                                 __clc_uint32_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec8_uint32_t,
                                                 __clc_size_t,
                                                 __clc_uint32_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec16_uint32_t, __clc_size_t, __clc_uint32_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec16_uint32_t,
                                                 __clc_size_t,
                                                 __clc_uint32_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec16_uint32_t,
                                                 __clc_size_t,
                                                 __clc_uint32_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_uint64_t, __clc_size_t, __clc_uint64_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec2_uint64_t,
                                                 __clc_size_t,
                                                 __clc_uint64_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec2_uint64_t,
                                                 __clc_size_t,
                                                 __clc_uint64_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_uint64_t, __clc_size_t, __clc_uint64_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec3_uint64_t,
                                                 __clc_size_t,
                                                 __clc_uint64_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec3_uint64_t,
                                                 __clc_size_t,
                                                 __clc_uint64_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_uint64_t, __clc_size_t, __clc_uint64_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec4_uint64_t,
                                                 __clc_size_t,
                                                 __clc_uint64_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec4_uint64_t,
                                                 __clc_size_t,
                                                 __clc_uint64_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_uint64_t, __clc_size_t, __clc_uint64_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec8_uint64_t,
                                                 __clc_size_t,
                                                 __clc_uint64_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec8_uint64_t,
                                                 __clc_size_t,
                                                 __clc_uint64_t __global *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec16_uint64_t, __clc_size_t, __clc_uint64_t *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec16_uint64_t,
                                                 __clc_size_t,
                                                 __clc_uint64_t __local *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec16_uint64_t,
                                                 __clc_size_t,
                                                 __clc_uint64_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec2_fp32_t,
                                                 __clc_size_t, __clc_fp32_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_fp32_t, __clc_size_t, __clc_fp32_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_fp32_t, __clc_size_t, __clc_fp32_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec3_fp32_t,
                                                 __clc_size_t, __clc_fp32_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_fp32_t, __clc_size_t, __clc_fp32_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_fp32_t, __clc_size_t, __clc_fp32_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec4_fp32_t,
                                                 __clc_size_t, __clc_fp32_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_fp32_t, __clc_size_t, __clc_fp32_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_fp32_t, __clc_size_t, __clc_fp32_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec8_fp32_t,
                                                 __clc_size_t, __clc_fp32_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_fp32_t, __clc_size_t, __clc_fp32_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_fp32_t, __clc_size_t, __clc_fp32_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec16_fp32_t,
                                                 __clc_size_t, __clc_fp32_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec16_fp32_t, __clc_size_t, __clc_fp32_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec16_fp32_t, __clc_size_t, __clc_fp32_t __global *);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec2_fp64_t,
                                                 __clc_size_t, __clc_fp64_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_fp64_t, __clc_size_t, __clc_fp64_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_fp64_t, __clc_size_t, __clc_fp64_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec3_fp64_t,
                                                 __clc_size_t, __clc_fp64_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_fp64_t, __clc_size_t, __clc_fp64_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_fp64_t, __clc_size_t, __clc_fp64_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec4_fp64_t,
                                                 __clc_size_t, __clc_fp64_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_fp64_t, __clc_size_t, __clc_fp64_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_fp64_t, __clc_size_t, __clc_fp64_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec8_fp64_t,
                                                 __clc_size_t, __clc_fp64_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_fp64_t, __clc_size_t, __clc_fp64_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_fp64_t, __clc_size_t, __clc_fp64_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec16_fp64_t,
                                                 __clc_size_t, __clc_fp64_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec16_fp64_t, __clc_size_t, __clc_fp64_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec16_fp64_t, __clc_size_t, __clc_fp64_t __global *);
#endif

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec2_fp16_t,
                                                 __clc_size_t, __clc_fp16_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_fp16_t, __clc_size_t, __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_fp16_t, __clc_size_t, __clc_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec3_fp16_t,
                                                 __clc_size_t, __clc_fp16_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_fp16_t, __clc_size_t, __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_fp16_t, __clc_size_t, __clc_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec4_fp16_t,
                                                 __clc_size_t, __clc_fp16_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_fp16_t, __clc_size_t, __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_fp16_t, __clc_size_t, __clc_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec8_fp16_t,
                                                 __clc_size_t, __clc_fp16_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_fp16_t, __clc_size_t, __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_fp16_t, __clc_size_t, __clc_fp16_t __global *);
_CLC_OVERLOAD _CLC_DECL void __spirv_ocl_vstoren(__clc_vec16_fp16_t,
                                                 __clc_size_t, __clc_fp16_t *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec16_fp16_t, __clc_size_t, __clc_fp16_t __local *);
_CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec16_fp16_t, __clc_size_t, __clc_fp16_t __global *);
#endif

#endif
