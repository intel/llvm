//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <func.h>
#include <lp64_types.h>

#define __private __attribute__((opencl_private))

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_CONVERGENT _CLC_DECL void
__spirv_GroupWaitEvents(__clc_uint32_t, __clc_int32_t, __clc_event_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_CONVERGENT _CLC_DEF void
__spirv_GroupWaitEvents(__clc_uint32_t args_0, __clc_int32_t args_1,
                        __clc_event_t __private *args_2) {
  __spirv_GroupWaitEvents(args_0, args_1, (__clc_event_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_fp32_t
__spirv_ocl_fract(__clc_fp32_t, __clc_fp32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_fp32_t
__spirv_ocl_fract(__clc_fp32_t args_0, __clc_fp32_t __private *args_1) {
  return __spirv_ocl_fract(args_0, (__clc_fp32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_fract(__clc_vec2_fp32_t, __clc_vec2_fp32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_fp32_t __spirv_ocl_fract(
    __clc_vec2_fp32_t args_0, __clc_vec2_fp32_t __private *args_1) {
  return __spirv_ocl_fract(args_0, (__clc_vec2_fp32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_fract(__clc_vec3_fp32_t, __clc_vec3_fp32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_fp32_t __spirv_ocl_fract(
    __clc_vec3_fp32_t args_0, __clc_vec3_fp32_t __private *args_1) {
  return __spirv_ocl_fract(args_0, (__clc_vec3_fp32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_fract(__clc_vec4_fp32_t, __clc_vec4_fp32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_fp32_t __spirv_ocl_fract(
    __clc_vec4_fp32_t args_0, __clc_vec4_fp32_t __private *args_1) {
  return __spirv_ocl_fract(args_0, (__clc_vec4_fp32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_fract(__clc_vec8_fp32_t, __clc_vec8_fp32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_fp32_t __spirv_ocl_fract(
    __clc_vec8_fp32_t args_0, __clc_vec8_fp32_t __private *args_1) {
  return __spirv_ocl_fract(args_0, (__clc_vec8_fp32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_fract(__clc_vec16_fp32_t, __clc_vec16_fp32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_fp32_t __spirv_ocl_fract(
    __clc_vec16_fp32_t args_0, __clc_vec16_fp32_t __private *args_1) {
  return __spirv_ocl_fract(args_0, (__clc_vec16_fp32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_fp64_t
__spirv_ocl_fract(__clc_fp64_t, __clc_fp64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_fp64_t
__spirv_ocl_fract(__clc_fp64_t args_0, __clc_fp64_t __private *args_1) {
  return __spirv_ocl_fract(args_0, (__clc_fp64_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_fract(__clc_vec2_fp64_t, __clc_vec2_fp64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_fp64_t __spirv_ocl_fract(
    __clc_vec2_fp64_t args_0, __clc_vec2_fp64_t __private *args_1) {
  return __spirv_ocl_fract(args_0, (__clc_vec2_fp64_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_fract(__clc_vec3_fp64_t, __clc_vec3_fp64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_fp64_t __spirv_ocl_fract(
    __clc_vec3_fp64_t args_0, __clc_vec3_fp64_t __private *args_1) {
  return __spirv_ocl_fract(args_0, (__clc_vec3_fp64_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_fract(__clc_vec4_fp64_t, __clc_vec4_fp64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_fp64_t __spirv_ocl_fract(
    __clc_vec4_fp64_t args_0, __clc_vec4_fp64_t __private *args_1) {
  return __spirv_ocl_fract(args_0, (__clc_vec4_fp64_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_fract(__clc_vec8_fp64_t, __clc_vec8_fp64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_fp64_t __spirv_ocl_fract(
    __clc_vec8_fp64_t args_0, __clc_vec8_fp64_t __private *args_1) {
  return __spirv_ocl_fract(args_0, (__clc_vec8_fp64_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_fract(__clc_vec16_fp64_t, __clc_vec16_fp64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_fp64_t __spirv_ocl_fract(
    __clc_vec16_fp64_t args_0, __clc_vec16_fp64_t __private *args_1) {
  return __spirv_ocl_fract(args_0, (__clc_vec16_fp64_t *)(args_1));
}

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_fp16_t
__spirv_ocl_fract(__clc_fp16_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_fp16_t
__spirv_ocl_fract(__clc_fp16_t args_0, __clc_fp16_t __private *args_1) {
  return __spirv_ocl_fract(args_0, (__clc_fp16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_fract(__clc_vec2_fp16_t, __clc_vec2_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t __spirv_ocl_fract(
    __clc_vec2_fp16_t args_0, __clc_vec2_fp16_t __private *args_1) {
  return __spirv_ocl_fract(args_0, (__clc_vec2_fp16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_fract(__clc_vec3_fp16_t, __clc_vec3_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t __spirv_ocl_fract(
    __clc_vec3_fp16_t args_0, __clc_vec3_fp16_t __private *args_1) {
  return __spirv_ocl_fract(args_0, (__clc_vec3_fp16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_fract(__clc_vec4_fp16_t, __clc_vec4_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t __spirv_ocl_fract(
    __clc_vec4_fp16_t args_0, __clc_vec4_fp16_t __private *args_1) {
  return __spirv_ocl_fract(args_0, (__clc_vec4_fp16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_fract(__clc_vec8_fp16_t, __clc_vec8_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t __spirv_ocl_fract(
    __clc_vec8_fp16_t args_0, __clc_vec8_fp16_t __private *args_1) {
  return __spirv_ocl_fract(args_0, (__clc_vec8_fp16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_fract(__clc_vec16_fp16_t, __clc_vec16_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t __spirv_ocl_fract(
    __clc_vec16_fp16_t args_0, __clc_vec16_fp16_t __private *args_1) {
  return __spirv_ocl_fract(args_0, (__clc_vec16_fp16_t *)(args_1));
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_float16_t
__spirv_ocl_fract(__clc_float16_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_float16_t
__spirv_ocl_fract(__clc_float16_t args_0, __clc_float16_t __private *args_1) {
  return __spirv_ocl_fract(args_0, (__clc_float16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_float16_t
__spirv_ocl_fract(__clc_vec2_float16_t, __clc_vec2_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_float16_t __spirv_ocl_fract(
    __clc_vec2_float16_t args_0, __clc_vec2_float16_t __private *args_1) {
  return __spirv_ocl_fract(args_0, (__clc_vec2_float16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_float16_t
__spirv_ocl_fract(__clc_vec3_float16_t, __clc_vec3_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_float16_t __spirv_ocl_fract(
    __clc_vec3_float16_t args_0, __clc_vec3_float16_t __private *args_1) {
  return __spirv_ocl_fract(args_0, (__clc_vec3_float16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_float16_t
__spirv_ocl_fract(__clc_vec4_float16_t, __clc_vec4_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_float16_t __spirv_ocl_fract(
    __clc_vec4_float16_t args_0, __clc_vec4_float16_t __private *args_1) {
  return __spirv_ocl_fract(args_0, (__clc_vec4_float16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_float16_t
__spirv_ocl_fract(__clc_vec8_float16_t, __clc_vec8_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_float16_t __spirv_ocl_fract(
    __clc_vec8_float16_t args_0, __clc_vec8_float16_t __private *args_1) {
  return __spirv_ocl_fract(args_0, (__clc_vec8_float16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_float16_t
__spirv_ocl_fract(__clc_vec16_float16_t, __clc_vec16_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_float16_t __spirv_ocl_fract(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t __private *args_1) {
  return __spirv_ocl_fract(args_0, (__clc_vec16_float16_t *)(args_1));
}

#endif

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_fp32_t
__spirv_ocl_frexp(__clc_fp32_t, __clc_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_fp32_t
__spirv_ocl_frexp(__clc_fp32_t args_0, __clc_int32_t __private *args_1) {
  return __spirv_ocl_frexp(args_0, (__clc_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_frexp(__clc_vec2_fp32_t, __clc_vec2_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_fp32_t __spirv_ocl_frexp(
    __clc_vec2_fp32_t args_0, __clc_vec2_int32_t __private *args_1) {
  return __spirv_ocl_frexp(args_0, (__clc_vec2_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_frexp(__clc_vec3_fp32_t, __clc_vec3_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_fp32_t __spirv_ocl_frexp(
    __clc_vec3_fp32_t args_0, __clc_vec3_int32_t __private *args_1) {
  return __spirv_ocl_frexp(args_0, (__clc_vec3_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_frexp(__clc_vec4_fp32_t, __clc_vec4_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_fp32_t __spirv_ocl_frexp(
    __clc_vec4_fp32_t args_0, __clc_vec4_int32_t __private *args_1) {
  return __spirv_ocl_frexp(args_0, (__clc_vec4_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_frexp(__clc_vec8_fp32_t, __clc_vec8_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_fp32_t __spirv_ocl_frexp(
    __clc_vec8_fp32_t args_0, __clc_vec8_int32_t __private *args_1) {
  return __spirv_ocl_frexp(args_0, (__clc_vec8_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_frexp(__clc_vec16_fp32_t, __clc_vec16_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_fp32_t __spirv_ocl_frexp(
    __clc_vec16_fp32_t args_0, __clc_vec16_int32_t __private *args_1) {
  return __spirv_ocl_frexp(args_0, (__clc_vec16_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_fp64_t
__spirv_ocl_frexp(__clc_fp64_t, __clc_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_fp64_t
__spirv_ocl_frexp(__clc_fp64_t args_0, __clc_int32_t __private *args_1) {
  return __spirv_ocl_frexp(args_0, (__clc_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_frexp(__clc_vec2_fp64_t, __clc_vec2_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_fp64_t __spirv_ocl_frexp(
    __clc_vec2_fp64_t args_0, __clc_vec2_int32_t __private *args_1) {
  return __spirv_ocl_frexp(args_0, (__clc_vec2_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_frexp(__clc_vec3_fp64_t, __clc_vec3_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_fp64_t __spirv_ocl_frexp(
    __clc_vec3_fp64_t args_0, __clc_vec3_int32_t __private *args_1) {
  return __spirv_ocl_frexp(args_0, (__clc_vec3_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_frexp(__clc_vec4_fp64_t, __clc_vec4_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_fp64_t __spirv_ocl_frexp(
    __clc_vec4_fp64_t args_0, __clc_vec4_int32_t __private *args_1) {
  return __spirv_ocl_frexp(args_0, (__clc_vec4_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_frexp(__clc_vec8_fp64_t, __clc_vec8_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_fp64_t __spirv_ocl_frexp(
    __clc_vec8_fp64_t args_0, __clc_vec8_int32_t __private *args_1) {
  return __spirv_ocl_frexp(args_0, (__clc_vec8_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_frexp(__clc_vec16_fp64_t, __clc_vec16_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_fp64_t __spirv_ocl_frexp(
    __clc_vec16_fp64_t args_0, __clc_vec16_int32_t __private *args_1) {
  return __spirv_ocl_frexp(args_0, (__clc_vec16_int32_t *)(args_1));
}

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_fp16_t
__spirv_ocl_frexp(__clc_fp16_t, __clc_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_fp16_t
__spirv_ocl_frexp(__clc_fp16_t args_0, __clc_int32_t __private *args_1) {
  return __spirv_ocl_frexp(args_0, (__clc_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_frexp(__clc_vec2_fp16_t, __clc_vec2_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t __spirv_ocl_frexp(
    __clc_vec2_fp16_t args_0, __clc_vec2_int32_t __private *args_1) {
  return __spirv_ocl_frexp(args_0, (__clc_vec2_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_frexp(__clc_vec3_fp16_t, __clc_vec3_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t __spirv_ocl_frexp(
    __clc_vec3_fp16_t args_0, __clc_vec3_int32_t __private *args_1) {
  return __spirv_ocl_frexp(args_0, (__clc_vec3_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_frexp(__clc_vec4_fp16_t, __clc_vec4_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t __spirv_ocl_frexp(
    __clc_vec4_fp16_t args_0, __clc_vec4_int32_t __private *args_1) {
  return __spirv_ocl_frexp(args_0, (__clc_vec4_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_frexp(__clc_vec8_fp16_t, __clc_vec8_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t __spirv_ocl_frexp(
    __clc_vec8_fp16_t args_0, __clc_vec8_int32_t __private *args_1) {
  return __spirv_ocl_frexp(args_0, (__clc_vec8_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_frexp(__clc_vec16_fp16_t, __clc_vec16_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t __spirv_ocl_frexp(
    __clc_vec16_fp16_t args_0, __clc_vec16_int32_t __private *args_1) {
  return __spirv_ocl_frexp(args_0, (__clc_vec16_int32_t *)(args_1));
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_float16_t
__spirv_ocl_frexp(__clc_float16_t, __clc_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_float16_t
__spirv_ocl_frexp(__clc_float16_t args_0, __clc_int32_t __private *args_1) {
  return __spirv_ocl_frexp(args_0, (__clc_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_float16_t
__spirv_ocl_frexp(__clc_vec2_float16_t, __clc_vec2_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_float16_t __spirv_ocl_frexp(
    __clc_vec2_float16_t args_0, __clc_vec2_int32_t __private *args_1) {
  return __spirv_ocl_frexp(args_0, (__clc_vec2_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_float16_t
__spirv_ocl_frexp(__clc_vec3_float16_t, __clc_vec3_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_float16_t __spirv_ocl_frexp(
    __clc_vec3_float16_t args_0, __clc_vec3_int32_t __private *args_1) {
  return __spirv_ocl_frexp(args_0, (__clc_vec3_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_float16_t
__spirv_ocl_frexp(__clc_vec4_float16_t, __clc_vec4_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_float16_t __spirv_ocl_frexp(
    __clc_vec4_float16_t args_0, __clc_vec4_int32_t __private *args_1) {
  return __spirv_ocl_frexp(args_0, (__clc_vec4_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_float16_t
__spirv_ocl_frexp(__clc_vec8_float16_t, __clc_vec8_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_float16_t __spirv_ocl_frexp(
    __clc_vec8_float16_t args_0, __clc_vec8_int32_t __private *args_1) {
  return __spirv_ocl_frexp(args_0, (__clc_vec8_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_float16_t
__spirv_ocl_frexp(__clc_vec16_float16_t, __clc_vec16_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_float16_t __spirv_ocl_frexp(
    __clc_vec16_float16_t args_0, __clc_vec16_int32_t __private *args_1) {
  return __spirv_ocl_frexp(args_0, (__clc_vec16_int32_t *)(args_1));
}

#endif

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_fp32_t
__spirv_ocl_lgamma_r(__clc_fp32_t, __clc_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_fp32_t
__spirv_ocl_lgamma_r(__clc_fp32_t args_0, __clc_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(args_0, (__clc_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_lgamma_r(__clc_vec2_fp32_t, __clc_vec2_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_fp32_t __spirv_ocl_lgamma_r(
    __clc_vec2_fp32_t args_0, __clc_vec2_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(args_0, (__clc_vec2_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_lgamma_r(__clc_vec3_fp32_t, __clc_vec3_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_fp32_t __spirv_ocl_lgamma_r(
    __clc_vec3_fp32_t args_0, __clc_vec3_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(args_0, (__clc_vec3_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_lgamma_r(__clc_vec4_fp32_t, __clc_vec4_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_fp32_t __spirv_ocl_lgamma_r(
    __clc_vec4_fp32_t args_0, __clc_vec4_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(args_0, (__clc_vec4_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_lgamma_r(__clc_vec8_fp32_t, __clc_vec8_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_fp32_t __spirv_ocl_lgamma_r(
    __clc_vec8_fp32_t args_0, __clc_vec8_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(args_0, (__clc_vec8_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_lgamma_r(__clc_vec16_fp32_t, __clc_vec16_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_fp32_t __spirv_ocl_lgamma_r(
    __clc_vec16_fp32_t args_0, __clc_vec16_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(args_0, (__clc_vec16_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_fp64_t
__spirv_ocl_lgamma_r(__clc_fp64_t, __clc_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_fp64_t
__spirv_ocl_lgamma_r(__clc_fp64_t args_0, __clc_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(args_0, (__clc_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_lgamma_r(__clc_vec2_fp64_t, __clc_vec2_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_fp64_t __spirv_ocl_lgamma_r(
    __clc_vec2_fp64_t args_0, __clc_vec2_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(args_0, (__clc_vec2_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_lgamma_r(__clc_vec3_fp64_t, __clc_vec3_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_fp64_t __spirv_ocl_lgamma_r(
    __clc_vec3_fp64_t args_0, __clc_vec3_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(args_0, (__clc_vec3_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_lgamma_r(__clc_vec4_fp64_t, __clc_vec4_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_fp64_t __spirv_ocl_lgamma_r(
    __clc_vec4_fp64_t args_0, __clc_vec4_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(args_0, (__clc_vec4_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_lgamma_r(__clc_vec8_fp64_t, __clc_vec8_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_fp64_t __spirv_ocl_lgamma_r(
    __clc_vec8_fp64_t args_0, __clc_vec8_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(args_0, (__clc_vec8_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_lgamma_r(__clc_vec16_fp64_t, __clc_vec16_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_fp64_t __spirv_ocl_lgamma_r(
    __clc_vec16_fp64_t args_0, __clc_vec16_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(args_0, (__clc_vec16_int32_t *)(args_1));
}

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_fp16_t
__spirv_ocl_lgamma_r(__clc_fp16_t, __clc_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_fp16_t
__spirv_ocl_lgamma_r(__clc_fp16_t args_0, __clc_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(args_0, (__clc_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_lgamma_r(__clc_vec2_fp16_t, __clc_vec2_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t __spirv_ocl_lgamma_r(
    __clc_vec2_fp16_t args_0, __clc_vec2_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(args_0, (__clc_vec2_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_lgamma_r(__clc_vec3_fp16_t, __clc_vec3_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t __spirv_ocl_lgamma_r(
    __clc_vec3_fp16_t args_0, __clc_vec3_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(args_0, (__clc_vec3_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_lgamma_r(__clc_vec4_fp16_t, __clc_vec4_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t __spirv_ocl_lgamma_r(
    __clc_vec4_fp16_t args_0, __clc_vec4_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(args_0, (__clc_vec4_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_lgamma_r(__clc_vec8_fp16_t, __clc_vec8_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t __spirv_ocl_lgamma_r(
    __clc_vec8_fp16_t args_0, __clc_vec8_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(args_0, (__clc_vec8_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_lgamma_r(__clc_vec16_fp16_t, __clc_vec16_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t __spirv_ocl_lgamma_r(
    __clc_vec16_fp16_t args_0, __clc_vec16_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(args_0, (__clc_vec16_int32_t *)(args_1));
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_float16_t
__spirv_ocl_lgamma_r(__clc_float16_t, __clc_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_float16_t
__spirv_ocl_lgamma_r(__clc_float16_t args_0, __clc_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(args_0, (__clc_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_float16_t
__spirv_ocl_lgamma_r(__clc_vec2_float16_t, __clc_vec2_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_float16_t __spirv_ocl_lgamma_r(
    __clc_vec2_float16_t args_0, __clc_vec2_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(args_0, (__clc_vec2_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_float16_t
__spirv_ocl_lgamma_r(__clc_vec3_float16_t, __clc_vec3_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_float16_t __spirv_ocl_lgamma_r(
    __clc_vec3_float16_t args_0, __clc_vec3_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(args_0, (__clc_vec3_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_float16_t
__spirv_ocl_lgamma_r(__clc_vec4_float16_t, __clc_vec4_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_float16_t __spirv_ocl_lgamma_r(
    __clc_vec4_float16_t args_0, __clc_vec4_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(args_0, (__clc_vec4_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_float16_t
__spirv_ocl_lgamma_r(__clc_vec8_float16_t, __clc_vec8_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_float16_t __spirv_ocl_lgamma_r(
    __clc_vec8_float16_t args_0, __clc_vec8_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(args_0, (__clc_vec8_int32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_float16_t
__spirv_ocl_lgamma_r(__clc_vec16_float16_t, __clc_vec16_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_float16_t __spirv_ocl_lgamma_r(
    __clc_vec16_float16_t args_0, __clc_vec16_int32_t __private *args_1) {
  return __spirv_ocl_lgamma_r(args_0, (__clc_vec16_int32_t *)(args_1));
}

#endif

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_fp32_t
__spirv_ocl_modf(__clc_fp32_t, __clc_fp32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_fp32_t
__spirv_ocl_modf(__clc_fp32_t args_0, __clc_fp32_t __private *args_1) {
  return __spirv_ocl_modf(args_0, (__clc_fp32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_modf(__clc_vec2_fp32_t, __clc_vec2_fp32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_fp32_t __spirv_ocl_modf(
    __clc_vec2_fp32_t args_0, __clc_vec2_fp32_t __private *args_1) {
  return __spirv_ocl_modf(args_0, (__clc_vec2_fp32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_modf(__clc_vec3_fp32_t, __clc_vec3_fp32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_fp32_t __spirv_ocl_modf(
    __clc_vec3_fp32_t args_0, __clc_vec3_fp32_t __private *args_1) {
  return __spirv_ocl_modf(args_0, (__clc_vec3_fp32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_modf(__clc_vec4_fp32_t, __clc_vec4_fp32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_fp32_t __spirv_ocl_modf(
    __clc_vec4_fp32_t args_0, __clc_vec4_fp32_t __private *args_1) {
  return __spirv_ocl_modf(args_0, (__clc_vec4_fp32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_modf(__clc_vec8_fp32_t, __clc_vec8_fp32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_fp32_t __spirv_ocl_modf(
    __clc_vec8_fp32_t args_0, __clc_vec8_fp32_t __private *args_1) {
  return __spirv_ocl_modf(args_0, (__clc_vec8_fp32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_modf(__clc_vec16_fp32_t, __clc_vec16_fp32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_fp32_t __spirv_ocl_modf(
    __clc_vec16_fp32_t args_0, __clc_vec16_fp32_t __private *args_1) {
  return __spirv_ocl_modf(args_0, (__clc_vec16_fp32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_fp64_t
__spirv_ocl_modf(__clc_fp64_t, __clc_fp64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_fp64_t
__spirv_ocl_modf(__clc_fp64_t args_0, __clc_fp64_t __private *args_1) {
  return __spirv_ocl_modf(args_0, (__clc_fp64_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_modf(__clc_vec2_fp64_t, __clc_vec2_fp64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_fp64_t __spirv_ocl_modf(
    __clc_vec2_fp64_t args_0, __clc_vec2_fp64_t __private *args_1) {
  return __spirv_ocl_modf(args_0, (__clc_vec2_fp64_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_modf(__clc_vec3_fp64_t, __clc_vec3_fp64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_fp64_t __spirv_ocl_modf(
    __clc_vec3_fp64_t args_0, __clc_vec3_fp64_t __private *args_1) {
  return __spirv_ocl_modf(args_0, (__clc_vec3_fp64_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_modf(__clc_vec4_fp64_t, __clc_vec4_fp64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_fp64_t __spirv_ocl_modf(
    __clc_vec4_fp64_t args_0, __clc_vec4_fp64_t __private *args_1) {
  return __spirv_ocl_modf(args_0, (__clc_vec4_fp64_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_modf(__clc_vec8_fp64_t, __clc_vec8_fp64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_fp64_t __spirv_ocl_modf(
    __clc_vec8_fp64_t args_0, __clc_vec8_fp64_t __private *args_1) {
  return __spirv_ocl_modf(args_0, (__clc_vec8_fp64_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_modf(__clc_vec16_fp64_t, __clc_vec16_fp64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_fp64_t __spirv_ocl_modf(
    __clc_vec16_fp64_t args_0, __clc_vec16_fp64_t __private *args_1) {
  return __spirv_ocl_modf(args_0, (__clc_vec16_fp64_t *)(args_1));
}

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_fp16_t
__spirv_ocl_modf(__clc_fp16_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_fp16_t
__spirv_ocl_modf(__clc_fp16_t args_0, __clc_fp16_t __private *args_1) {
  return __spirv_ocl_modf(args_0, (__clc_fp16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_modf(__clc_vec2_fp16_t, __clc_vec2_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t __spirv_ocl_modf(
    __clc_vec2_fp16_t args_0, __clc_vec2_fp16_t __private *args_1) {
  return __spirv_ocl_modf(args_0, (__clc_vec2_fp16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_modf(__clc_vec3_fp16_t, __clc_vec3_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t __spirv_ocl_modf(
    __clc_vec3_fp16_t args_0, __clc_vec3_fp16_t __private *args_1) {
  return __spirv_ocl_modf(args_0, (__clc_vec3_fp16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_modf(__clc_vec4_fp16_t, __clc_vec4_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t __spirv_ocl_modf(
    __clc_vec4_fp16_t args_0, __clc_vec4_fp16_t __private *args_1) {
  return __spirv_ocl_modf(args_0, (__clc_vec4_fp16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_modf(__clc_vec8_fp16_t, __clc_vec8_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t __spirv_ocl_modf(
    __clc_vec8_fp16_t args_0, __clc_vec8_fp16_t __private *args_1) {
  return __spirv_ocl_modf(args_0, (__clc_vec8_fp16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_modf(__clc_vec16_fp16_t, __clc_vec16_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t __spirv_ocl_modf(
    __clc_vec16_fp16_t args_0, __clc_vec16_fp16_t __private *args_1) {
  return __spirv_ocl_modf(args_0, (__clc_vec16_fp16_t *)(args_1));
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_float16_t
__spirv_ocl_modf(__clc_float16_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_float16_t
__spirv_ocl_modf(__clc_float16_t args_0, __clc_float16_t __private *args_1) {
  return __spirv_ocl_modf(args_0, (__clc_float16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_float16_t
__spirv_ocl_modf(__clc_vec2_float16_t, __clc_vec2_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_float16_t __spirv_ocl_modf(
    __clc_vec2_float16_t args_0, __clc_vec2_float16_t __private *args_1) {
  return __spirv_ocl_modf(args_0, (__clc_vec2_float16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_float16_t
__spirv_ocl_modf(__clc_vec3_float16_t, __clc_vec3_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_float16_t __spirv_ocl_modf(
    __clc_vec3_float16_t args_0, __clc_vec3_float16_t __private *args_1) {
  return __spirv_ocl_modf(args_0, (__clc_vec3_float16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_float16_t
__spirv_ocl_modf(__clc_vec4_float16_t, __clc_vec4_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_float16_t __spirv_ocl_modf(
    __clc_vec4_float16_t args_0, __clc_vec4_float16_t __private *args_1) {
  return __spirv_ocl_modf(args_0, (__clc_vec4_float16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_float16_t
__spirv_ocl_modf(__clc_vec8_float16_t, __clc_vec8_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_float16_t __spirv_ocl_modf(
    __clc_vec8_float16_t args_0, __clc_vec8_float16_t __private *args_1) {
  return __spirv_ocl_modf(args_0, (__clc_vec8_float16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_float16_t
__spirv_ocl_modf(__clc_vec16_float16_t, __clc_vec16_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_float16_t __spirv_ocl_modf(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t __private *args_1) {
  return __spirv_ocl_modf(args_0, (__clc_vec16_float16_t *)(args_1));
}

#endif

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_fp32_t
__spirv_ocl_remquo(__clc_fp32_t, __clc_fp32_t, __clc_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_fp32_t __spirv_ocl_remquo(
    __clc_fp32_t args_0, __clc_fp32_t args_1, __clc_int32_t __private *args_2) {
  return __spirv_ocl_remquo(args_0, args_1, (__clc_int32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_remquo(__clc_vec2_fp32_t, __clc_vec2_fp32_t, __clc_vec2_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_fp32_t
__spirv_ocl_remquo(__clc_vec2_fp32_t args_0, __clc_vec2_fp32_t args_1,
                   __clc_vec2_int32_t __private *args_2) {
  return __spirv_ocl_remquo(args_0, args_1, (__clc_vec2_int32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_remquo(__clc_vec3_fp32_t, __clc_vec3_fp32_t, __clc_vec3_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_fp32_t
__spirv_ocl_remquo(__clc_vec3_fp32_t args_0, __clc_vec3_fp32_t args_1,
                   __clc_vec3_int32_t __private *args_2) {
  return __spirv_ocl_remquo(args_0, args_1, (__clc_vec3_int32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_remquo(__clc_vec4_fp32_t, __clc_vec4_fp32_t, __clc_vec4_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_fp32_t
__spirv_ocl_remquo(__clc_vec4_fp32_t args_0, __clc_vec4_fp32_t args_1,
                   __clc_vec4_int32_t __private *args_2) {
  return __spirv_ocl_remquo(args_0, args_1, (__clc_vec4_int32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_remquo(__clc_vec8_fp32_t, __clc_vec8_fp32_t, __clc_vec8_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_fp32_t
__spirv_ocl_remquo(__clc_vec8_fp32_t args_0, __clc_vec8_fp32_t args_1,
                   __clc_vec8_int32_t __private *args_2) {
  return __spirv_ocl_remquo(args_0, args_1, (__clc_vec8_int32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t __spirv_ocl_remquo(
    __clc_vec16_fp32_t, __clc_vec16_fp32_t, __clc_vec16_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_fp32_t
__spirv_ocl_remquo(__clc_vec16_fp32_t args_0, __clc_vec16_fp32_t args_1,
                   __clc_vec16_int32_t __private *args_2) {
  return __spirv_ocl_remquo(args_0, args_1, (__clc_vec16_int32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_fp64_t
__spirv_ocl_remquo(__clc_fp64_t, __clc_fp64_t, __clc_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_fp64_t __spirv_ocl_remquo(
    __clc_fp64_t args_0, __clc_fp64_t args_1, __clc_int32_t __private *args_2) {
  return __spirv_ocl_remquo(args_0, args_1, (__clc_int32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_remquo(__clc_vec2_fp64_t, __clc_vec2_fp64_t, __clc_vec2_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_fp64_t
__spirv_ocl_remquo(__clc_vec2_fp64_t args_0, __clc_vec2_fp64_t args_1,
                   __clc_vec2_int32_t __private *args_2) {
  return __spirv_ocl_remquo(args_0, args_1, (__clc_vec2_int32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_remquo(__clc_vec3_fp64_t, __clc_vec3_fp64_t, __clc_vec3_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_fp64_t
__spirv_ocl_remquo(__clc_vec3_fp64_t args_0, __clc_vec3_fp64_t args_1,
                   __clc_vec3_int32_t __private *args_2) {
  return __spirv_ocl_remquo(args_0, args_1, (__clc_vec3_int32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_remquo(__clc_vec4_fp64_t, __clc_vec4_fp64_t, __clc_vec4_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_fp64_t
__spirv_ocl_remquo(__clc_vec4_fp64_t args_0, __clc_vec4_fp64_t args_1,
                   __clc_vec4_int32_t __private *args_2) {
  return __spirv_ocl_remquo(args_0, args_1, (__clc_vec4_int32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_remquo(__clc_vec8_fp64_t, __clc_vec8_fp64_t, __clc_vec8_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_fp64_t
__spirv_ocl_remquo(__clc_vec8_fp64_t args_0, __clc_vec8_fp64_t args_1,
                   __clc_vec8_int32_t __private *args_2) {
  return __spirv_ocl_remquo(args_0, args_1, (__clc_vec8_int32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t __spirv_ocl_remquo(
    __clc_vec16_fp64_t, __clc_vec16_fp64_t, __clc_vec16_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_fp64_t
__spirv_ocl_remquo(__clc_vec16_fp64_t args_0, __clc_vec16_fp64_t args_1,
                   __clc_vec16_int32_t __private *args_2) {
  return __spirv_ocl_remquo(args_0, args_1, (__clc_vec16_int32_t *)(args_2));
}

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_fp16_t
__spirv_ocl_remquo(__clc_fp16_t, __clc_fp16_t, __clc_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_fp16_t __spirv_ocl_remquo(
    __clc_fp16_t args_0, __clc_fp16_t args_1, __clc_int32_t __private *args_2) {
  return __spirv_ocl_remquo(args_0, args_1, (__clc_int32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_remquo(__clc_vec2_fp16_t, __clc_vec2_fp16_t, __clc_vec2_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t
__spirv_ocl_remquo(__clc_vec2_fp16_t args_0, __clc_vec2_fp16_t args_1,
                   __clc_vec2_int32_t __private *args_2) {
  return __spirv_ocl_remquo(args_0, args_1, (__clc_vec2_int32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_remquo(__clc_vec3_fp16_t, __clc_vec3_fp16_t, __clc_vec3_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t
__spirv_ocl_remquo(__clc_vec3_fp16_t args_0, __clc_vec3_fp16_t args_1,
                   __clc_vec3_int32_t __private *args_2) {
  return __spirv_ocl_remquo(args_0, args_1, (__clc_vec3_int32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_remquo(__clc_vec4_fp16_t, __clc_vec4_fp16_t, __clc_vec4_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t
__spirv_ocl_remquo(__clc_vec4_fp16_t args_0, __clc_vec4_fp16_t args_1,
                   __clc_vec4_int32_t __private *args_2) {
  return __spirv_ocl_remquo(args_0, args_1, (__clc_vec4_int32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_remquo(__clc_vec8_fp16_t, __clc_vec8_fp16_t, __clc_vec8_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t
__spirv_ocl_remquo(__clc_vec8_fp16_t args_0, __clc_vec8_fp16_t args_1,
                   __clc_vec8_int32_t __private *args_2) {
  return __spirv_ocl_remquo(args_0, args_1, (__clc_vec8_int32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t __spirv_ocl_remquo(
    __clc_vec16_fp16_t, __clc_vec16_fp16_t, __clc_vec16_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t
__spirv_ocl_remquo(__clc_vec16_fp16_t args_0, __clc_vec16_fp16_t args_1,
                   __clc_vec16_int32_t __private *args_2) {
  return __spirv_ocl_remquo(args_0, args_1, (__clc_vec16_int32_t *)(args_2));
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_float16_t
__spirv_ocl_remquo(__clc_float16_t, __clc_float16_t, __clc_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_float16_t
__spirv_ocl_remquo(__clc_float16_t args_0, __clc_float16_t args_1,
                   __clc_int32_t __private *args_2) {
  return __spirv_ocl_remquo(args_0, args_1, (__clc_int32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_float16_t __spirv_ocl_remquo(
    __clc_vec2_float16_t, __clc_vec2_float16_t, __clc_vec2_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_float16_t
__spirv_ocl_remquo(__clc_vec2_float16_t args_0, __clc_vec2_float16_t args_1,
                   __clc_vec2_int32_t __private *args_2) {
  return __spirv_ocl_remquo(args_0, args_1, (__clc_vec2_int32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_float16_t __spirv_ocl_remquo(
    __clc_vec3_float16_t, __clc_vec3_float16_t, __clc_vec3_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_float16_t
__spirv_ocl_remquo(__clc_vec3_float16_t args_0, __clc_vec3_float16_t args_1,
                   __clc_vec3_int32_t __private *args_2) {
  return __spirv_ocl_remquo(args_0, args_1, (__clc_vec3_int32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_float16_t __spirv_ocl_remquo(
    __clc_vec4_float16_t, __clc_vec4_float16_t, __clc_vec4_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_float16_t
__spirv_ocl_remquo(__clc_vec4_float16_t args_0, __clc_vec4_float16_t args_1,
                   __clc_vec4_int32_t __private *args_2) {
  return __spirv_ocl_remquo(args_0, args_1, (__clc_vec4_int32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_float16_t __spirv_ocl_remquo(
    __clc_vec8_float16_t, __clc_vec8_float16_t, __clc_vec8_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_float16_t
__spirv_ocl_remquo(__clc_vec8_float16_t args_0, __clc_vec8_float16_t args_1,
                   __clc_vec8_int32_t __private *args_2) {
  return __spirv_ocl_remquo(args_0, args_1, (__clc_vec8_int32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_float16_t __spirv_ocl_remquo(
    __clc_vec16_float16_t, __clc_vec16_float16_t, __clc_vec16_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_float16_t
__spirv_ocl_remquo(__clc_vec16_float16_t args_0, __clc_vec16_float16_t args_1,
                   __clc_vec16_int32_t __private *args_2) {
  return __spirv_ocl_remquo(args_0, args_1, (__clc_vec16_int32_t *)(args_2));
}

#endif

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_fp32_t
__spirv_ocl_sincos(__clc_fp32_t, __clc_fp32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_fp32_t
__spirv_ocl_sincos(__clc_fp32_t args_0, __clc_fp32_t __private *args_1) {
  return __spirv_ocl_sincos(args_0, (__clc_fp32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_sincos(__clc_vec2_fp32_t, __clc_vec2_fp32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_fp32_t __spirv_ocl_sincos(
    __clc_vec2_fp32_t args_0, __clc_vec2_fp32_t __private *args_1) {
  return __spirv_ocl_sincos(args_0, (__clc_vec2_fp32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_sincos(__clc_vec3_fp32_t, __clc_vec3_fp32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_fp32_t __spirv_ocl_sincos(
    __clc_vec3_fp32_t args_0, __clc_vec3_fp32_t __private *args_1) {
  return __spirv_ocl_sincos(args_0, (__clc_vec3_fp32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_sincos(__clc_vec4_fp32_t, __clc_vec4_fp32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_fp32_t __spirv_ocl_sincos(
    __clc_vec4_fp32_t args_0, __clc_vec4_fp32_t __private *args_1) {
  return __spirv_ocl_sincos(args_0, (__clc_vec4_fp32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_sincos(__clc_vec8_fp32_t, __clc_vec8_fp32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_fp32_t __spirv_ocl_sincos(
    __clc_vec8_fp32_t args_0, __clc_vec8_fp32_t __private *args_1) {
  return __spirv_ocl_sincos(args_0, (__clc_vec8_fp32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_sincos(__clc_vec16_fp32_t, __clc_vec16_fp32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_fp32_t __spirv_ocl_sincos(
    __clc_vec16_fp32_t args_0, __clc_vec16_fp32_t __private *args_1) {
  return __spirv_ocl_sincos(args_0, (__clc_vec16_fp32_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_fp64_t
__spirv_ocl_sincos(__clc_fp64_t, __clc_fp64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_fp64_t
__spirv_ocl_sincos(__clc_fp64_t args_0, __clc_fp64_t __private *args_1) {
  return __spirv_ocl_sincos(args_0, (__clc_fp64_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_sincos(__clc_vec2_fp64_t, __clc_vec2_fp64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_fp64_t __spirv_ocl_sincos(
    __clc_vec2_fp64_t args_0, __clc_vec2_fp64_t __private *args_1) {
  return __spirv_ocl_sincos(args_0, (__clc_vec2_fp64_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_sincos(__clc_vec3_fp64_t, __clc_vec3_fp64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_fp64_t __spirv_ocl_sincos(
    __clc_vec3_fp64_t args_0, __clc_vec3_fp64_t __private *args_1) {
  return __spirv_ocl_sincos(args_0, (__clc_vec3_fp64_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_sincos(__clc_vec4_fp64_t, __clc_vec4_fp64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_fp64_t __spirv_ocl_sincos(
    __clc_vec4_fp64_t args_0, __clc_vec4_fp64_t __private *args_1) {
  return __spirv_ocl_sincos(args_0, (__clc_vec4_fp64_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_sincos(__clc_vec8_fp64_t, __clc_vec8_fp64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_fp64_t __spirv_ocl_sincos(
    __clc_vec8_fp64_t args_0, __clc_vec8_fp64_t __private *args_1) {
  return __spirv_ocl_sincos(args_0, (__clc_vec8_fp64_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_sincos(__clc_vec16_fp64_t, __clc_vec16_fp64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_fp64_t __spirv_ocl_sincos(
    __clc_vec16_fp64_t args_0, __clc_vec16_fp64_t __private *args_1) {
  return __spirv_ocl_sincos(args_0, (__clc_vec16_fp64_t *)(args_1));
}

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_fp16_t
__spirv_ocl_sincos(__clc_fp16_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_fp16_t
__spirv_ocl_sincos(__clc_fp16_t args_0, __clc_fp16_t __private *args_1) {
  return __spirv_ocl_sincos(args_0, (__clc_fp16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_sincos(__clc_vec2_fp16_t, __clc_vec2_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t __spirv_ocl_sincos(
    __clc_vec2_fp16_t args_0, __clc_vec2_fp16_t __private *args_1) {
  return __spirv_ocl_sincos(args_0, (__clc_vec2_fp16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_sincos(__clc_vec3_fp16_t, __clc_vec3_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t __spirv_ocl_sincos(
    __clc_vec3_fp16_t args_0, __clc_vec3_fp16_t __private *args_1) {
  return __spirv_ocl_sincos(args_0, (__clc_vec3_fp16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_sincos(__clc_vec4_fp16_t, __clc_vec4_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t __spirv_ocl_sincos(
    __clc_vec4_fp16_t args_0, __clc_vec4_fp16_t __private *args_1) {
  return __spirv_ocl_sincos(args_0, (__clc_vec4_fp16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_sincos(__clc_vec8_fp16_t, __clc_vec8_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t __spirv_ocl_sincos(
    __clc_vec8_fp16_t args_0, __clc_vec8_fp16_t __private *args_1) {
  return __spirv_ocl_sincos(args_0, (__clc_vec8_fp16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_sincos(__clc_vec16_fp16_t, __clc_vec16_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t __spirv_ocl_sincos(
    __clc_vec16_fp16_t args_0, __clc_vec16_fp16_t __private *args_1) {
  return __spirv_ocl_sincos(args_0, (__clc_vec16_fp16_t *)(args_1));
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_float16_t
__spirv_ocl_sincos(__clc_float16_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_float16_t
__spirv_ocl_sincos(__clc_float16_t args_0, __clc_float16_t __private *args_1) {
  return __spirv_ocl_sincos(args_0, (__clc_float16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_float16_t
__spirv_ocl_sincos(__clc_vec2_float16_t, __clc_vec2_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_float16_t __spirv_ocl_sincos(
    __clc_vec2_float16_t args_0, __clc_vec2_float16_t __private *args_1) {
  return __spirv_ocl_sincos(args_0, (__clc_vec2_float16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_float16_t
__spirv_ocl_sincos(__clc_vec3_float16_t, __clc_vec3_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_float16_t __spirv_ocl_sincos(
    __clc_vec3_float16_t args_0, __clc_vec3_float16_t __private *args_1) {
  return __spirv_ocl_sincos(args_0, (__clc_vec3_float16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_float16_t
__spirv_ocl_sincos(__clc_vec4_float16_t, __clc_vec4_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_float16_t __spirv_ocl_sincos(
    __clc_vec4_float16_t args_0, __clc_vec4_float16_t __private *args_1) {
  return __spirv_ocl_sincos(args_0, (__clc_vec4_float16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_float16_t
__spirv_ocl_sincos(__clc_vec8_float16_t, __clc_vec8_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_float16_t __spirv_ocl_sincos(
    __clc_vec8_float16_t args_0, __clc_vec8_float16_t __private *args_1) {
  return __spirv_ocl_sincos(args_0, (__clc_vec8_float16_t *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_float16_t
__spirv_ocl_sincos(__clc_vec16_float16_t, __clc_vec16_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_float16_t __spirv_ocl_sincos(
    __clc_vec16_float16_t args_0, __clc_vec16_float16_t __private *args_1) {
  return __spirv_ocl_sincos(args_0, (__clc_vec16_float16_t *)(args_1));
}

#endif

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_fp32_t
__spirv_ocl_vload_half(__clc_size_t, __clc_fp16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_fp32_t __spirv_ocl_vload_half(
    __clc_size_t args_0, __clc_fp16_t const __private *args_1) {
  return __spirv_ocl_vload_half(args_0, (__clc_fp16_t const *)(args_1));
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_fp32_t
__spirv_ocl_vload_half(__clc_size_t, __clc_float16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_fp32_t __spirv_ocl_vload_half(
    __clc_size_t args_0, __clc_float16_t const __private *args_1) {
  return __spirv_ocl_vload_half(args_0, (__clc_float16_t const *)(args_1));
}

#endif

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_vload_halfn_Rfloat16(__clc_size_t, __clc_fp16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_fp32_t
__spirv_ocl_vload_halfn_Rfloat16(__clc_size_t args_0,
                                 __clc_fp16_t const __private *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat16(args_0,
                                          (__clc_fp16_t const *)(args_1));
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_vload_halfn_Rfloat16(__clc_size_t, __clc_float16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_fp32_t
__spirv_ocl_vload_halfn_Rfloat16(__clc_size_t args_0,
                                 __clc_float16_t const __private *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat16(args_0,
                                          (__clc_float16_t const *)(args_1));
}

#endif

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_vload_halfn_Rfloat2(__clc_size_t, __clc_fp16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_fp32_t
__spirv_ocl_vload_halfn_Rfloat2(__clc_size_t args_0,
                                __clc_fp16_t const __private *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat2(args_0,
                                         (__clc_fp16_t const *)(args_1));
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_vload_halfn_Rfloat2(__clc_size_t, __clc_float16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_fp32_t
__spirv_ocl_vload_halfn_Rfloat2(__clc_size_t args_0,
                                __clc_float16_t const __private *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat2(args_0,
                                         (__clc_float16_t const *)(args_1));
}

#endif

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_vload_halfn_Rfloat3(__clc_size_t, __clc_fp16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_fp32_t
__spirv_ocl_vload_halfn_Rfloat3(__clc_size_t args_0,
                                __clc_fp16_t const __private *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat3(args_0,
                                         (__clc_fp16_t const *)(args_1));
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_vload_halfn_Rfloat3(__clc_size_t, __clc_float16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_fp32_t
__spirv_ocl_vload_halfn_Rfloat3(__clc_size_t args_0,
                                __clc_float16_t const __private *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat3(args_0,
                                         (__clc_float16_t const *)(args_1));
}

#endif

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_vload_halfn_Rfloat4(__clc_size_t, __clc_fp16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_fp32_t
__spirv_ocl_vload_halfn_Rfloat4(__clc_size_t args_0,
                                __clc_fp16_t const __private *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat4(args_0,
                                         (__clc_fp16_t const *)(args_1));
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_vload_halfn_Rfloat4(__clc_size_t, __clc_float16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_fp32_t
__spirv_ocl_vload_halfn_Rfloat4(__clc_size_t args_0,
                                __clc_float16_t const __private *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat4(args_0,
                                         (__clc_float16_t const *)(args_1));
}

#endif

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_vload_halfn_Rfloat8(__clc_size_t, __clc_fp16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_fp32_t
__spirv_ocl_vload_halfn_Rfloat8(__clc_size_t args_0,
                                __clc_fp16_t const __private *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat8(args_0,
                                         (__clc_fp16_t const *)(args_1));
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_vload_halfn_Rfloat8(__clc_size_t, __clc_float16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_fp32_t
__spirv_ocl_vload_halfn_Rfloat8(__clc_size_t args_0,
                                __clc_float16_t const __private *args_1) {
  return __spirv_ocl_vload_halfn_Rfloat8(args_0,
                                         (__clc_float16_t const *)(args_1));
}

#endif

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_vloada_halfn_Rfloat16(__clc_size_t, __clc_fp16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_fp32_t
__spirv_ocl_vloada_halfn_Rfloat16(__clc_size_t args_0,
                                  __clc_fp16_t const __private *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat16(args_0,
                                           (__clc_fp16_t const *)(args_1));
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_vloada_halfn_Rfloat16(__clc_size_t, __clc_float16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_fp32_t
__spirv_ocl_vloada_halfn_Rfloat16(__clc_size_t args_0,
                                  __clc_float16_t const __private *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat16(args_0,
                                           (__clc_float16_t const *)(args_1));
}

#endif

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_vloada_halfn_Rfloat2(__clc_size_t, __clc_fp16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_fp32_t
__spirv_ocl_vloada_halfn_Rfloat2(__clc_size_t args_0,
                                 __clc_fp16_t const __private *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat2(args_0,
                                          (__clc_fp16_t const *)(args_1));
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_vloada_halfn_Rfloat2(__clc_size_t, __clc_float16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_fp32_t
__spirv_ocl_vloada_halfn_Rfloat2(__clc_size_t args_0,
                                 __clc_float16_t const __private *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat2(args_0,
                                          (__clc_float16_t const *)(args_1));
}

#endif

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_vloada_halfn_Rfloat3(__clc_size_t, __clc_fp16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_fp32_t
__spirv_ocl_vloada_halfn_Rfloat3(__clc_size_t args_0,
                                 __clc_fp16_t const __private *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat3(args_0,
                                          (__clc_fp16_t const *)(args_1));
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_vloada_halfn_Rfloat3(__clc_size_t, __clc_float16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_fp32_t
__spirv_ocl_vloada_halfn_Rfloat3(__clc_size_t args_0,
                                 __clc_float16_t const __private *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat3(args_0,
                                          (__clc_float16_t const *)(args_1));
}

#endif

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_vloada_halfn_Rfloat4(__clc_size_t, __clc_fp16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_fp32_t
__spirv_ocl_vloada_halfn_Rfloat4(__clc_size_t args_0,
                                 __clc_fp16_t const __private *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat4(args_0,
                                          (__clc_fp16_t const *)(args_1));
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_vloada_halfn_Rfloat4(__clc_size_t, __clc_float16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_fp32_t
__spirv_ocl_vloada_halfn_Rfloat4(__clc_size_t args_0,
                                 __clc_float16_t const __private *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat4(args_0,
                                          (__clc_float16_t const *)(args_1));
}

#endif

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_vloada_halfn_Rfloat8(__clc_size_t, __clc_fp16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_fp32_t
__spirv_ocl_vloada_halfn_Rfloat8(__clc_size_t args_0,
                                 __clc_fp16_t const __private *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat8(args_0,
                                          (__clc_fp16_t const *)(args_1));
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_vloada_halfn_Rfloat8(__clc_size_t, __clc_float16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_fp32_t
__spirv_ocl_vloada_halfn_Rfloat8(__clc_size_t args_0,
                                 __clc_float16_t const __private *args_1) {
  return __spirv_ocl_vloada_halfn_Rfloat8(args_0,
                                          (__clc_float16_t const *)(args_1));
}

#endif

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_int8_t
__spirv_ocl_vloadn_Rchar16(__clc_size_t, __clc_int8_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_int8_t
__spirv_ocl_vloadn_Rchar16(__clc_size_t args_0,
                           __clc_int8_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rchar16(args_0, (__clc_int8_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_int8_t
__spirv_ocl_vloadn_Rchar2(__clc_size_t, __clc_int8_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_int8_t
__spirv_ocl_vloadn_Rchar2(__clc_size_t args_0,
                          __clc_int8_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rchar2(args_0, (__clc_int8_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_int8_t
__spirv_ocl_vloadn_Rchar3(__clc_size_t, __clc_int8_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_int8_t
__spirv_ocl_vloadn_Rchar3(__clc_size_t args_0,
                          __clc_int8_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rchar3(args_0, (__clc_int8_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_int8_t
__spirv_ocl_vloadn_Rchar4(__clc_size_t, __clc_int8_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_int8_t
__spirv_ocl_vloadn_Rchar4(__clc_size_t args_0,
                          __clc_int8_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rchar4(args_0, (__clc_int8_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_int8_t
__spirv_ocl_vloadn_Rchar8(__clc_size_t, __clc_int8_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_int8_t
__spirv_ocl_vloadn_Rchar8(__clc_size_t args_0,
                          __clc_int8_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rchar8(args_0, (__clc_int8_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_fp64_t
__spirv_ocl_vloadn_Rdouble16(__clc_size_t, __clc_fp64_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_fp64_t
__spirv_ocl_vloadn_Rdouble16(__clc_size_t args_0,
                             __clc_fp64_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rdouble16(args_0, (__clc_fp64_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_fp64_t
__spirv_ocl_vloadn_Rdouble2(__clc_size_t, __clc_fp64_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_fp64_t
__spirv_ocl_vloadn_Rdouble2(__clc_size_t args_0,
                            __clc_fp64_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rdouble2(args_0, (__clc_fp64_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_fp64_t
__spirv_ocl_vloadn_Rdouble3(__clc_size_t, __clc_fp64_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_fp64_t
__spirv_ocl_vloadn_Rdouble3(__clc_size_t args_0,
                            __clc_fp64_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rdouble3(args_0, (__clc_fp64_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_fp64_t
__spirv_ocl_vloadn_Rdouble4(__clc_size_t, __clc_fp64_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_fp64_t
__spirv_ocl_vloadn_Rdouble4(__clc_size_t args_0,
                            __clc_fp64_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rdouble4(args_0, (__clc_fp64_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_fp64_t
__spirv_ocl_vloadn_Rdouble8(__clc_size_t, __clc_fp64_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_fp64_t
__spirv_ocl_vloadn_Rdouble8(__clc_size_t args_0,
                            __clc_fp64_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rdouble8(args_0, (__clc_fp64_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_fp32_t
__spirv_ocl_vloadn_Rfloat16(__clc_size_t, __clc_fp32_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_fp32_t
__spirv_ocl_vloadn_Rfloat16(__clc_size_t args_0,
                            __clc_fp32_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rfloat16(args_0, (__clc_fp32_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_fp32_t
__spirv_ocl_vloadn_Rfloat2(__clc_size_t, __clc_fp32_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_fp32_t
__spirv_ocl_vloadn_Rfloat2(__clc_size_t args_0,
                           __clc_fp32_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rfloat2(args_0, (__clc_fp32_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_fp32_t
__spirv_ocl_vloadn_Rfloat3(__clc_size_t, __clc_fp32_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_fp32_t
__spirv_ocl_vloadn_Rfloat3(__clc_size_t args_0,
                           __clc_fp32_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rfloat3(args_0, (__clc_fp32_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_fp32_t
__spirv_ocl_vloadn_Rfloat4(__clc_size_t, __clc_fp32_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_fp32_t
__spirv_ocl_vloadn_Rfloat4(__clc_size_t args_0,
                           __clc_fp32_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rfloat4(args_0, (__clc_fp32_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_fp32_t
__spirv_ocl_vloadn_Rfloat8(__clc_size_t, __clc_fp32_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_fp32_t
__spirv_ocl_vloadn_Rfloat8(__clc_size_t args_0,
                           __clc_fp32_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rfloat8(args_0, (__clc_fp32_t const *)(args_1));
}

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_fp16_t
__spirv_ocl_vloadn_Rhalf16(__clc_size_t, __clc_fp16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_fp16_t
__spirv_ocl_vloadn_Rhalf16(__clc_size_t args_0,
                           __clc_fp16_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rhalf16(args_0, (__clc_fp16_t const *)(args_1));
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_float16_t
__spirv_ocl_vloadn_Rhalf16(__clc_size_t, __clc_float16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_float16_t
__spirv_ocl_vloadn_Rhalf16(__clc_size_t args_0,
                           __clc_float16_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rhalf16(args_0, (__clc_float16_t const *)(args_1));
}

#endif

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_fp16_t
__spirv_ocl_vloadn_Rhalf2(__clc_size_t, __clc_fp16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_fp16_t
__spirv_ocl_vloadn_Rhalf2(__clc_size_t args_0,
                          __clc_fp16_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rhalf2(args_0, (__clc_fp16_t const *)(args_1));
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_float16_t
__spirv_ocl_vloadn_Rhalf2(__clc_size_t, __clc_float16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_float16_t
__spirv_ocl_vloadn_Rhalf2(__clc_size_t args_0,
                          __clc_float16_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rhalf2(args_0, (__clc_float16_t const *)(args_1));
}

#endif

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_fp16_t
__spirv_ocl_vloadn_Rhalf3(__clc_size_t, __clc_fp16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_fp16_t
__spirv_ocl_vloadn_Rhalf3(__clc_size_t args_0,
                          __clc_fp16_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rhalf3(args_0, (__clc_fp16_t const *)(args_1));
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_float16_t
__spirv_ocl_vloadn_Rhalf3(__clc_size_t, __clc_float16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_float16_t
__spirv_ocl_vloadn_Rhalf3(__clc_size_t args_0,
                          __clc_float16_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rhalf3(args_0, (__clc_float16_t const *)(args_1));
}

#endif

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_fp16_t
__spirv_ocl_vloadn_Rhalf4(__clc_size_t, __clc_fp16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_fp16_t
__spirv_ocl_vloadn_Rhalf4(__clc_size_t args_0,
                          __clc_fp16_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rhalf4(args_0, (__clc_fp16_t const *)(args_1));
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_float16_t
__spirv_ocl_vloadn_Rhalf4(__clc_size_t, __clc_float16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_float16_t
__spirv_ocl_vloadn_Rhalf4(__clc_size_t args_0,
                          __clc_float16_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rhalf4(args_0, (__clc_float16_t const *)(args_1));
}

#endif

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_fp16_t
__spirv_ocl_vloadn_Rhalf8(__clc_size_t, __clc_fp16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_fp16_t
__spirv_ocl_vloadn_Rhalf8(__clc_size_t args_0,
                          __clc_fp16_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rhalf8(args_0, (__clc_fp16_t const *)(args_1));
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_float16_t
__spirv_ocl_vloadn_Rhalf8(__clc_size_t, __clc_float16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_float16_t
__spirv_ocl_vloadn_Rhalf8(__clc_size_t args_0,
                          __clc_float16_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rhalf8(args_0, (__clc_float16_t const *)(args_1));
}

#endif

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_int32_t
__spirv_ocl_vloadn_Rint16(__clc_size_t, __clc_int32_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_int32_t
__spirv_ocl_vloadn_Rint16(__clc_size_t args_0,
                          __clc_int32_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rint16(args_0, (__clc_int32_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_int32_t
__spirv_ocl_vloadn_Rint2(__clc_size_t, __clc_int32_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_int32_t
__spirv_ocl_vloadn_Rint2(__clc_size_t args_0,
                         __clc_int32_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rint2(args_0, (__clc_int32_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_int32_t
__spirv_ocl_vloadn_Rint3(__clc_size_t, __clc_int32_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_int32_t
__spirv_ocl_vloadn_Rint3(__clc_size_t args_0,
                         __clc_int32_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rint3(args_0, (__clc_int32_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_int32_t
__spirv_ocl_vloadn_Rint4(__clc_size_t, __clc_int32_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_int32_t
__spirv_ocl_vloadn_Rint4(__clc_size_t args_0,
                         __clc_int32_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rint4(args_0, (__clc_int32_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_int32_t
__spirv_ocl_vloadn_Rint8(__clc_size_t, __clc_int32_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_int32_t
__spirv_ocl_vloadn_Rint8(__clc_size_t args_0,
                         __clc_int32_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rint8(args_0, (__clc_int32_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_int64_t
__spirv_ocl_vloadn_Rlong16(__clc_size_t, __clc_int64_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_int64_t
__spirv_ocl_vloadn_Rlong16(__clc_size_t args_0,
                           __clc_int64_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rlong16(args_0, (__clc_int64_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_int64_t
__spirv_ocl_vloadn_Rlong2(__clc_size_t, __clc_int64_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_int64_t
__spirv_ocl_vloadn_Rlong2(__clc_size_t args_0,
                          __clc_int64_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rlong2(args_0, (__clc_int64_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_int64_t
__spirv_ocl_vloadn_Rlong3(__clc_size_t, __clc_int64_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_int64_t
__spirv_ocl_vloadn_Rlong3(__clc_size_t args_0,
                          __clc_int64_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rlong3(args_0, (__clc_int64_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_int64_t
__spirv_ocl_vloadn_Rlong4(__clc_size_t, __clc_int64_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_int64_t
__spirv_ocl_vloadn_Rlong4(__clc_size_t args_0,
                          __clc_int64_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rlong4(args_0, (__clc_int64_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_int64_t
__spirv_ocl_vloadn_Rlong8(__clc_size_t, __clc_int64_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_int64_t
__spirv_ocl_vloadn_Rlong8(__clc_size_t args_0,
                          __clc_int64_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rlong8(args_0, (__clc_int64_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_int16_t
__spirv_ocl_vloadn_Rshort16(__clc_size_t, __clc_int16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_int16_t
__spirv_ocl_vloadn_Rshort16(__clc_size_t args_0,
                            __clc_int16_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rshort16(args_0, (__clc_int16_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_int16_t
__spirv_ocl_vloadn_Rshort2(__clc_size_t, __clc_int16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_int16_t
__spirv_ocl_vloadn_Rshort2(__clc_size_t args_0,
                           __clc_int16_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rshort2(args_0, (__clc_int16_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_int16_t
__spirv_ocl_vloadn_Rshort3(__clc_size_t, __clc_int16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_int16_t
__spirv_ocl_vloadn_Rshort3(__clc_size_t args_0,
                           __clc_int16_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rshort3(args_0, (__clc_int16_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_int16_t
__spirv_ocl_vloadn_Rshort4(__clc_size_t, __clc_int16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_int16_t
__spirv_ocl_vloadn_Rshort4(__clc_size_t args_0,
                           __clc_int16_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rshort4(args_0, (__clc_int16_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_int16_t
__spirv_ocl_vloadn_Rshort8(__clc_size_t, __clc_int16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_int16_t
__spirv_ocl_vloadn_Rshort8(__clc_size_t args_0,
                           __clc_int16_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rshort8(args_0, (__clc_int16_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_uint8_t
__spirv_ocl_vloadn_Ruchar16(__clc_size_t, __clc_uint8_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_uint8_t
__spirv_ocl_vloadn_Ruchar16(__clc_size_t args_0,
                            __clc_uint8_t const __private *args_1) {
  return __spirv_ocl_vloadn_Ruchar16(args_0, (__clc_uint8_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_uint8_t
__spirv_ocl_vloadn_Ruchar2(__clc_size_t, __clc_uint8_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_uint8_t
__spirv_ocl_vloadn_Ruchar2(__clc_size_t args_0,
                           __clc_uint8_t const __private *args_1) {
  return __spirv_ocl_vloadn_Ruchar2(args_0, (__clc_uint8_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_uint8_t
__spirv_ocl_vloadn_Ruchar3(__clc_size_t, __clc_uint8_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_uint8_t
__spirv_ocl_vloadn_Ruchar3(__clc_size_t args_0,
                           __clc_uint8_t const __private *args_1) {
  return __spirv_ocl_vloadn_Ruchar3(args_0, (__clc_uint8_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_uint8_t
__spirv_ocl_vloadn_Ruchar4(__clc_size_t, __clc_uint8_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_uint8_t
__spirv_ocl_vloadn_Ruchar4(__clc_size_t args_0,
                           __clc_uint8_t const __private *args_1) {
  return __spirv_ocl_vloadn_Ruchar4(args_0, (__clc_uint8_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_uint8_t
__spirv_ocl_vloadn_Ruchar8(__clc_size_t, __clc_uint8_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_uint8_t
__spirv_ocl_vloadn_Ruchar8(__clc_size_t args_0,
                           __clc_uint8_t const __private *args_1) {
  return __spirv_ocl_vloadn_Ruchar8(args_0, (__clc_uint8_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_uint32_t
__spirv_ocl_vloadn_Ruint16(__clc_size_t, __clc_uint32_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_uint32_t
__spirv_ocl_vloadn_Ruint16(__clc_size_t args_0,
                           __clc_uint32_t const __private *args_1) {
  return __spirv_ocl_vloadn_Ruint16(args_0, (__clc_uint32_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_uint32_t
__spirv_ocl_vloadn_Ruint2(__clc_size_t, __clc_uint32_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_uint32_t
__spirv_ocl_vloadn_Ruint2(__clc_size_t args_0,
                          __clc_uint32_t const __private *args_1) {
  return __spirv_ocl_vloadn_Ruint2(args_0, (__clc_uint32_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_uint32_t
__spirv_ocl_vloadn_Ruint3(__clc_size_t, __clc_uint32_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_uint32_t
__spirv_ocl_vloadn_Ruint3(__clc_size_t args_0,
                          __clc_uint32_t const __private *args_1) {
  return __spirv_ocl_vloadn_Ruint3(args_0, (__clc_uint32_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_uint32_t
__spirv_ocl_vloadn_Ruint4(__clc_size_t, __clc_uint32_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_uint32_t
__spirv_ocl_vloadn_Ruint4(__clc_size_t args_0,
                          __clc_uint32_t const __private *args_1) {
  return __spirv_ocl_vloadn_Ruint4(args_0, (__clc_uint32_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_uint32_t
__spirv_ocl_vloadn_Ruint8(__clc_size_t, __clc_uint32_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_uint32_t
__spirv_ocl_vloadn_Ruint8(__clc_size_t args_0,
                          __clc_uint32_t const __private *args_1) {
  return __spirv_ocl_vloadn_Ruint8(args_0, (__clc_uint32_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_uint64_t
__spirv_ocl_vloadn_Rulong16(__clc_size_t, __clc_uint64_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_uint64_t
__spirv_ocl_vloadn_Rulong16(__clc_size_t args_0,
                            __clc_uint64_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rulong16(args_0, (__clc_uint64_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_uint64_t
__spirv_ocl_vloadn_Rulong2(__clc_size_t, __clc_uint64_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_uint64_t
__spirv_ocl_vloadn_Rulong2(__clc_size_t args_0,
                           __clc_uint64_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rulong2(args_0, (__clc_uint64_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_uint64_t
__spirv_ocl_vloadn_Rulong3(__clc_size_t, __clc_uint64_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_uint64_t
__spirv_ocl_vloadn_Rulong3(__clc_size_t args_0,
                           __clc_uint64_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rulong3(args_0, (__clc_uint64_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_uint64_t
__spirv_ocl_vloadn_Rulong4(__clc_size_t, __clc_uint64_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_uint64_t
__spirv_ocl_vloadn_Rulong4(__clc_size_t args_0,
                           __clc_uint64_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rulong4(args_0, (__clc_uint64_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_uint64_t
__spirv_ocl_vloadn_Rulong8(__clc_size_t, __clc_uint64_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_uint64_t
__spirv_ocl_vloadn_Rulong8(__clc_size_t args_0,
                           __clc_uint64_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rulong8(args_0, (__clc_uint64_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec16_uint16_t
__spirv_ocl_vloadn_Rushort16(__clc_size_t, __clc_uint16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec16_uint16_t
__spirv_ocl_vloadn_Rushort16(__clc_size_t args_0,
                             __clc_uint16_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rushort16(args_0, (__clc_uint16_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec2_uint16_t
__spirv_ocl_vloadn_Rushort2(__clc_size_t, __clc_uint16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec2_uint16_t
__spirv_ocl_vloadn_Rushort2(__clc_size_t args_0,
                            __clc_uint16_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rushort2(args_0, (__clc_uint16_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec3_uint16_t
__spirv_ocl_vloadn_Rushort3(__clc_size_t, __clc_uint16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec3_uint16_t
__spirv_ocl_vloadn_Rushort3(__clc_size_t args_0,
                            __clc_uint16_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rushort3(args_0, (__clc_uint16_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec4_uint16_t
__spirv_ocl_vloadn_Rushort4(__clc_size_t, __clc_uint16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec4_uint16_t
__spirv_ocl_vloadn_Rushort4(__clc_size_t args_0,
                            __clc_uint16_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rushort4(args_0, (__clc_uint16_t const *)(args_1));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL __clc_vec8_uint16_t
__spirv_ocl_vloadn_Rushort8(__clc_size_t, __clc_uint16_t const *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF __clc_vec8_uint16_t
__spirv_ocl_vloadn_Rushort8(__clc_size_t args_0,
                            __clc_uint16_t const __private *args_1) {
  return __spirv_ocl_vloadn_Rushort8(args_0, (__clc_uint16_t const *)(args_1));
}

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_half(__clc_fp32_t, __clc_size_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_half(__clc_fp32_t args_0, __clc_size_t args_1,
                        __clc_fp16_t __private *args_2) {
  __spirv_ocl_vstore_half(args_0, args_1, (__clc_fp16_t *)(args_2));
}

#endif

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_half(__clc_fp64_t, __clc_size_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_half(__clc_fp64_t args_0, __clc_size_t args_1,
                        __clc_fp16_t __private *args_2) {
  __spirv_ocl_vstore_half(args_0, args_1, (__clc_fp16_t *)(args_2));
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_half(__clc_fp32_t, __clc_size_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_half(__clc_fp32_t args_0, __clc_size_t args_1,
                        __clc_float16_t __private *args_2) {
  __spirv_ocl_vstore_half(args_0, args_1, (__clc_float16_t *)(args_2));
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_half(__clc_fp64_t, __clc_size_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_half(__clc_fp64_t args_0, __clc_size_t args_1,
                        __clc_float16_t __private *args_2) {
  __spirv_ocl_vstore_half(args_0, args_1, (__clc_float16_t *)(args_2));
}

#endif

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_half_r(__clc_fp32_t, __clc_size_t, __clc_fp16_t *,
                          __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_half_r(__clc_fp32_t args_0, __clc_size_t args_1,
                          __clc_fp16_t __private *args_2,
                          __clc_uint32_t args_3) {
  __spirv_ocl_vstore_half_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

#endif

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_half_r(__clc_fp64_t, __clc_size_t, __clc_fp16_t *,
                          __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_half_r(__clc_fp64_t args_0, __clc_size_t args_1,
                          __clc_fp16_t __private *args_2,
                          __clc_uint32_t args_3) {
  __spirv_ocl_vstore_half_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_half_r(__clc_fp32_t, __clc_size_t, __clc_float16_t *,
                          __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_half_r(__clc_fp32_t args_0, __clc_size_t args_1,
                          __clc_float16_t __private *args_2,
                          __clc_uint32_t args_3) {
  __spirv_ocl_vstore_half_r(args_0, args_1, (__clc_float16_t *)(args_2),
                            args_3);
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_half_r(__clc_fp64_t, __clc_size_t, __clc_float16_t *,
                          __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_half_r(__clc_fp64_t args_0, __clc_size_t args_1,
                          __clc_float16_t __private *args_2,
                          __clc_uint32_t args_3) {
  __spirv_ocl_vstore_half_r(args_0, args_1, (__clc_float16_t *)(args_2),
                            args_3);
}

#endif

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec2_fp32_t, __clc_size_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec2_fp32_t args_0, __clc_size_t args_1,
                         __clc_fp16_t __private *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec3_fp32_t, __clc_size_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec3_fp32_t args_0, __clc_size_t args_1,
                         __clc_fp16_t __private *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec4_fp32_t, __clc_size_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec4_fp32_t args_0, __clc_size_t args_1,
                         __clc_fp16_t __private *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec8_fp32_t, __clc_size_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec8_fp32_t args_0, __clc_size_t args_1,
                         __clc_fp16_t __private *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec16_fp32_t, __clc_size_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec16_fp32_t args_0, __clc_size_t args_1,
                         __clc_fp16_t __private *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

#endif

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec2_fp64_t, __clc_size_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec2_fp64_t args_0, __clc_size_t args_1,
                         __clc_fp16_t __private *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec3_fp64_t, __clc_size_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec3_fp64_t args_0, __clc_size_t args_1,
                         __clc_fp16_t __private *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec4_fp64_t, __clc_size_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec4_fp64_t args_0, __clc_size_t args_1,
                         __clc_fp16_t __private *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec8_fp64_t, __clc_size_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec8_fp64_t args_0, __clc_size_t args_1,
                         __clc_fp16_t __private *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec16_fp64_t, __clc_size_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec16_fp64_t args_0, __clc_size_t args_1,
                         __clc_fp16_t __private *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec2_fp32_t, __clc_size_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec2_fp32_t args_0, __clc_size_t args_1,
                         __clc_float16_t __private *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_float16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec3_fp32_t, __clc_size_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec3_fp32_t args_0, __clc_size_t args_1,
                         __clc_float16_t __private *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_float16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec4_fp32_t, __clc_size_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec4_fp32_t args_0, __clc_size_t args_1,
                         __clc_float16_t __private *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_float16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec8_fp32_t, __clc_size_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec8_fp32_t args_0, __clc_size_t args_1,
                         __clc_float16_t __private *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_float16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec16_fp32_t, __clc_size_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec16_fp32_t args_0, __clc_size_t args_1,
                         __clc_float16_t __private *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_float16_t *)(args_2));
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec2_fp64_t, __clc_size_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec2_fp64_t args_0, __clc_size_t args_1,
                         __clc_float16_t __private *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_float16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec3_fp64_t, __clc_size_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec3_fp64_t args_0, __clc_size_t args_1,
                         __clc_float16_t __private *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_float16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec4_fp64_t, __clc_size_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec4_fp64_t args_0, __clc_size_t args_1,
                         __clc_float16_t __private *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_float16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec8_fp64_t, __clc_size_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec8_fp64_t args_0, __clc_size_t args_1,
                         __clc_float16_t __private *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_float16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn(__clc_vec16_fp64_t, __clc_size_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn(__clc_vec16_fp64_t args_0, __clc_size_t args_1,
                         __clc_float16_t __private *args_2) {
  __spirv_ocl_vstore_halfn(args_0, args_1, (__clc_float16_t *)(args_2));
}

#endif

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn_r(__clc_vec2_fp32_t, __clc_size_t, __clc_fp16_t *,
                           __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec2_fp32_t args_0, __clc_size_t args_1,
                           __clc_fp16_t __private *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn_r(__clc_vec3_fp32_t, __clc_size_t, __clc_fp16_t *,
                           __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec3_fp32_t args_0, __clc_size_t args_1,
                           __clc_fp16_t __private *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn_r(__clc_vec4_fp32_t, __clc_size_t, __clc_fp16_t *,
                           __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec4_fp32_t args_0, __clc_size_t args_1,
                           __clc_fp16_t __private *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn_r(__clc_vec8_fp32_t, __clc_size_t, __clc_fp16_t *,
                           __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec8_fp32_t args_0, __clc_size_t args_1,
                           __clc_fp16_t __private *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn_r(__clc_vec16_fp32_t, __clc_size_t, __clc_fp16_t *,
                           __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec16_fp32_t args_0, __clc_size_t args_1,
                           __clc_fp16_t __private *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

#endif

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn_r(__clc_vec2_fp64_t, __clc_size_t, __clc_fp16_t *,
                           __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec2_fp64_t args_0, __clc_size_t args_1,
                           __clc_fp16_t __private *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn_r(__clc_vec3_fp64_t, __clc_size_t, __clc_fp16_t *,
                           __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec3_fp64_t args_0, __clc_size_t args_1,
                           __clc_fp16_t __private *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn_r(__clc_vec4_fp64_t, __clc_size_t, __clc_fp16_t *,
                           __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec4_fp64_t args_0, __clc_size_t args_1,
                           __clc_fp16_t __private *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn_r(__clc_vec8_fp64_t, __clc_size_t, __clc_fp16_t *,
                           __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec8_fp64_t args_0, __clc_size_t args_1,
                           __clc_fp16_t __private *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn_r(__clc_vec16_fp64_t, __clc_size_t, __clc_fp16_t *,
                           __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec16_fp64_t args_0, __clc_size_t args_1,
                           __clc_fp16_t __private *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn_r(__clc_vec2_fp32_t, __clc_size_t, __clc_float16_t *,
                           __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec2_fp32_t args_0, __clc_size_t args_1,
                           __clc_float16_t __private *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_float16_t *)(args_2),
                             args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn_r(__clc_vec3_fp32_t, __clc_size_t, __clc_float16_t *,
                           __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec3_fp32_t args_0, __clc_size_t args_1,
                           __clc_float16_t __private *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_float16_t *)(args_2),
                             args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn_r(__clc_vec4_fp32_t, __clc_size_t, __clc_float16_t *,
                           __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec4_fp32_t args_0, __clc_size_t args_1,
                           __clc_float16_t __private *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_float16_t *)(args_2),
                             args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn_r(__clc_vec8_fp32_t, __clc_size_t, __clc_float16_t *,
                           __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec8_fp32_t args_0, __clc_size_t args_1,
                           __clc_float16_t __private *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_float16_t *)(args_2),
                             args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn_r(__clc_vec16_fp32_t, __clc_size_t, __clc_float16_t *,
                           __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec16_fp32_t args_0, __clc_size_t args_1,
                           __clc_float16_t __private *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_float16_t *)(args_2),
                             args_3);
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn_r(__clc_vec2_fp64_t, __clc_size_t, __clc_float16_t *,
                           __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec2_fp64_t args_0, __clc_size_t args_1,
                           __clc_float16_t __private *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_float16_t *)(args_2),
                             args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn_r(__clc_vec3_fp64_t, __clc_size_t, __clc_float16_t *,
                           __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec3_fp64_t args_0, __clc_size_t args_1,
                           __clc_float16_t __private *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_float16_t *)(args_2),
                             args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn_r(__clc_vec4_fp64_t, __clc_size_t, __clc_float16_t *,
                           __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec4_fp64_t args_0, __clc_size_t args_1,
                           __clc_float16_t __private *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_float16_t *)(args_2),
                             args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn_r(__clc_vec8_fp64_t, __clc_size_t, __clc_float16_t *,
                           __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec8_fp64_t args_0, __clc_size_t args_1,
                           __clc_float16_t __private *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_float16_t *)(args_2),
                             args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstore_halfn_r(__clc_vec16_fp64_t, __clc_size_t, __clc_float16_t *,
                           __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstore_halfn_r(__clc_vec16_fp64_t args_0, __clc_size_t args_1,
                           __clc_float16_t __private *args_2,
                           __clc_uint32_t args_3) {
  __spirv_ocl_vstore_halfn_r(args_0, args_1, (__clc_float16_t *)(args_2),
                             args_3);
}

#endif

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec2_fp32_t, __clc_size_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec2_fp32_t args_0, __clc_size_t args_1,
                          __clc_fp16_t __private *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec3_fp32_t, __clc_size_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec3_fp32_t args_0, __clc_size_t args_1,
                          __clc_fp16_t __private *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec4_fp32_t, __clc_size_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec4_fp32_t args_0, __clc_size_t args_1,
                          __clc_fp16_t __private *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec8_fp32_t, __clc_size_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec8_fp32_t args_0, __clc_size_t args_1,
                          __clc_fp16_t __private *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec16_fp32_t, __clc_size_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec16_fp32_t args_0, __clc_size_t args_1,
                          __clc_fp16_t __private *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

#endif

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec2_fp64_t, __clc_size_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec2_fp64_t args_0, __clc_size_t args_1,
                          __clc_fp16_t __private *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec3_fp64_t, __clc_size_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec3_fp64_t args_0, __clc_size_t args_1,
                          __clc_fp16_t __private *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec4_fp64_t, __clc_size_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec4_fp64_t args_0, __clc_size_t args_1,
                          __clc_fp16_t __private *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec8_fp64_t, __clc_size_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec8_fp64_t args_0, __clc_size_t args_1,
                          __clc_fp16_t __private *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec16_fp64_t, __clc_size_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec16_fp64_t args_0, __clc_size_t args_1,
                          __clc_fp16_t __private *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_fp16_t *)(args_2));
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec2_fp32_t, __clc_size_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec2_fp32_t args_0, __clc_size_t args_1,
                          __clc_float16_t __private *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_float16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec3_fp32_t, __clc_size_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec3_fp32_t args_0, __clc_size_t args_1,
                          __clc_float16_t __private *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_float16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec4_fp32_t, __clc_size_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec4_fp32_t args_0, __clc_size_t args_1,
                          __clc_float16_t __private *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_float16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec8_fp32_t, __clc_size_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec8_fp32_t args_0, __clc_size_t args_1,
                          __clc_float16_t __private *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_float16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec16_fp32_t, __clc_size_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec16_fp32_t args_0, __clc_size_t args_1,
                          __clc_float16_t __private *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_float16_t *)(args_2));
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec2_fp64_t, __clc_size_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec2_fp64_t args_0, __clc_size_t args_1,
                          __clc_float16_t __private *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_float16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec3_fp64_t, __clc_size_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec3_fp64_t args_0, __clc_size_t args_1,
                          __clc_float16_t __private *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_float16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec4_fp64_t, __clc_size_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec4_fp64_t args_0, __clc_size_t args_1,
                          __clc_float16_t __private *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_float16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec8_fp64_t, __clc_size_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec8_fp64_t args_0, __clc_size_t args_1,
                          __clc_float16_t __private *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_float16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn(__clc_vec16_fp64_t, __clc_size_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn(__clc_vec16_fp64_t args_0, __clc_size_t args_1,
                          __clc_float16_t __private *args_2) {
  __spirv_ocl_vstorea_halfn(args_0, args_1, (__clc_float16_t *)(args_2));
}

#endif

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec2_fp32_t, __clc_size_t, __clc_fp16_t *,
                            __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec2_fp32_t args_0, __clc_size_t args_1,
                            __clc_fp16_t __private *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec3_fp32_t, __clc_size_t, __clc_fp16_t *,
                            __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec3_fp32_t args_0, __clc_size_t args_1,
                            __clc_fp16_t __private *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec4_fp32_t, __clc_size_t, __clc_fp16_t *,
                            __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec4_fp32_t args_0, __clc_size_t args_1,
                            __clc_fp16_t __private *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec8_fp32_t, __clc_size_t, __clc_fp16_t *,
                            __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec8_fp32_t args_0, __clc_size_t args_1,
                            __clc_fp16_t __private *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec16_fp32_t, __clc_size_t, __clc_fp16_t *,
                            __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec16_fp32_t args_0, __clc_size_t args_1,
                            __clc_fp16_t __private *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

#endif

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec2_fp64_t, __clc_size_t, __clc_fp16_t *,
                            __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec2_fp64_t args_0, __clc_size_t args_1,
                            __clc_fp16_t __private *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec3_fp64_t, __clc_size_t, __clc_fp16_t *,
                            __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec3_fp64_t args_0, __clc_size_t args_1,
                            __clc_fp16_t __private *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec4_fp64_t, __clc_size_t, __clc_fp16_t *,
                            __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec4_fp64_t args_0, __clc_size_t args_1,
                            __clc_fp16_t __private *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec8_fp64_t, __clc_size_t, __clc_fp16_t *,
                            __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec8_fp64_t args_0, __clc_size_t args_1,
                            __clc_fp16_t __private *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec16_fp64_t, __clc_size_t, __clc_fp16_t *,
                            __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec16_fp64_t args_0, __clc_size_t args_1,
                            __clc_fp16_t __private *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_fp16_t *)(args_2), args_3);
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec2_fp32_t, __clc_size_t, __clc_float16_t *,
                            __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec2_fp32_t args_0, __clc_size_t args_1,
                            __clc_float16_t __private *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_float16_t *)(args_2),
                              args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec3_fp32_t, __clc_size_t, __clc_float16_t *,
                            __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec3_fp32_t args_0, __clc_size_t args_1,
                            __clc_float16_t __private *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_float16_t *)(args_2),
                              args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec4_fp32_t, __clc_size_t, __clc_float16_t *,
                            __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec4_fp32_t args_0, __clc_size_t args_1,
                            __clc_float16_t __private *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_float16_t *)(args_2),
                              args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec8_fp32_t, __clc_size_t, __clc_float16_t *,
                            __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec8_fp32_t args_0, __clc_size_t args_1,
                            __clc_float16_t __private *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_float16_t *)(args_2),
                              args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec16_fp32_t, __clc_size_t, __clc_float16_t *,
                            __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec16_fp32_t args_0, __clc_size_t args_1,
                            __clc_float16_t __private *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_float16_t *)(args_2),
                              args_3);
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec2_fp64_t, __clc_size_t, __clc_float16_t *,
                            __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec2_fp64_t args_0, __clc_size_t args_1,
                            __clc_float16_t __private *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_float16_t *)(args_2),
                              args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec3_fp64_t, __clc_size_t, __clc_float16_t *,
                            __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec3_fp64_t args_0, __clc_size_t args_1,
                            __clc_float16_t __private *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_float16_t *)(args_2),
                              args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec4_fp64_t, __clc_size_t, __clc_float16_t *,
                            __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec4_fp64_t args_0, __clc_size_t args_1,
                            __clc_float16_t __private *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_float16_t *)(args_2),
                              args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec8_fp64_t, __clc_size_t, __clc_float16_t *,
                            __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec8_fp64_t args_0, __clc_size_t args_1,
                            __clc_float16_t __private *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_float16_t *)(args_2),
                              args_3);
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstorea_halfn_r(__clc_vec16_fp64_t, __clc_size_t, __clc_float16_t *,
                            __clc_uint32_t);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstorea_halfn_r(__clc_vec16_fp64_t args_0, __clc_size_t args_1,
                            __clc_float16_t __private *args_2,
                            __clc_uint32_t args_3) {
  __spirv_ocl_vstorea_halfn_r(args_0, args_1, (__clc_float16_t *)(args_2),
                              args_3);
}

#endif

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_int8_t, __clc_size_t, __clc_int8_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec2_int8_t args_0, __clc_size_t args_1,
                    __clc_int8_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_int8_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_int8_t, __clc_size_t, __clc_int8_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec3_int8_t args_0, __clc_size_t args_1,
                    __clc_int8_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_int8_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_int8_t, __clc_size_t, __clc_int8_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec4_int8_t args_0, __clc_size_t args_1,
                    __clc_int8_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_int8_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_int8_t, __clc_size_t, __clc_int8_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec8_int8_t args_0, __clc_size_t args_1,
                    __clc_int8_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_int8_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec16_int8_t, __clc_size_t, __clc_int8_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec16_int8_t args_0, __clc_size_t args_1,
                    __clc_int8_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_int8_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_int16_t, __clc_size_t, __clc_int16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec2_int16_t args_0, __clc_size_t args_1,
                    __clc_int16_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_int16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_int16_t, __clc_size_t, __clc_int16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec3_int16_t args_0, __clc_size_t args_1,
                    __clc_int16_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_int16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_int16_t, __clc_size_t, __clc_int16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec4_int16_t args_0, __clc_size_t args_1,
                    __clc_int16_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_int16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_int16_t, __clc_size_t, __clc_int16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec8_int16_t args_0, __clc_size_t args_1,
                    __clc_int16_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_int16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec16_int16_t, __clc_size_t, __clc_int16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec16_int16_t args_0, __clc_size_t args_1,
                    __clc_int16_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_int16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_int32_t, __clc_size_t, __clc_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec2_int32_t args_0, __clc_size_t args_1,
                    __clc_int32_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_int32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_int32_t, __clc_size_t, __clc_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec3_int32_t args_0, __clc_size_t args_1,
                    __clc_int32_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_int32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_int32_t, __clc_size_t, __clc_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec4_int32_t args_0, __clc_size_t args_1,
                    __clc_int32_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_int32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_int32_t, __clc_size_t, __clc_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec8_int32_t args_0, __clc_size_t args_1,
                    __clc_int32_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_int32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec16_int32_t, __clc_size_t, __clc_int32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec16_int32_t args_0, __clc_size_t args_1,
                    __clc_int32_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_int32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_int64_t, __clc_size_t, __clc_int64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec2_int64_t args_0, __clc_size_t args_1,
                    __clc_int64_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_int64_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_int64_t, __clc_size_t, __clc_int64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec3_int64_t args_0, __clc_size_t args_1,
                    __clc_int64_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_int64_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_int64_t, __clc_size_t, __clc_int64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec4_int64_t args_0, __clc_size_t args_1,
                    __clc_int64_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_int64_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_int64_t, __clc_size_t, __clc_int64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec8_int64_t args_0, __clc_size_t args_1,
                    __clc_int64_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_int64_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec16_int64_t, __clc_size_t, __clc_int64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec16_int64_t args_0, __clc_size_t args_1,
                    __clc_int64_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_int64_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_uint8_t, __clc_size_t, __clc_uint8_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec2_uint8_t args_0, __clc_size_t args_1,
                    __clc_uint8_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_uint8_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_uint8_t, __clc_size_t, __clc_uint8_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec3_uint8_t args_0, __clc_size_t args_1,
                    __clc_uint8_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_uint8_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_uint8_t, __clc_size_t, __clc_uint8_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec4_uint8_t args_0, __clc_size_t args_1,
                    __clc_uint8_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_uint8_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_uint8_t, __clc_size_t, __clc_uint8_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec8_uint8_t args_0, __clc_size_t args_1,
                    __clc_uint8_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_uint8_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec16_uint8_t, __clc_size_t, __clc_uint8_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec16_uint8_t args_0, __clc_size_t args_1,
                    __clc_uint8_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_uint8_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_uint16_t, __clc_size_t, __clc_uint16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec2_uint16_t args_0, __clc_size_t args_1,
                    __clc_uint16_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_uint16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_uint16_t, __clc_size_t, __clc_uint16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec3_uint16_t args_0, __clc_size_t args_1,
                    __clc_uint16_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_uint16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_uint16_t, __clc_size_t, __clc_uint16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec4_uint16_t args_0, __clc_size_t args_1,
                    __clc_uint16_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_uint16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_uint16_t, __clc_size_t, __clc_uint16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec8_uint16_t args_0, __clc_size_t args_1,
                    __clc_uint16_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_uint16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec16_uint16_t, __clc_size_t, __clc_uint16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec16_uint16_t args_0, __clc_size_t args_1,
                    __clc_uint16_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_uint16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_uint32_t, __clc_size_t, __clc_uint32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec2_uint32_t args_0, __clc_size_t args_1,
                    __clc_uint32_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_uint32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_uint32_t, __clc_size_t, __clc_uint32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec3_uint32_t args_0, __clc_size_t args_1,
                    __clc_uint32_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_uint32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_uint32_t, __clc_size_t, __clc_uint32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec4_uint32_t args_0, __clc_size_t args_1,
                    __clc_uint32_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_uint32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_uint32_t, __clc_size_t, __clc_uint32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec8_uint32_t args_0, __clc_size_t args_1,
                    __clc_uint32_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_uint32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec16_uint32_t, __clc_size_t, __clc_uint32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec16_uint32_t args_0, __clc_size_t args_1,
                    __clc_uint32_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_uint32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_uint64_t, __clc_size_t, __clc_uint64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec2_uint64_t args_0, __clc_size_t args_1,
                    __clc_uint64_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_uint64_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_uint64_t, __clc_size_t, __clc_uint64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec3_uint64_t args_0, __clc_size_t args_1,
                    __clc_uint64_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_uint64_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_uint64_t, __clc_size_t, __clc_uint64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec4_uint64_t args_0, __clc_size_t args_1,
                    __clc_uint64_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_uint64_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_uint64_t, __clc_size_t, __clc_uint64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec8_uint64_t args_0, __clc_size_t args_1,
                    __clc_uint64_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_uint64_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec16_uint64_t, __clc_size_t, __clc_uint64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec16_uint64_t args_0, __clc_size_t args_1,
                    __clc_uint64_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_uint64_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_fp32_t, __clc_size_t, __clc_fp32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec2_fp32_t args_0, __clc_size_t args_1,
                    __clc_fp32_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_fp32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_fp32_t, __clc_size_t, __clc_fp32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec3_fp32_t args_0, __clc_size_t args_1,
                    __clc_fp32_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_fp32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_fp32_t, __clc_size_t, __clc_fp32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec4_fp32_t args_0, __clc_size_t args_1,
                    __clc_fp32_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_fp32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_fp32_t, __clc_size_t, __clc_fp32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec8_fp32_t args_0, __clc_size_t args_1,
                    __clc_fp32_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_fp32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec16_fp32_t, __clc_size_t, __clc_fp32_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec16_fp32_t args_0, __clc_size_t args_1,
                    __clc_fp32_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_fp32_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_fp64_t, __clc_size_t, __clc_fp64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec2_fp64_t args_0, __clc_size_t args_1,
                    __clc_fp64_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_fp64_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_fp64_t, __clc_size_t, __clc_fp64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec3_fp64_t args_0, __clc_size_t args_1,
                    __clc_fp64_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_fp64_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_fp64_t, __clc_size_t, __clc_fp64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec4_fp64_t args_0, __clc_size_t args_1,
                    __clc_fp64_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_fp64_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_fp64_t, __clc_size_t, __clc_fp64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec8_fp64_t args_0, __clc_size_t args_1,
                    __clc_fp64_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_fp64_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec16_fp64_t, __clc_size_t, __clc_fp64_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec16_fp64_t args_0, __clc_size_t args_1,
                    __clc_fp64_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_fp64_t *)(args_2));
}

#ifdef cl_khr_fp16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_fp16_t, __clc_size_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec2_fp16_t args_0, __clc_size_t args_1,
                    __clc_fp16_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_fp16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_fp16_t, __clc_size_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec3_fp16_t args_0, __clc_size_t args_1,
                    __clc_fp16_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_fp16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_fp16_t, __clc_size_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec4_fp16_t args_0, __clc_size_t args_1,
                    __clc_fp16_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_fp16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_fp16_t, __clc_size_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec8_fp16_t args_0, __clc_size_t args_1,
                    __clc_fp16_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_fp16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec16_fp16_t, __clc_size_t, __clc_fp16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec16_fp16_t args_0, __clc_size_t args_1,
                    __clc_fp16_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_fp16_t *)(args_2));
}

#endif

#ifdef __CLC_HAS_FLOAT16
SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec2_float16_t, __clc_size_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec2_float16_t args_0, __clc_size_t args_1,
                    __clc_float16_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_float16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec3_float16_t, __clc_size_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec3_float16_t args_0, __clc_size_t args_1,
                    __clc_float16_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_float16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec4_float16_t, __clc_size_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec4_float16_t args_0, __clc_size_t args_1,
                    __clc_float16_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_float16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec8_float16_t, __clc_size_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec8_float16_t args_0, __clc_size_t args_1,
                    __clc_float16_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_float16_t *)(args_2));
}

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DECL void
__spirv_ocl_vstoren(__clc_vec16_float16_t, __clc_size_t, __clc_float16_t *);

SYCL_EXTERNAL _CLC_OVERLOAD _CLC_DEF void
__spirv_ocl_vstoren(__clc_vec16_float16_t args_0, __clc_size_t args_1,
                    __clc_float16_t __private *args_2) {
  __spirv_ocl_vstoren(args_0, args_1, (__clc_float16_t *)(args_2));
}

#endif
