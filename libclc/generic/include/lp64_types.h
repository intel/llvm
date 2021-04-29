//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLC_LP64_TYPES
#define CLC_LP64_TYPES

#ifdef __FLT16_MAX__
#define __CLC_HAS_FLOAT16
#endif

typedef bool __clc_bool_t;

typedef char __clc_char_t;
typedef char __clc_vec2_char_t __attribute__((ext_vector_type(2)));
typedef char __clc_vec3_char_t __attribute__((ext_vector_type(3)));
typedef char __clc_vec4_char_t __attribute__((ext_vector_type(4)));
typedef char __clc_vec8_char_t __attribute__((ext_vector_type(8)));
typedef char __clc_vec16_char_t __attribute__((ext_vector_type(16)));

typedef signed char __clc_int8_t;
typedef signed char __clc_vec2_int8_t __attribute__((ext_vector_type(2)));
typedef signed char __clc_vec3_int8_t __attribute__((ext_vector_type(3)));
typedef signed char __clc_vec4_int8_t __attribute__((ext_vector_type(4)));
typedef signed char __clc_vec8_int8_t __attribute__((ext_vector_type(8)));
typedef signed char __clc_vec16_int8_t __attribute__((ext_vector_type(16)));

typedef unsigned char __clc_uint8_t;
typedef unsigned char __clc_vec2_uint8_t __attribute__((ext_vector_type(2)));
typedef unsigned char __clc_vec3_uint8_t __attribute__((ext_vector_type(3)));
typedef unsigned char __clc_vec4_uint8_t __attribute__((ext_vector_type(4)));
typedef unsigned char __clc_vec8_uint8_t __attribute__((ext_vector_type(8)));
typedef unsigned char __clc_vec16_uint8_t __attribute__((ext_vector_type(16)));

typedef short __clc_int16_t;
typedef short __clc_vec2_int16_t __attribute__((ext_vector_type(2)));
typedef short __clc_vec3_int16_t __attribute__((ext_vector_type(3)));
typedef short __clc_vec4_int16_t __attribute__((ext_vector_type(4)));
typedef short __clc_vec8_int16_t __attribute__((ext_vector_type(8)));
typedef short __clc_vec16_int16_t __attribute__((ext_vector_type(16)));

typedef unsigned short __clc_uint16_t;
typedef unsigned short __clc_vec2_uint16_t __attribute__((ext_vector_type(2)));
typedef unsigned short __clc_vec3_uint16_t __attribute__((ext_vector_type(3)));
typedef unsigned short __clc_vec4_uint16_t __attribute__((ext_vector_type(4)));
typedef unsigned short __clc_vec8_uint16_t __attribute__((ext_vector_type(8)));
typedef unsigned short __clc_vec16_uint16_t
    __attribute__((ext_vector_type(16)));

typedef int __clc_int32_t;
typedef int __clc_vec2_int32_t __attribute__((ext_vector_type(2)));
typedef int __clc_vec3_int32_t __attribute__((ext_vector_type(3)));
typedef int __clc_vec4_int32_t __attribute__((ext_vector_type(4)));
typedef int __clc_vec8_int32_t __attribute__((ext_vector_type(8)));
typedef int __clc_vec16_int32_t __attribute__((ext_vector_type(16)));

typedef unsigned int __clc_uint32_t;
typedef unsigned int __clc_vec2_uint32_t __attribute__((ext_vector_type(2)));
typedef unsigned int __clc_vec3_uint32_t __attribute__((ext_vector_type(3)));
typedef unsigned int __clc_vec4_uint32_t __attribute__((ext_vector_type(4)));
typedef unsigned int __clc_vec8_uint32_t __attribute__((ext_vector_type(8)));
typedef unsigned int __clc_vec16_uint32_t __attribute__((ext_vector_type(16)));

typedef long __clc_int64_t;
typedef long __clc_vec2_int64_t __attribute__((ext_vector_type(2)));
typedef long __clc_vec3_int64_t __attribute__((ext_vector_type(3)));
typedef long __clc_vec4_int64_t __attribute__((ext_vector_type(4)));
typedef long __clc_vec8_int64_t __attribute__((ext_vector_type(8)));
typedef long __clc_vec16_int64_t __attribute__((ext_vector_type(16)));

typedef unsigned long __clc_uint64_t;
typedef unsigned long __clc_vec2_uint64_t __attribute__((ext_vector_type(2)));
typedef unsigned long __clc_vec3_uint64_t __attribute__((ext_vector_type(3)));
typedef unsigned long __clc_vec4_uint64_t __attribute__((ext_vector_type(4)));
typedef unsigned long __clc_vec8_uint64_t __attribute__((ext_vector_type(8)));
typedef unsigned long __clc_vec16_uint64_t __attribute__((ext_vector_type(16)));

typedef float __clc_fp32_t;
typedef float __clc_vec2_fp32_t __attribute__((ext_vector_type(2)));
typedef float __clc_vec3_fp32_t __attribute__((ext_vector_type(3)));
typedef float __clc_vec4_fp32_t __attribute__((ext_vector_type(4)));
typedef float __clc_vec8_fp32_t __attribute__((ext_vector_type(8)));
typedef float __clc_vec16_fp32_t __attribute__((ext_vector_type(16)));

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef double __clc_fp64_t;
typedef double __clc_vec2_fp64_t __attribute__((ext_vector_type(2)));
typedef double __clc_vec3_fp64_t __attribute__((ext_vector_type(3)));
typedef double __clc_vec4_fp64_t __attribute__((ext_vector_type(4)));
typedef double __clc_vec8_fp64_t __attribute__((ext_vector_type(8)));
typedef double __clc_vec16_fp64_t __attribute__((ext_vector_type(16)));

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef half __clc_fp16_t;
typedef half __clc_vec2_fp16_t __attribute__((ext_vector_type(2)));
typedef half __clc_vec3_fp16_t __attribute__((ext_vector_type(3)));
typedef half __clc_vec4_fp16_t __attribute__((ext_vector_type(4)));
typedef half __clc_vec8_fp16_t __attribute__((ext_vector_type(8)));
typedef half __clc_vec16_fp16_t __attribute__((ext_vector_type(16)));

#endif

#ifdef __CLC_HAS_FLOAT16

typedef _Float16 __clc_float16_t;
typedef _Float16 __clc_vec2_float16_t __attribute__((ext_vector_type(2)));
typedef _Float16 __clc_vec3_float16_t __attribute__((ext_vector_type(3)));
typedef _Float16 __clc_vec4_float16_t __attribute__((ext_vector_type(4)));
typedef _Float16 __clc_vec8_float16_t __attribute__((ext_vector_type(8)));
typedef _Float16 __clc_vec16_float16_t __attribute__((ext_vector_type(16)));

#endif
typedef __clc_int64_t __clc_size_t;

typedef event_t __clc_event_t;

#endif // CLC_LP64_TYPES
