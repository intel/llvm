/*===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===*/

// NVVM Image Helper Intrinsics - Inline PTX Assembly Implementation

// Vector type definitions
typedef short __attribute__((ext_vector_type(2))) short2;
typedef short __attribute__((ext_vector_type(4))) short4;
typedef int __attribute__((ext_vector_type(2))) int2;
typedef int __attribute__((ext_vector_type(4))) int4;
typedef unsigned int __attribute__((ext_vector_type(2))) uint2;
typedef unsigned int __attribute__((ext_vector_type(4))) uint4;
typedef float __attribute__((ext_vector_type(4))) float4;
typedef unsigned int uint;
typedef unsigned long ulong;

// Struct definitions matching NVVM intrinsic return types
typedef struct {
  short x, y, z, w;
} __nvvm_v4i16_t;
typedef struct {
  short x, y;
} __nvvm_v2i16_t;
typedef struct {
  int x, y, z, w;
} __nvvm_v4i32_t;
typedef struct {
  int x, y;
} __nvvm_v2i32_t;
typedef struct {
  uint x, y, z, w;
} __nvvm_v4u32_t;
typedef struct {
  uint x, y;
} __nvvm_v2u32_t;
typedef struct {
  float x, y, z, w;
} __nvvm_v4f32_t;

// Struct-to-vector conversion helpers
inline short4 __nvvm_v4i16_to_vec(__nvvm_v4i16_t s) {
  return __builtin_bit_cast(short4, s);
}

inline short2 __nvvm_v2i16_to_vec(__nvvm_v2i16_t s) {
  return __builtin_bit_cast(short2, s);
}

inline int4 __nvvm_v4i32_to_vec(__nvvm_v4i32_t s) {
  return __builtin_bit_cast(int4, s);
}

inline int2 __nvvm_v2i32_to_vec(__nvvm_v2i32_t s) {
  return __builtin_bit_cast(int2, s);
}

inline uint4 __nvvm_v4u32_to_vec(__nvvm_v4u32_t s) {
  return __builtin_bit_cast(uint4, s);
}

inline uint2 __nvvm_v2u32_to_vec(__nvvm_v2u32_t s) {
  return __builtin_bit_cast(uint2, s);
}

inline float4 __nvvm_v4f32_to_vec(__nvvm_v4f32_t s) {
  return __builtin_bit_cast(float4, s);
}

// Sampled image pack/unpack helpers
ulong __clc__sampled_image_unpack_image(ulong img, uint sampl) { return img; }

uint __clc__sampled_image_unpack_sampler(ulong img, uint sampl) {
  return sampl;
}

typedef struct {
  ulong img;
  uint sampl;
} __clc_sampled_image_t;

__clc_sampled_image_t __clc__sampled_image_pack(ulong img, uint sampl) {
  __clc_sampled_image_t result;
  result.img = img;
  result.sampl = sampl;
  return result;
}

// Sampler property extraction
uint __clc__sampler_extract_normalized_coords_prop(uint sampl) {
  return sampl & 1;
}

uint __clc__sampler_extract_filter_mode_prop(uint sampl) {
  return (sampl >> 1) & 1;
}

uint __clc__sampler_extract_addressing_mode_prop(uint sampl) {
  return sampl >> 2;
}

// NVVM Surface Load Intrinsics - v4i16 (trap/clamp/zero)
short4 __clc_llvm_nvvm_suld_1d_v4i16_trap(ulong img, int x) {
  __nvvm_v4i16_t r;
  __asm__("suld.b.1d.v4.b16.trap {%0, %1, %2, %3}, [%4, {%5}];"
          : "=h"(r.x), "=h"(r.y), "=h"(r.z), "=h"(r.w)
          : "l"(img), "r"(x));
  return __nvvm_v4i16_to_vec(r);
}
short4 __clc_llvm_nvvm_suld_2d_v4i16_trap(ulong img, int x, int y) {
  __nvvm_v4i16_t r;
  __asm__("suld.b.2d.v4.b16.trap {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=h"(r.x), "=h"(r.y), "=h"(r.z), "=h"(r.w)
          : "l"(img), "r"(x), "r"(y));
  return __nvvm_v4i16_to_vec(r);
}
short4 __clc_llvm_nvvm_suld_3d_v4i16_trap(ulong img, int x, int y, int z) {
  __nvvm_v4i16_t r;
  __asm__("suld.b.3d.v4.b16.trap {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
          : "=h"(r.x), "=h"(r.y), "=h"(r.z), "=h"(r.w)
          : "l"(img), "r"(x), "r"(y), "r"(z));
  return __nvvm_v4i16_to_vec(r);
}
short4 __clc_llvm_nvvm_suld_1d_v4i16_clamp(ulong img, int x) {
  __nvvm_v4i16_t r;
  __asm__("suld.b.1d.v4.b16.clamp {%0, %1, %2, %3}, [%4, {%5}];"
          : "=h"(r.x), "=h"(r.y), "=h"(r.z), "=h"(r.w)
          : "l"(img), "r"(x));
  return __nvvm_v4i16_to_vec(r);
}
short4 __clc_llvm_nvvm_suld_2d_v4i16_clamp(ulong img, int x, int y) {
  __nvvm_v4i16_t r;
  __asm__("suld.b.2d.v4.b16.clamp {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=h"(r.x), "=h"(r.y), "=h"(r.z), "=h"(r.w)
          : "l"(img), "r"(x), "r"(y));
  return __nvvm_v4i16_to_vec(r);
}
short4 __clc_llvm_nvvm_suld_3d_v4i16_clamp(ulong img, int x, int y, int z) {
  __nvvm_v4i16_t r;
  __asm__("suld.b.3d.v4.b16.clamp {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
          : "=h"(r.x), "=h"(r.y), "=h"(r.z), "=h"(r.w)
          : "l"(img), "r"(x), "r"(y), "r"(z));
  return __nvvm_v4i16_to_vec(r);
}
short4 __clc_llvm_nvvm_suld_1d_v4i16_zero(ulong img, int x) {
  __nvvm_v4i16_t r;
  __asm__("suld.b.1d.v4.b16.zero {%0, %1, %2, %3}, [%4, {%5}];"
          : "=h"(r.x), "=h"(r.y), "=h"(r.z), "=h"(r.w)
          : "l"(img), "r"(x));
  return __nvvm_v4i16_to_vec(r);
}
short4 __clc_llvm_nvvm_suld_2d_v4i16_zero(ulong img, int x, int y) {
  __nvvm_v4i16_t r;
  __asm__("suld.b.2d.v4.b16.zero {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=h"(r.x), "=h"(r.y), "=h"(r.z), "=h"(r.w)
          : "l"(img), "r"(x), "r"(y));
  return __nvvm_v4i16_to_vec(r);
}
short4 __clc_llvm_nvvm_suld_3d_v4i16_zero(ulong img, int x, int y, int z) {
  __nvvm_v4i16_t r;
  __asm__("suld.b.3d.v4.b16.zero {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
          : "=h"(r.x), "=h"(r.y), "=h"(r.z), "=h"(r.w)
          : "l"(img), "r"(x), "r"(y), "r"(z));
  return __nvvm_v4i16_to_vec(r);
}

// NVVM Surface Load Intrinsics - v4i32 (trap/clamp/zero)
int4 __clc_llvm_nvvm_suld_1d_v4i32_trap(ulong img, int x) {
  __nvvm_v4i32_t r;
  __asm__("suld.b.1d.v4.b32.trap {%0, %1, %2, %3}, [%4, {%5}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "r"(x));
  return __nvvm_v4i32_to_vec(r);
}
int4 __clc_llvm_nvvm_suld_2d_v4i32_trap(ulong img, int x, int y) {
  __nvvm_v4i32_t r;
  __asm__("suld.b.2d.v4.b32.trap {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "r"(x), "r"(y));
  return __nvvm_v4i32_to_vec(r);
}
int4 __clc_llvm_nvvm_suld_3d_v4i32_trap(ulong img, int x, int y, int z) {
  __nvvm_v4i32_t r;
  __asm__("suld.b.3d.v4.b32.trap {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "r"(x), "r"(y), "r"(z));
  return __nvvm_v4i32_to_vec(r);
}
int4 __clc_llvm_nvvm_suld_1d_v4i32_clamp(ulong img, int x) {
  __nvvm_v4i32_t r;
  __asm__("suld.b.1d.v4.b32.clamp {%0, %1, %2, %3}, [%4, {%5}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "r"(x));
  return __nvvm_v4i32_to_vec(r);
}
int4 __clc_llvm_nvvm_suld_2d_v4i32_clamp(ulong img, int x, int y) {
  __nvvm_v4i32_t r;
  __asm__("suld.b.2d.v4.b32.clamp {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "r"(x), "r"(y));
  return __nvvm_v4i32_to_vec(r);
}
int4 __clc_llvm_nvvm_suld_3d_v4i32_clamp(ulong img, int x, int y, int z) {
  __nvvm_v4i32_t r;
  __asm__("suld.b.3d.v4.b32.clamp {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "r"(x), "r"(y), "r"(z));
  return __nvvm_v4i32_to_vec(r);
}
int4 __clc_llvm_nvvm_suld_1d_v4i32_zero(ulong img, int x) {
  __nvvm_v4i32_t r;
  __asm__("suld.b.1d.v4.b32.zero {%0, %1, %2, %3}, [%4, {%5}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "r"(x));
  return __nvvm_v4i32_to_vec(r);
}
int4 __clc_llvm_nvvm_suld_2d_v4i32_zero(ulong img, int x, int y) {
  __nvvm_v4i32_t r;
  __asm__("suld.b.2d.v4.b32.zero {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "r"(x), "r"(y));
  return __nvvm_v4i32_to_vec(r);
}
int4 __clc_llvm_nvvm_suld_3d_v4i32_zero(ulong img, int x, int y, int z) {
  __nvvm_v4i32_t r;
  __asm__("suld.b.3d.v4.b32.zero {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "r"(x), "r"(y), "r"(z));
  return __nvvm_v4i32_to_vec(r);
}

// BINDLESS IMAGES - SURFACES v2i8 and v4i8
short2 __clc_llvm_nvvm_suld_1d_v2i8_clamp(ulong img, int x) {
  __nvvm_v2i16_t r;
  __asm__("suld.b.1d.v2.b8.clamp {%0, %1}, [%2, {%3}];"
          : "=h"(r.x), "=h"(r.y)
          : "l"(img), "r"(x));
  return __nvvm_v2i16_to_vec(r);
}
short2 __clc_llvm_nvvm_suld_2d_v2i8_clamp(ulong img, int x, int y) {
  __nvvm_v2i16_t r;
  __asm__("suld.b.2d.v2.b8.clamp {%0, %1}, [%2, {%3, %4}];"
          : "=h"(r.x), "=h"(r.y)
          : "l"(img), "r"(x), "r"(y));
  return __nvvm_v2i16_to_vec(r);
}
short2 __clc_llvm_nvvm_suld_3d_v2i8_clamp(ulong img, int x, int y, int z) {
  __nvvm_v2i16_t r;
  __asm__("suld.b.3d.v2.b8.clamp {%0, %1}, [%2, {%3, %4, %5, %5}];"
          : "=h"(r.x), "=h"(r.y)
          : "l"(img), "r"(x), "r"(y), "r"(z));
  return __nvvm_v2i16_to_vec(r);
}
short4 __clc_llvm_nvvm_suld_1d_v4i8_clamp(ulong img, int x) {
  __nvvm_v4i16_t r;
  __asm__("suld.b.1d.v4.b8.clamp {%0, %1, %2, %3}, [%4, {%5}];"
          : "=h"(r.x), "=h"(r.y), "=h"(r.z), "=h"(r.w)
          : "l"(img), "r"(x));
  return __nvvm_v4i16_to_vec(r);
}
short4 __clc_llvm_nvvm_suld_2d_v4i8_clamp(ulong img, int x, int y) {
  __nvvm_v4i16_t r;
  __asm__("suld.b.2d.v4.b8.clamp {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=h"(r.x), "=h"(r.y), "=h"(r.z), "=h"(r.w)
          : "l"(img), "r"(x), "r"(y));
  return __nvvm_v4i16_to_vec(r);
}
short4 __clc_llvm_nvvm_suld_3d_v4i8_clamp(ulong img, int x, int y, int z) {
  __nvvm_v4i16_t r;
  __asm__("suld.b.3d.v4.b8.clamp {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
          : "=h"(r.x), "=h"(r.y), "=h"(r.z), "=h"(r.w)
          : "l"(img), "r"(x), "r"(y), "r"(z));
  return __nvvm_v4i16_to_vec(r);
}

// TEXTURE SAMPLING (floating-point coordinates)
int4 __clc_llvm_nvvm_tex_1d_v4i32_f32(ulong img, float x) {
  __nvvm_v4i32_t r;
  __asm__("tex.1d.v4.s32.f32 {%0, %1, %2, %3}, [%4, {%5}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "f"(x));
  return __nvvm_v4i32_to_vec(r);
}
int4 __clc_llvm_nvvm_tex_2d_v4i32_f32(ulong img, float x, float y) {
  __nvvm_v4i32_t r;
  __asm__("tex.2d.v4.s32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "f"(x), "f"(y));
  return __nvvm_v4i32_to_vec(r);
}
int4 __clc_llvm_nvvm_tex_3d_v4i32_f32(ulong img, float x, float y, float z) {
  __nvvm_v4i32_t r;
  __asm__("tex.3d.v4.s32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "f"(x), "f"(y), "f"(z));
  return __nvvm_v4i32_to_vec(r);
}
uint4 __clc_llvm_nvvm_tex_1d_v4j32_f32(ulong img, float x) {
  __nvvm_v4u32_t r;
  __asm__("tex.1d.v4.u32.f32 {%0, %1, %2, %3}, [%4, {%5}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "f"(x));
  return __nvvm_v4u32_to_vec(r);
}
uint4 __clc_llvm_nvvm_tex_2d_v4j32_f32(ulong img, float x, float y) {
  __nvvm_v4u32_t r;
  __asm__("tex.2d.v4.u32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "f"(x), "f"(y));
  return __nvvm_v4u32_to_vec(r);
}
uint4 __clc_llvm_nvvm_tex_3d_v4j32_f32(ulong img, float x, float y, float z) {
  __nvvm_v4u32_t r;
  __asm__("tex.3d.v4.u32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "f"(x), "f"(y), "f"(z));
  return __nvvm_v4u32_to_vec(r);
}
float4 __clc_llvm_nvvm_tex_1d_v4f32_f32(ulong img, float x) {
  __nvvm_v4f32_t r;
  __asm__("tex.1d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5}];"
          : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
          : "l"(img), "f"(x));
  return __nvvm_v4f32_to_vec(r);
}
float4 __clc_llvm_nvvm_tex_2d_v4f32_f32(ulong img, float x, float y) {
  __nvvm_v4f32_t r;
  __asm__("tex.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
          : "l"(img), "f"(x), "f"(y));
  return __nvvm_v4f32_to_vec(r);
}
float4 __clc_llvm_nvvm_tex_3d_v4f32_f32(ulong img, float x, float y, float z) {
  __nvvm_v4f32_t r;
  __asm__("tex.3d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
          : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
          : "l"(img), "f"(x), "f"(y), "f"(z));
  return __nvvm_v4f32_to_vec(r);
}

// TEXTURE GATHER
float4 __clc_llvm_nvvm_tld4_r_2d_v4f32_f32(ulong img, float x, float y) {
  __nvvm_v4f32_t r;
  __asm__("tld4.r.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
          : "l"(img), "f"(x), "f"(y));
  return __nvvm_v4f32_to_vec(r);
}
float4 __clc_llvm_nvvm_tld4_g_2d_v4f32_f32(ulong img, float x, float y) {
  __nvvm_v4f32_t r;
  __asm__("tld4.g.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
          : "l"(img), "f"(x), "f"(y));
  return __nvvm_v4f32_to_vec(r);
}
float4 __clc_llvm_nvvm_tld4_b_2d_v4f32_f32(ulong img, float x, float y) {
  __nvvm_v4f32_t r;
  __asm__("tld4.b.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
          : "l"(img), "f"(x), "f"(y));
  return __nvvm_v4f32_to_vec(r);
}
float4 __clc_llvm_nvvm_tld4_a_2d_v4f32_f32(ulong img, float x, float y) {
  __nvvm_v4f32_t r;
  __asm__("tld4.a.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
          : "l"(img), "f"(x), "f"(y));
  return __nvvm_v4f32_to_vec(r);
}
int4 __clc_llvm_nvvm_tld4_r_2d_v4s32_f32(ulong img, float x, float y) {
  __nvvm_v4i32_t r;
  __asm__("tld4.r.2d.v4.s32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "f"(x), "f"(y));
  return __nvvm_v4i32_to_vec(r);
}
int4 __clc_llvm_nvvm_tld4_g_2d_v4s32_f32(ulong img, float x, float y) {
  __nvvm_v4i32_t r;
  __asm__("tld4.g.2d.v4.s32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "f"(x), "f"(y));
  return __nvvm_v4i32_to_vec(r);
}
int4 __clc_llvm_nvvm_tld4_b_2d_v4s32_f32(ulong img, float x, float y) {
  __nvvm_v4i32_t r;
  __asm__("tld4.b.2d.v4.s32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "f"(x), "f"(y));
  return __nvvm_v4i32_to_vec(r);
}
int4 __clc_llvm_nvvm_tld4_a_2d_v4s32_f32(ulong img, float x, float y) {
  __nvvm_v4i32_t r;
  __asm__("tld4.a.2d.v4.s32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "f"(x), "f"(y));
  return __nvvm_v4i32_to_vec(r);
}
int4 __clc_llvm_nvvm_tld4_r_2d_v4u32_f32(ulong img, float x, float y) {
  __nvvm_v4i32_t r;
  __asm__("tld4.r.2d.v4.u32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "f"(x), "f"(y));
  return __nvvm_v4i32_to_vec(r);
}
int4 __clc_llvm_nvvm_tld4_g_2d_v4u32_f32(ulong img, float x, float y) {
  __nvvm_v4i32_t r;
  __asm__("tld4.g.2d.v4.u32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "f"(x), "f"(y));
  return __nvvm_v4i32_to_vec(r);
}
int4 __clc_llvm_nvvm_tld4_b_2d_v4u32_f32(ulong img, float x, float y) {
  __nvvm_v4i32_t r;
  __asm__("tld4.b.2d.v4.u32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "f"(x), "f"(y));
  return __nvvm_v4i32_to_vec(r);
}
int4 __clc_llvm_nvvm_tld4_a_2d_v4u32_f32(ulong img, float x, float y) {
  __nvvm_v4i32_t r;
  __asm__("tld4.a.2d.v4.u32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "f"(x), "f"(y));
  return __nvvm_v4i32_to_vec(r);
}

// TEXTURE FETCHING (integer coordinates)
int4 __clc_llvm_nvvm_tex_1d_v4i32_s32(ulong img, int x) {
  __nvvm_v4i32_t r;
  __asm__("tex.1d.v4.s32.s32 {%0, %1, %2, %3}, [%4, {%5}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "r"(x));
  return __nvvm_v4i32_to_vec(r);
}
int4 __clc_llvm_nvvm_tex_2d_v4i32_s32(ulong img, int x, int y) {
  __nvvm_v4i32_t r;
  __asm__("tex.2d.v4.s32.s32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "r"(x), "r"(y));
  return __nvvm_v4i32_to_vec(r);
}
int4 __clc_llvm_nvvm_tex_3d_v4i32_s32(ulong img, int x, int y, int z) {
  __nvvm_v4i32_t r;
  __asm__("tex.3d.v4.s32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "r"(x), "r"(y), "r"(z));
  return __nvvm_v4i32_to_vec(r);
}
uint4 __clc_llvm_nvvm_tex_1d_v4j32_s32(ulong img, int x) {
  __nvvm_v4u32_t r;
  __asm__("tex.1d.v4.u32.s32 {%0, %1, %2, %3}, [%4, {%5}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "r"(x));
  return __nvvm_v4u32_to_vec(r);
}
uint4 __clc_llvm_nvvm_tex_2d_v4j32_s32(ulong img, int x, int y) {
  __nvvm_v4u32_t r;
  __asm__("tex.2d.v4.s32.s32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "r"(x), "r"(y));
  return __nvvm_v4u32_to_vec(r);
}
uint4 __clc_llvm_nvvm_tex_3d_v4j32_s32(ulong img, int x, int y, int z) {
  __nvvm_v4u32_t r;
  __asm__("tex.3d.v4.u32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "r"(x), "r"(y), "r"(z));
  return __nvvm_v4u32_to_vec(r);
}
float4 __clc_llvm_nvvm_tex_1d_v4f32_s32(ulong img, int x) {
  __nvvm_v4f32_t r;
  __asm__("tex.1d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5}];"
          : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
          : "l"(img), "r"(x));
  return __nvvm_v4f32_to_vec(r);
}
float4 __clc_llvm_nvvm_tex_2d_v4f32_s32(ulong img, int x, int y) {
  __nvvm_v4f32_t r;
  __asm__("tex.2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
          : "l"(img), "r"(x), "r"(y));
  return __nvvm_v4f32_to_vec(r);
}
float4 __clc_llvm_nvvm_tex_3d_v4f32_s32(ulong img, int x, int y, int z) {
  __nvvm_v4f32_t r;
  __asm__("tex.3d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
          : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
          : "l"(img), "r"(x), "r"(y), "r"(z));
  return __nvvm_v4f32_to_vec(r);
}

// MIPMAP - Level
float4 __clc_llvm_nvvm_tex_1d_level_v4f32_f32(ulong img, float x, float lvl) {
  __nvvm_v4f32_t r;
  __asm__("tex.level.1d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5}], %6;"
          : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
          : "l"(img), "f"(x), "f"(lvl));
  return __nvvm_v4f32_to_vec(r);
}
float4 __clc_llvm_nvvm_tex_2d_level_v4f32_f32(ulong img, float x, float y,
                                              float lvl) {
  __nvvm_v4f32_t r;
  __asm__("tex.level.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}], %7;"
          : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
          : "l"(img), "f"(x), "f"(y), "f"(lvl));
  return __nvvm_v4f32_to_vec(r);
}
float4 __clc_llvm_nvvm_tex_3d_level_v4f32_f32(ulong img, float x, float y,
                                              float z, float lvl) {
  __nvvm_v4f32_t r;
  __asm__(
      "tex.level.3d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}], %8;"
      : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
      : "l"(img), "f"(x), "f"(y), "f"(z), "f"(lvl));
  return __nvvm_v4f32_to_vec(r);
}
int4 __clc_llvm_nvvm_tex_1d_level_v4i32_f32(ulong img, float x, float lvl) {
  __nvvm_v4i32_t r;
  __asm__("tex.level.1d.v4.s32.f32 {%0, %1, %2, %3}, [%4, {%5}], %6;"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "f"(x), "f"(lvl));
  return __nvvm_v4i32_to_vec(r);
}
int4 __clc_llvm_nvvm_tex_2d_level_v4i32_f32(ulong img, float x, float y,
                                            float lvl) {
  __nvvm_v4i32_t r;
  __asm__("tex.level.2d.v4.s32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}], %7;"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "f"(x), "f"(y), "f"(lvl));
  return __nvvm_v4i32_to_vec(r);
}
int4 __clc_llvm_nvvm_tex_3d_level_v4i32_f32(ulong img, float x, float y,
                                            float z, float lvl) {
  __nvvm_v4i32_t r;
  __asm__(
      "tex.level.3d.v4.s32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}], %8;"
      : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
      : "l"(img), "f"(x), "f"(y), "f"(z), "f"(lvl));
  return __nvvm_v4i32_to_vec(r);
}
uint4 __clc_llvm_nvvm_tex_1d_level_v4j32_f32(ulong img, float x, float lvl) {
  __nvvm_v4u32_t r;
  __asm__("tex.level.1d.v4.u32.f32 {%0, %1, %2, %3}, [%4, {%5}], %6;"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "f"(x), "f"(lvl));
  return __nvvm_v4u32_to_vec(r);
}
uint4 __clc_llvm_nvvm_tex_2d_level_v4j32_f32(ulong img, float x, float y,
                                             float lvl) {
  __nvvm_v4u32_t r;
  __asm__("tex.level.2d.v4.u32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}], %7;"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "f"(x), "f"(y), "f"(lvl));
  return __nvvm_v4u32_to_vec(r);
}
uint4 __clc_llvm_nvvm_tex_3d_level_v4j32_f32(ulong img, float x, float y,
                                             float z, float lvl) {
  __nvvm_v4u32_t r;
  __asm__(
      "tex.level.3d.v4.u32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}], %8;"
      : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
      : "l"(img), "f"(x), "f"(y), "f"(z), "f"(lvl));
  return __nvvm_v4u32_to_vec(r);
}

// MIPMAP - Grad
float4 __clc_llvm_nvvm_tex_1d_grad_v4f32_f32(ulong img, float x, float dX,
                                             float dY) {
  __nvvm_v4f32_t r;
  __asm__("tex.grad.1d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5}], {%6}, {%7};"
          : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
          : "l"(img), "f"(x), "f"(dX), "f"(dY));
  return __nvvm_v4f32_to_vec(r);
}
float4 __clc_llvm_nvvm_tex_2d_grad_v4f32_f32(ulong img, float x, float y,
                                             float dXx, float dXy, float dYx,
                                             float dYy) {
  __nvvm_v4f32_t r;
  __asm__("tex.grad.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}], {%7, %8}, "
          "{%9, %10};"
          : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
          : "l"(img), "f"(x), "f"(y), "f"(dXx), "f"(dXy), "f"(dYx), "f"(dYy));
  return __nvvm_v4f32_to_vec(r);
}
float4 __clc_llvm_nvvm_tex_3d_grad_v4f32_f32(ulong img, float x, float y,
                                             float z, float dXx, float dXy,
                                             float dXz, float dYx, float dYy,
                                             float dYz) {
  __nvvm_v4f32_t r;
  __asm__("tex.grad.3d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}], "
          "{%8, %9, %10, %10}, {%11, %12, %13, %13};"
          : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
          : "l"(img), "f"(x), "f"(y), "f"(z), "f"(dXx), "f"(dXy), "f"(dXz),
            "f"(dYx), "f"(dYy), "f"(dYz));
  return __nvvm_v4f32_to_vec(r);
}
int4 __clc_llvm_nvvm_tex_1d_grad_v4i32_f32(ulong img, float x, float dX,
                                           float dY) {
  __nvvm_v4i32_t r;
  __asm__("tex.grad.1d.v4.s32.f32 {%0, %1, %2, %3}, [%4, {%5}], {%6}, {%7};"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "f"(x), "f"(dX), "f"(dY));
  return __nvvm_v4i32_to_vec(r);
}
int4 __clc_llvm_nvvm_tex_2d_grad_v4i32_f32(ulong img, float x, float y,
                                           float dXx, float dXy, float dYx,
                                           float dYy) {
  __nvvm_v4i32_t r;
  __asm__("tex.grad.2d.v4.s32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}], {%7, %8}, "
          "{%9, %10};"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "f"(x), "f"(y), "f"(dXx), "f"(dXy), "f"(dYx), "f"(dYy));
  return __nvvm_v4i32_to_vec(r);
}
int4 __clc_llvm_nvvm_tex_3d_grad_v4i32_f32(ulong img, float x, float y, float z,
                                           float dXx, float dXy, float dXz,
                                           float dYx, float dYy, float dYz) {
  __nvvm_v4i32_t r;
  __asm__("tex.grad.3d.v4.s32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}], "
          "{%8, %9, %10, %10}, {%11, %12, %13, %13};"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "f"(x), "f"(y), "f"(z), "f"(dXx), "f"(dXy), "f"(dXz),
            "f"(dYx), "f"(dYy), "f"(dYz));
  return __nvvm_v4i32_to_vec(r);
}
uint4 __clc_llvm_nvvm_tex_1d_grad_v4j32_f32(ulong img, float x, float dX,
                                            float dY) {
  __nvvm_v4u32_t r;
  __asm__("tex.grad.1d.v4.u32.f32 {%0, %1, %2, %3}, [%4, {%5}], {%6}, {%7};"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "f"(x), "f"(dX), "f"(dY));
  return __nvvm_v4u32_to_vec(r);
}
uint4 __clc_llvm_nvvm_tex_2d_grad_v4j32_f32(ulong img, float x, float y,
                                            float dXx, float dXy, float dYx,
                                            float dYy) {
  __nvvm_v4u32_t r;
  __asm__("tex.grad.2d.v4.u32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}], {%7, %8}, "
          "{%9, %10};"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "f"(x), "f"(y), "f"(dXx), "f"(dXy), "f"(dYx), "f"(dYy));
  return __nvvm_v4u32_to_vec(r);
}
uint4 __clc_llvm_nvvm_tex_3d_grad_v4j32_f32(ulong img, float x, float y,
                                            float z, float dXx, float dXy,
                                            float dXz, float dYx, float dYy,
                                            float dYz) {
  __nvvm_v4u32_t r;
  __asm__("tex.grad.3d.v4.u32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}], "
          "{%8, %9, %10, %10}, {%11, %12, %13, %13};"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "f"(x), "f"(y), "f"(z), "f"(dXx), "f"(dXy), "f"(dXz),
            "f"(dYx), "f"(dYy), "f"(dYz));
  return __nvvm_v4u32_to_vec(r);
}

// CUBEMAP
float4 __clc_llvm_nvvm_tex_cube_v4f32_f32(ulong img, float x, float y,
                                          float z) {
  __nvvm_v4f32_t r;
  __asm__("tex.cube.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
          : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
          : "l"(img), "f"(x), "f"(y), "f"(z));
  return __nvvm_v4f32_to_vec(r);
}
int4 __clc_llvm_nvvm_tex_cube_v4i32_f32(ulong img, float x, float y, float z) {
  __nvvm_v4i32_t r;
  __asm__("tex.cube.v4.s32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "f"(x), "f"(y), "f"(z));
  return __nvvm_v4i32_to_vec(r);
}
uint4 __clc_llvm_nvvm_tex_cube_v4j32_f32(ulong img, float x, float y, float z) {
  __nvvm_v4u32_t r;
  __asm__("tex.cube.v4.u32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "f"(x), "f"(y), "f"(z));
  return __nvvm_v4u32_to_vec(r);
}

// SURFACE ARRAY LOADS
short2 __clc_llvm_nvvm_suld_1d_array_v2i8_clamp(ulong img, int idx, int x) {
  __nvvm_v2i16_t r;
  __asm__("suld.b.a1d.v2.b8.clamp {%0, %1}, [%2, {%3, %4}];"
          : "=h"(r.x), "=h"(r.y)
          : "l"(img), "r"(idx), "r"(x));
  return __nvvm_v2i16_to_vec(r);
}
short2 __clc_llvm_nvvm_suld_2d_array_v2i8_clamp(ulong img, int idx, int x,
                                                int y) {
  __nvvm_v2i16_t r;
  __asm__("suld.b.a2d.v2.b8.clamp {%0, %1}, [%2, {%3, %4, %5, %5}];"
          : "=h"(r.x), "=h"(r.y)
          : "l"(img), "r"(idx), "r"(x), "r"(y));
  return __nvvm_v2i16_to_vec(r);
}
short4 __clc_llvm_nvvm_suld_1d_array_v4i8_clamp(ulong img, int idx, int x) {
  __nvvm_v4i16_t r;
  __asm__("suld.b.a1d.v4.b8.clamp {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=h"(r.x), "=h"(r.y), "=h"(r.z), "=h"(r.w)
          : "l"(img), "r"(idx), "r"(x));
  return __nvvm_v4i16_to_vec(r);
}
short4 __clc_llvm_nvvm_suld_2d_array_v4i8_clamp(ulong img, int idx, int x,
                                                int y) {
  __nvvm_v4i16_t r;
  __asm__("suld.b.a2d.v4.b8.clamp {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
          : "=h"(r.x), "=h"(r.y), "=h"(r.z), "=h"(r.w)
          : "l"(img), "r"(idx), "r"(x), "r"(y));
  return __nvvm_v4i16_to_vec(r);
}
short2 __clc_llvm_nvvm_suld_1d_array_v2i16_clamp(ulong img, int idx, int x) {
  __nvvm_v2i16_t r;
  __asm__("suld.b.a1d.v2.b16.clamp {%0, %1}, [%2, {%3, %4}];"
          : "=h"(r.x), "=h"(r.y)
          : "l"(img), "r"(idx), "r"(x));
  return __nvvm_v2i16_to_vec(r);
}
short2 __clc_llvm_nvvm_suld_2d_array_v2i16_clamp(ulong img, int idx, int x,
                                                 int y) {
  __nvvm_v2i16_t r;
  __asm__("suld.b.a2d.v2.b16.clamp {%0, %1}, [%2, {%3, %4, %5, %5}];"
          : "=h"(r.x), "=h"(r.y)
          : "l"(img), "r"(idx), "r"(x), "r"(y));
  return __nvvm_v2i16_to_vec(r);
}
short4 __clc_llvm_nvvm_suld_1d_array_v4i16_clamp(ulong img, int idx, int x) {
  __nvvm_v4i16_t r;
  __asm__("suld.b.a1d.v4.b16.clamp {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=h"(r.x), "=h"(r.y), "=h"(r.z), "=h"(r.w)
          : "l"(img), "r"(idx), "r"(x));
  return __nvvm_v4i16_to_vec(r);
}
short4 __clc_llvm_nvvm_suld_2d_array_v4i16_clamp(ulong img, int idx, int x,
                                                 int y) {
  __nvvm_v4i16_t r;
  __asm__("suld.b.a2d.v4.b16.clamp {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
          : "=h"(r.x), "=h"(r.y), "=h"(r.z), "=h"(r.w)
          : "l"(img), "r"(idx), "r"(x), "r"(y));
  return __nvvm_v4i16_to_vec(r);
}
int2 __clc_llvm_nvvm_suld_1d_array_v2i32_clamp(ulong img, int idx, int x) {
  __nvvm_v2i32_t r;
  __asm__("suld.b.a1d.v2.b32.clamp {%0, %1}, [%2, {%3, %4}];"
          : "=r"(r.x), "=r"(r.y)
          : "l"(img), "r"(idx), "r"(x));
  return __nvvm_v2i32_to_vec(r);
}
int2 __clc_llvm_nvvm_suld_2d_array_v2i32_clamp(ulong img, int idx, int x,
                                               int y) {
  __nvvm_v2i32_t r;
  __asm__("suld.b.a2d.v2.b32.clamp {%0, %1}, [%2, {%3, %4, %5, %5}];"
          : "=r"(r.x), "=r"(r.y)
          : "l"(img), "r"(idx), "r"(x), "r"(y));
  return __nvvm_v2i32_to_vec(r);
}
int4 __clc_llvm_nvvm_suld_1d_array_v4i32_clamp(ulong img, int idx, int x) {
  __nvvm_v4i32_t r;
  __asm__("suld.b.a1d.v4.b32.clamp {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "r"(idx), "r"(x));
  return __nvvm_v4i32_to_vec(r);
}
int4 __clc_llvm_nvvm_suld_2d_array_v4i32_clamp(ulong img, int idx, int x,
                                               int y) {
  __nvvm_v4i32_t r;
  __asm__("suld.b.a2d.v4.b32.clamp {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "r"(idx), "r"(x), "r"(y));
  return __nvvm_v4i32_to_vec(r);
}

// TEXTURE ARRAY - float coords
float4 __clc_llvm_nvvm_tex_unified_1d_array_v4f32_f32(ulong img, int idx,
                                                      float x) {
  __nvvm_v4f32_t r;
  __asm__("tex.a1d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
          : "l"(img), "r"(idx), "f"(x));
  return __nvvm_v4f32_to_vec(r);
}
float4 __clc_llvm_nvvm_tex_unified_2d_array_v4f32_f32(ulong img, int idx,
                                                      float x, float y) {
  __nvvm_v4f32_t r;
  __asm__("tex.a2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
          : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
          : "l"(img), "r"(idx), "f"(x), "f"(y));
  return __nvvm_v4f32_to_vec(r);
}
int4 __clc_llvm_nvvm_tex_unified_1d_array_v4i32_f32(ulong img, int idx,
                                                    float x) {
  __nvvm_v4i32_t r;
  __asm__("tex.a1d.v4.s32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "r"(idx), "f"(x));
  return __nvvm_v4i32_to_vec(r);
}
int4 __clc_llvm_nvvm_tex_unified_2d_array_v4i32_f32(ulong img, int idx, float x,
                                                    float y) {
  __nvvm_v4i32_t r;
  __asm__("tex.a2d.v4.s32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "r"(idx), "f"(x), "f"(y));
  return __nvvm_v4i32_to_vec(r);
}
uint4 __clc_llvm_nvvm_tex_unified_1d_array_v4j32_f32(ulong img, int idx,
                                                     float x) {
  __nvvm_v4u32_t r;
  __asm__("tex.a1d.v4.u32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "r"(idx), "f"(x));
  return __nvvm_v4u32_to_vec(r);
}
uint4 __clc_llvm_nvvm_tex_unified_2d_array_v4j32_f32(ulong img, int idx,
                                                     float x, float y) {
  __nvvm_v4u32_t r;
  __asm__("tex.a2d.v4.u32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "r"(idx), "f"(x), "f"(y));
  return __nvvm_v4u32_to_vec(r);
}

// TEXTURE ARRAY - int coords
float4 __clc_llvm_nvvm_tex_unified_1d_array_v4f32_i32(ulong img, int idx,
                                                      int x) {
  __nvvm_v4f32_t r;
  __asm__("tex.a1d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
          : "l"(img), "r"(idx), "r"(x));
  return __nvvm_v4f32_to_vec(r);
}
float4 __clc_llvm_nvvm_tex_unified_2d_array_v4f32_i32(ulong img, int idx, int x,
                                                      int y) {
  __nvvm_v4f32_t r;
  __asm__("tex.a2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
          : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
          : "l"(img), "r"(idx), "r"(x), "r"(y));
  return __nvvm_v4f32_to_vec(r);
}
int4 __clc_llvm_nvvm_tex_unified_1d_array_v4i32_i32(ulong img, int idx, int x) {
  __nvvm_v4i32_t r;
  __asm__("tex.a1d.v4.s32.s32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "r"(idx), "r"(x));
  return __nvvm_v4i32_to_vec(r);
}
int4 __clc_llvm_nvvm_tex_unified_2d_array_v4i32_i32(ulong img, int idx, int x,
                                                    int y) {
  __nvvm_v4i32_t r;
  __asm__("tex.a2d.v4.s32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "r"(idx), "r"(x), "r"(y));
  return __nvvm_v4i32_to_vec(r);
}
uint4 __clc_llvm_nvvm_tex_unified_1d_array_v4j32_i32(ulong img, int idx,
                                                     int x) {
  __nvvm_v4u32_t r;
  __asm__("tex.a1d.v4.u32.s32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "r"(idx), "r"(x));
  return __nvvm_v4u32_to_vec(r);
}
uint4 __clc_llvm_nvvm_tex_unified_2d_array_v4j32_i32(ulong img, int idx, int x,
                                                     int y) {
  __nvvm_v4u32_t r;
  __asm__("tex.a2d.v4.u32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
          : "l"(img), "r"(idx), "r"(x), "r"(y));
  return __nvvm_v4u32_to_vec(r);
}
