//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>
#include <spirv/spirv.h>

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#ifdef cl_khr_3d_image_writes
#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable
#endif

#ifdef _WIN32
#define MANGLE_FUNC_IMG_HANDLE(namelength, name, prefix, postfix)              \
  _Z##namelength##name##prefix##y##postfix
#else
#define MANGLE_FUNC_IMG_HANDLE(namelength, name, prefix, postfix)              \
  _Z##namelength##name##prefix##m##postfix
#endif

// CLC helpers
int __clc__sampler_extract_normalized_coords_prop(int) __asm(
    "__clc__sampler_extract_normalized_coords_prop");
int __clc__sampler_extract_filter_mode_prop(int) __asm(
    "__clc__sampler_extract_filter_mode_prop");
int __clc__sampler_extract_addressing_mode_prop(int) __asm(
    "__clc__sampler_extract_addressing_mode_prop");

__ocl_sampled_image1d_ro_t
__clc__sampled_image1d_pack(read_only image1d_t,
                            sampler_t) __asm("__clc__sampled_image_pack");
__ocl_sampled_image2d_ro_t
__clc__sampled_image2d_pack(read_only image2d_t,
                            sampler_t) __asm("__clc__sampled_image_pack");
__ocl_sampled_image3d_ro_t
__clc__sampled_image3d_pack(read_only image3d_t,
                            sampler_t) __asm("__clc__sampled_image_pack");

long __clc__sampled_image1d_unpack_image(__ocl_sampled_image1d_ro_t) __asm(
    "__clc__sampled_image_unpack_image");
long __clc__sampled_image2d_unpack_image(__ocl_sampled_image2d_ro_t) __asm(
    "__clc__sampled_image_unpack_image");
long __clc__sampled_image3d_unpack_image(__ocl_sampled_image3d_ro_t) __asm(
    "__clc__sampled_image_unpack_image");

int __clc__sampled_image1d_unpack_sampler(__ocl_sampled_image1d_ro_t) __asm(
    "__clc__sampled_image_unpack_sampler");
int __clc__sampled_image2d_unpack_sampler(__ocl_sampled_image2d_ro_t) __asm(
    "__clc__sampled_image_unpack_sampler");
int __clc__sampled_image3d_unpack_sampler(__ocl_sampled_image3d_ro_t) __asm(
    "__clc__sampled_image_unpack_sampler");

// NVVM helpers
#ifdef cl_khr_fp16
short4
__nvvm_suld_1d_v4i16_trap_s(long, int) __asm("__clc_llvm_nvvm_suld_1d_v4i16_trap");
short4
__nvvm_suld_2d_v4i16_trap_s(long, int,
                            int) __asm("__clc_llvm_nvvm_suld_2d_v4i16_trap");
short4
__nvvm_suld_3d_v4i16_trap_s(long, int, int,
                            int) __asm("__clc_llvm_nvvm_suld_3d_v4i16_trap");

short4
__nvvm_suld_1d_v4i16_clamp_s(long, int) __asm("__clc_llvm_nvvm_suld_1d_v4i16_clamp");
short4
__nvvm_suld_2d_v4i16_clamp_s(long, int,
                             int) __asm("__clc_llvm_nvvm_suld_2d_v4i16_clamp");
short4
__nvvm_suld_3d_v4i16_clamp_s(long, int, int,
                             int) __asm("__clc_llvm_nvvm_suld_3d_v4i16_clamp");

short4
__nvvm_suld_1d_v4i16_zero_s(long, int) __asm("__clc_llvm_nvvm_suld_1d_v4i16_zero");
short4
__nvvm_suld_2d_v4i16_zero_s(long, int,
                            int) __asm("__clc_llvm_nvvm_suld_2d_v4i16_zero");
short4
__nvvm_suld_3d_v4i16_zero_s(long, int, int,
                            int) __asm("__clc_llvm_nvvm_suld_3d_v4i16_zero");

short4
__nvvm_suld_1d_v4i16_clamp(read_only image1d_t,
                           int) __asm("__clc_llvm_nvvm_suld_1d_v4i16_clamp");
short4
__nvvm_suld_2d_v4i16_clamp(read_only image2d_t, int,
                           int) __asm("__clc_llvm_nvvm_suld_2d_v4i16_clamp");
short4
__nvvm_suld_3d_v4i16_clamp(read_only image3d_t, int, int,
                           int) __asm("__clc_llvm_nvvm_suld_3d_v4i16_clamp");
#endif

int4
__nvvm_suld_1d_v4i32_trap_s(long, int) __asm("__clc_llvm_nvvm_suld_1d_v4i32_trap");
int4
__nvvm_suld_2d_v4i32_trap_s(long, int,
                            int) __asm("__clc_llvm_nvvm_suld_2d_v4i32_trap");
int4
__nvvm_suld_3d_v4i32_trap_s(long, int, int,
                            int) __asm("__clc_llvm_nvvm_suld_3d_v4i32_trap");

int4
__nvvm_suld_1d_v4i32_clamp_s(long, int) __asm("__clc_llvm_nvvm_suld_1d_v4i32_clamp");
int4
__nvvm_suld_2d_v4i32_clamp_s(long, int,
                             int) __asm("__clc_llvm_nvvm_suld_2d_v4i32_clamp");
int4
__nvvm_suld_3d_v4i32_clamp_s(long, int, int,
                             int) __asm("__clc_llvm_nvvm_suld_3d_v4i32_clamp");

int4
__nvvm_suld_1d_v4i32_zero_s(long, int) __asm("__clc_llvm_nvvm_suld_1d_v4i32_zero");
int4
__nvvm_suld_2d_v4i32_zero_s(long, int,
                            int) __asm("__clc_llvm_nvvm_suld_2d_v4i32_zero");
int4
__nvvm_suld_3d_v4i32_zero_s(long, int, int,
                            int) __asm("__clc_llvm_nvvm_suld_3d_v4i32_zero");

int4
__nvvm_suld_1d_v4i32_clamp(read_only image1d_t,
                           int) __asm("__clc_llvm_nvvm_suld_1d_v4i32_clamp");
int4
__nvvm_suld_2d_v4i32_clamp(read_only image2d_t, int,
                           int) __asm("__clc_llvm_nvvm_suld_2d_v4i32_clamp");
int4
__nvvm_suld_3d_v4i32_clamp(read_only image3d_t, int, int,
                           int) __asm("__clc_llvm_nvvm_suld_3d_v4i32_clamp");

void __nvvm_sust_1d_v4i16_clamp(write_only image1d_t, int, short, short, short,
                                short) __asm("llvm.nvvm.sust.b.1d.v4i16.clamp");
void __nvvm_sust_2d_v4i16_clamp(write_only image2d_t, int, int, short, short,
                                short,
                                short) __asm("llvm.nvvm.sust.b.2d.v4i16.clamp");
void __nvvm_sust_1d_v4i32_clamp(write_only image1d_t, int, int, int, int,
                                int) __asm("llvm.nvvm.sust.b.1d.v4i32.clamp");
void __nvvm_sust_2d_v4i32_clamp(write_only image2d_t, int, int, int, int, int,
                                int) __asm("llvm.nvvm.sust.b.2d.v4i32.clamp");

#ifdef cl_khr_3d_image_writes
void __nvvm_sust_3d_v4i16_clamp(write_only image3d_t, int, int, int, short,
                                short, short,
                                short) __asm("llvm.nvvm.sust.b.3d.v4i16.clamp");
void __nvvm_sust_3d_v4i32_clamp(write_only image3d_t, int, int, int, int, int,
                                int,
                                int) __asm("llvm.nvvm.sust.b.3d.v4i32.clamp");
#endif

int __nvvm_suq_width(long) __asm("llvm.nvvm.suq.width");
int __nvvm_suq_height(long) __asm("llvm.nvvm.suq.height");
int __nvvm_suq_depth(long arg) {
  // suq.depth generates runtime errors in CUDA
  return -1;
}

int __nvvm_suq_width_1i(read_only image1d_t) __asm("llvm.nvvm.suq.width");
int __nvvm_suq_width_2i(read_only image2d_t) __asm("llvm.nvvm.suq.width");
int __nvvm_suq_width_3i(read_only image3d_t) __asm("llvm.nvvm.suq.width");
int __nvvm_suq_height_2i(read_only image2d_t) __asm("llvm.nvvm.suq.height");
int __nvvm_suq_height_3i(read_only image3d_t) __asm("llvm.nvvm.suq.height");
int __nvvm_suq_depth_3i(read_only image3d_t arg) {
  return -1;
}

// Helpers

inline int is_normalized_coords(int sampler) {
  return __clc__sampler_extract_normalized_coords_prop(sampler) == 1;
}

inline int is_nearest_filter_mode(int sampler) {
  return __clc__sampler_extract_filter_mode_prop(sampler) == 0;
}

inline int is_address_mode(int sampler, int expected) {
  return __clc__sampler_extract_addressing_mode_prop(sampler) == expected;
}

float get_common_linear_fract_and_coords_fp32(float coord, int *x0, int *x1) {
  float ia;
  float a = __spirv_ocl_fract(coord - 0.5f, &ia);
  *x0 = (int)ia;
  *x1 = *x0 + 1;
  return a;
}

#ifdef cl_khr_fp16
half get_common_linear_fract_and_coords_fp16(float coord, int *x0, int *x1) {
  half ia;
  half a = __spirv_ocl_fract(coord - 0.5f, &ia);
  *x0 = (int)ia;
  *x1 = *x0 + 1;
  return a;
}
#endif

typedef half4 pixelf16;
typedef float4 pixelf32;
typedef half fp16;
typedef float fp32;

pixelf16 as_pixelf16(short4 v) { return as_half4(v); }
pixelf32 as_pixelf32(int4 v) { return as_float4(v); }

#define _DEFINE_VEC4_CAST(from_t, to_t)                                        \
  inline to_t##4 cast_##from_t##4_to_##to_t##4(from_t##4 from) {               \
    return (to_t##4)((to_t)from.x, (to_t)from.y, (to_t)from.z, (to_t)from.w);  \
  }

#define _DEFINE_VEC2_CAST(from_t, to_t)                                        \
  inline to_t##2 cast_##from_t##2_to_##to_t##2(from_t##2 from) {               \
    return (to_t##2)((to_t)from.x, (to_t)from.y);                              \
  }

#define _DEFINE_VEC4_TO_VEC2_CAST(from_t, to_t)                                \
  inline to_t##2 cast_##from_t##4_to_##to_t##2(from_t##4 from) {               \
    return (to_t##2)((to_t)from.x, (to_t)from.y);                              \
  }

#define _DEFINE_CAST(from_t, to_t)                                             \
  inline to_t cast_##from_t##_to_##to_t(from_t from) { return (to_t)from; }

#define _DEFINE_PIXELF_CAST(pixelf_size, pixelf_base_t, to_t)                  \
  inline to_t cast_pixelf##pixelf_size##_to_##to_t(pixelf##pixelf_size from) { \
    return cast_##pixelf_base_t##_to_##to_t(from);                             \
  }

#define _DEFINE_READ_1D_PIXELF(pixelf_size, cuda_address_mode)                 \
  pixelf##pixelf_size read_1d_##pixelf_size##_##cuda_address_mode(long image,  \
                                                                  int x) {     \
    return as_pixelf##pixelf_size(                                             \
        __nvvm_suld_1d_v4i##pixelf_size##_##cuda_address_mode##_s(             \
            image, x * sizeof(pixelf##pixelf_size)));                          \
  }

#define _DEFINE_READ_2D_PIXELF(pixelf_size, cuda_address_mode)                 \
  pixelf##pixelf_size read_2d_##pixelf_size##_##cuda_address_mode(             \
      long image, int x, int y) {                                              \
    return as_pixelf##pixelf_size(                                             \
        __nvvm_suld_2d_v4i##pixelf_size##_##cuda_address_mode##_s(             \
            image, x * sizeof(pixelf##pixelf_size), y));                       \
  }

#define _DEFINE_READ_3D_PIXELF(pixelf_size, cuda_address_mode)                 \
  pixelf##pixelf_size read_3d_##pixelf_size##_##cuda_address_mode(             \
      long image, int x, int y, int z) {                                       \
    return as_pixelf##pixelf_size(                                             \
        __nvvm_suld_3d_v4i##pixelf_size##_##cuda_address_mode##_s(             \
            image, x * sizeof(pixelf##pixelf_size), y, z));                    \
  }

_DEFINE_VEC4_CAST(float, int)
_DEFINE_VEC4_CAST(int, float)
_DEFINE_VEC4_CAST(float, uint)
_DEFINE_VEC4_CAST(uint, float)
_DEFINE_VEC4_CAST(uint, int)
_DEFINE_VEC4_CAST(int, uint)
_DEFINE_VEC4_CAST(int, short)
_DEFINE_VEC4_CAST(int, char)
_DEFINE_VEC4_CAST(uint, ushort)
_DEFINE_VEC4_CAST(uint, uchar)
_DEFINE_VEC4_CAST(short, char)
_DEFINE_VEC4_CAST(short, uchar)
_DEFINE_VEC4_CAST(float, half)

_DEFINE_VEC4_TO_VEC2_CAST(int, int)
_DEFINE_VEC4_TO_VEC2_CAST(uint, uint)
_DEFINE_VEC4_TO_VEC2_CAST(float, float)
_DEFINE_VEC4_TO_VEC2_CAST(short, short)
_DEFINE_VEC4_TO_VEC2_CAST(short, char)
_DEFINE_VEC4_TO_VEC2_CAST(int, short)
_DEFINE_VEC4_TO_VEC2_CAST(int, char)
_DEFINE_VEC4_TO_VEC2_CAST(uint, ushort)
_DEFINE_VEC4_TO_VEC2_CAST(uint, uchar)
_DEFINE_VEC4_TO_VEC2_CAST(float, half)

_DEFINE_VEC2_CAST(int, float)
_DEFINE_VEC2_CAST(short, char)
_DEFINE_VEC2_CAST(short, uchar)

_DEFINE_CAST(int, float)
_DEFINE_CAST(float, float)
_DEFINE_CAST(float2, float2)
_DEFINE_CAST(float4, float4)
_DEFINE_CAST(pixelf32, float4)
_DEFINE_CAST(pixelf32, pixelf32)
_DEFINE_CAST(float4, pixelf32)

_DEFINE_PIXELF_CAST(32, float4, int4)
_DEFINE_PIXELF_CAST(32, float4, uint4)

_DEFINE_READ_1D_PIXELF(32, trap)
_DEFINE_READ_2D_PIXELF(32, trap)
_DEFINE_READ_3D_PIXELF(32, trap)
_DEFINE_READ_1D_PIXELF(32, zero)
_DEFINE_READ_2D_PIXELF(32, zero)
_DEFINE_READ_3D_PIXELF(32, zero)
_DEFINE_READ_1D_PIXELF(32, clamp)
_DEFINE_READ_2D_PIXELF(32, clamp)
_DEFINE_READ_3D_PIXELF(32, clamp)

#ifdef cl_khr_fp16
_DEFINE_CAST(half, half)
_DEFINE_CAST(half2, half2)
_DEFINE_CAST(half4, half4)
_DEFINE_CAST(pixelf16, half4)
_DEFINE_CAST(pixelf16, pixelf16)
_DEFINE_CAST(half4, pixelf16)
_DEFINE_READ_1D_PIXELF(16, trap)
_DEFINE_READ_2D_PIXELF(16, trap)
_DEFINE_READ_3D_PIXELF(16, trap)
_DEFINE_READ_1D_PIXELF(16, zero)
_DEFINE_READ_2D_PIXELF(16, zero)
_DEFINE_READ_3D_PIXELF(16, zero)
_DEFINE_READ_1D_PIXELF(16, clamp)
_DEFINE_READ_2D_PIXELF(16, clamp)
_DEFINE_READ_3D_PIXELF(16, clamp)
#endif

#undef _DEFINE_VEC4_CAST
#undef _DEFINE_VEC2_CAST
#undef _DEFINE_CAST
#undef _DEFINE_READ_1D_PIXELF
#undef _DEFINE_READ_2D_PIXELF
#undef _DEFINE_READ_3D_PIXELF

// Builtins

// Unsampled images
#define _CLC_DEFINE_IMAGE1D_READ_BUILTIN(elem_t, elem_t_mangled, elem_size)         \
  _CLC_DEF                                                                          \
  elem_t##4 _Z17__spirv_ImageReadIDv4_##elem_t_mangled##14ocl_image1d_roiET_T0_T1_( \
      read_only image1d_t image, int x) {                                           \
    return as_##elem_t##4(                                                          \
        __nvvm_suld_1d_v4i##elem_size##_clamp(image, x * sizeof(elem_t##4)));       \
  }

#define _CLC_DEFINE_IMAGE2D_READ_BUILTIN(elem_t, elem_t_mangled, elem_size)             \
  _CLC_DEF                                                                              \
  elem_t##4 _Z17__spirv_ImageReadIDv4_##elem_t_mangled##14ocl_image2d_roDv2_iET_T0_T1_( \
      read_only image2d_t image, int2 coord) {                                          \
    return as_##elem_t##4(__nvvm_suld_2d_v4i##elem_size##_clamp(                        \
        image, coord.x * sizeof(elem_t##4), coord.y));                                  \
  }

#define _CLC_DEFINE_IMAGE3D_READ_BUILTIN(elem_t, elem_t_mangled, elem_size,                         \
                                         coord_mangled)                                             \
  _CLC_DEF                                                                                          \
  elem_t##4 _Z17__spirv_ImageReadIDv4_##elem_t_mangled##14ocl_image3d_ro##coord_mangled##ET_T0_T1_( \
      read_only image3d_t image, int4 coord) {                                                      \
    return as_##elem_t##4(__nvvm_suld_3d_v4i##elem_size##_clamp(                                    \
        image, coord.x * sizeof(elem_t##4), coord.y, coord.z));                                     \
  }

#define _CLC_DEFINE_IMAGE1D_WRITE_BUILTIN(elem_t, elem_t_mangled, elem_size,    \
                                          int_rep)                              \
  _CLC_DEF void                                                                 \
      _Z18__spirv_ImageWriteI14ocl_image1d_woiDv4_##elem_t_mangled##EvT_T0_T1_( \
          write_only image1d_t image, int x, elem_t##4 c) {                     \
    __nvvm_sust_1d_v4i##elem_size##_clamp(                                      \
        image, x * sizeof(elem_t##4), as_##int_rep(c.x), as_##int_rep(c.y),     \
        as_##int_rep(c.z), as_##int_rep(c.w));                                  \
  }

#define _CLC_DEFINE_IMAGE2D_WRITE_BUILTIN(elem_t, elem_t_mangled, elem_size,        \
                                          int_rep)                                  \
  _CLC_DEF void                                                                     \
      _Z18__spirv_ImageWriteI14ocl_image2d_woDv2_iDv4_##elem_t_mangled##EvT_T0_T1_( \
          write_only image2d_t image, int2 coord, elem_t##4 c) {                    \
    __nvvm_sust_2d_v4i##elem_size##_clamp(                                          \
        image, coord.x * sizeof(elem_t##4), coord.y, as_##int_rep(c.x),             \
        as_##int_rep(c.y), as_##int_rep(c.z), as_##int_rep(c.w));                   \
  }

#define _CLC_DEFINE_IMAGE3D_WRITE_BUILTIN(elem_t, elem_t_mangled, elem_size,   \
                                          int_rep, val_mangled)                \
  _CLC_DEF void                                                                \
      _Z18__spirv_ImageWriteI14ocl_image3d_woDv4_i##val_mangled##EvT_T0_T1_(   \
          write_only image3d_t image, int4 coord, elem_t##4 c) {               \
    __nvvm_sust_3d_v4i##elem_size##_clamp(                                     \
        image, coord.x * sizeof(elem_t##4), coord.y, coord.z,                  \
        as_##int_rep(c.x), as_##int_rep(c.y), as_##int_rep(c.z),               \
        as_##int_rep(c.w));                                                    \
  }

_CLC_DEFINE_IMAGE1D_READ_BUILTIN(float, f, 32)
_CLC_DEFINE_IMAGE1D_READ_BUILTIN(int, i, 32)
_CLC_DEFINE_IMAGE1D_READ_BUILTIN(uint, j, 32)

_CLC_DEFINE_IMAGE2D_READ_BUILTIN(float, f, 32)
_CLC_DEFINE_IMAGE2D_READ_BUILTIN(int, i, 32)
_CLC_DEFINE_IMAGE2D_READ_BUILTIN(uint, j, 32)

_CLC_DEFINE_IMAGE3D_READ_BUILTIN(float, f, 32, Dv4_i)
_CLC_DEFINE_IMAGE3D_READ_BUILTIN(int, i, 32, S0_)
_CLC_DEFINE_IMAGE3D_READ_BUILTIN(uint, j, 32, Dv4_i)

_CLC_DEFINE_IMAGE1D_WRITE_BUILTIN(float, f, 32, int)
_CLC_DEFINE_IMAGE1D_WRITE_BUILTIN(int, i, 32, int)
_CLC_DEFINE_IMAGE1D_WRITE_BUILTIN(uint, j, 32, int)

_CLC_DEFINE_IMAGE2D_WRITE_BUILTIN(float, f, 32, int)
_CLC_DEFINE_IMAGE2D_WRITE_BUILTIN(int, i, 32, int)
_CLC_DEFINE_IMAGE2D_WRITE_BUILTIN(uint, j, 32, int)

#ifdef cl_khr_3d_image_writes
_CLC_DEFINE_IMAGE3D_WRITE_BUILTIN(float, f, 32, int, Dv4_f)
_CLC_DEFINE_IMAGE3D_WRITE_BUILTIN(int, i, 32, int, S1_)
_CLC_DEFINE_IMAGE3D_WRITE_BUILTIN(uint, j, 32, int, Dv4_j)
#endif

#ifdef cl_khr_fp16
_CLC_DEFINE_IMAGE1D_READ_BUILTIN(half, DF16_, 16)
_CLC_DEFINE_IMAGE2D_READ_BUILTIN(half, DF16_, 16)
_CLC_DEFINE_IMAGE3D_READ_BUILTIN(half, DF16_, 16, Dv4_i)
_CLC_DEFINE_IMAGE1D_WRITE_BUILTIN(half, DF16_, 16, short)
_CLC_DEFINE_IMAGE2D_WRITE_BUILTIN(half, DF16_, 16, short)
#endif

#if defined(cl_khr_3d_image_writes) && defined(cl_khr_fp16)
_CLC_DEFINE_IMAGE3D_WRITE_BUILTIN(half, DF16_, 16, short, Dv4_DF16_)
#endif

// Sampled images
#define _CLC_DEFINE_SAMPLED_IMAGE_BUILTIN(dims)                                                                     \
  _CLC_DEF __ocl_sampled_image##dims##d_ro_t                                                                        \
      _Z20__spirv_SampledImageI14ocl_image##dims##d_ro32__spirv_SampledImage__image##dims##d_roET0_T_11ocl_sampler( \
          read_only image##dims##d_t image, sampler_t sampler) {                                                    \
    return __clc__sampled_image##dims##d_pack(image, sampler);                                                      \
  }

_CLC_DEFINE_SAMPLED_IMAGE_BUILTIN(1)
_CLC_DEFINE_SAMPLED_IMAGE_BUILTIN(2)
_CLC_DEFINE_SAMPLED_IMAGE_BUILTIN(3)

#undef _CLC_DEFINE_SAMPLED_IMAGE_BUILTIN

// TODO

// * coordinate_normalization_mode
// normalized
// unnormalized

// Connect each part in the resulting builtins.

float unnormalized_coord_1d(float coord, long image) {
  int width = __nvvm_suq_width(image);
  return coord * width;
}

float2 unnormalized_coord_2d(float2 coord, long image) {
  int width = __nvvm_suq_width(image);
  int height = __nvvm_suq_height(image);
  return (float2)(coord.x * width, coord.y * height);
}

float4 unnormalized_coord_3d(float4 coord, long image) {
  int width = __nvvm_suq_width(image);
  int height = __nvvm_suq_height(image);
  int depth = __nvvm_suq_depth(image);
  return (float4)(coord.x * width, coord.y * height, coord.z * depth, coord.w);
}

#define _DEFINE_COMMON_SAMPLED_LOAD_1D(elem_t, elem_size, ocl_address_mode,    \
                                       cuda_address_mode)                      \
  elem_t##4 ocl_address_mode##_address_mode_1d_##elem_t##4(                    \
      float coord, long image, int sampler) {                                  \
    if (is_nearest_filter_mode(sampler)) {                                     \
      int i = (int)__spirv_ocl_floor(coord);                                   \
      return as_##elem_t##4(                                                   \
          __nvvm_suld_1d_v4i##elem_size##_##cuda_address_mode##_s(             \
              image, i * sizeof(elem_t##4)));                                  \
    } else {                                                                   \
      int i0, i1;                                                              \
      fp##elem_size a =                                                        \
          get_common_linear_fract_and_coords_fp##elem_size(coord, &i0, &i1);   \
                                                                               \
      pixelf##elem_size Ti0 =                                                  \
          read_1d_##elem_size##_##cuda_address_mode(image, i0);                \
      pixelf##elem_size Ti1 =                                                  \
          read_1d_##elem_size##_##cuda_address_mode(image, i1);                \
                                                                               \
      return cast_pixelf##elem_size##_to_##elem_t##4((1 - a) * Ti0 + a * Ti1); \
    }                                                                          \
  }

#define _DEFINE_COMMON_SAMPLED_LOAD_2D(elem_t, elem_size, ocl_address_mode,    \
                                       cuda_address_mode)                      \
  elem_t##4 ocl_address_mode##_address_mode_2d_##elem_t##4(                    \
      float2 coord, long image, int sampler) {                                 \
    if (is_nearest_filter_mode(sampler)) {                                     \
      int i = (int)__spirv_ocl_floor(coord.x);                                 \
      int j = (int)__spirv_ocl_floor(coord.y);                                 \
      return as_##elem_t##4(                                                   \
          __nvvm_suld_2d_v4i##elem_size##_##cuda_address_mode##_s(             \
              image, i * sizeof(elem_t##4), j));                               \
    } else {                                                                   \
      int i0, i1, j0, j1;                                                      \
      fp##elem_size a =                                                        \
          get_common_linear_fract_and_coords_fp##elem_size(coord.x, &i0, &i1); \
      fp##elem_size b =                                                        \
          get_common_linear_fract_and_coords_fp##elem_size(coord.y, &j0, &j1); \
                                                                               \
      pixelf##elem_size Ti0j0 =                                                \
          read_2d_##elem_size##_##cuda_address_mode(image, i0, j0);            \
      pixelf##elem_size Ti1j0 =                                                \
          read_2d_##elem_size##_##cuda_address_mode(image, i1, j0);            \
      pixelf##elem_size Ti0j1 =                                                \
          read_2d_##elem_size##_##cuda_address_mode(image, i0, j1);            \
      pixelf##elem_size Ti1j1 =                                                \
          read_2d_##elem_size##_##cuda_address_mode(image, i1, j1);            \
                                                                               \
      return cast_pixelf##elem_size##_to_##elem_t##4(                          \
          (1 - a) * (1 - b) * Ti0j0 + a * (1 - b) * Ti1j0 +                    \
          (1 - a) * b * Ti0j1 + a * b * Ti1j1);                                \
    }                                                                          \
  }

#define _DEFINE_COMMON_SAMPLED_LOAD_3D(elem_t, elem_size, ocl_address_mode,    \
                                       cuda_address_mode)                      \
  elem_t##4 ocl_address_mode##_address_mode_3d_##elem_t##4(                    \
      float4 coord, long image, int sampler) {                                 \
    if (is_nearest_filter_mode(sampler)) {                                     \
      int i = (int)__spirv_ocl_floor(coord.x);                                 \
      int j = (int)__spirv_ocl_floor(coord.y);                                 \
      int k = (int)__spirv_ocl_floor(coord.z);                                 \
      return as_##elem_t##4(                                                   \
          __nvvm_suld_3d_v4i##elem_size##_##cuda_address_mode##_s(             \
              image, i * sizeof(elem_t##4), j, k));                            \
    } else {                                                                   \
      int i0, i1, j0, j1, k0, k1;                                              \
      fp##elem_size a =                                                        \
          get_common_linear_fract_and_coords_fp##elem_size(coord.x, &i0, &i1); \
      fp##elem_size b =                                                        \
          get_common_linear_fract_and_coords_fp##elem_size(coord.y, &j0, &j1); \
      fp##elem_size c =                                                        \
          get_common_linear_fract_and_coords_fp##elem_size(coord.z, &k0, &k1); \
                                                                               \
      pixelf##elem_size Ti0j0k0 =                                              \
          read_3d_##elem_size##_##cuda_address_mode(image, i0, j0, k0);        \
      pixelf##elem_size Ti1j0k0 =                                              \
          read_3d_##elem_size##_##cuda_address_mode(image, i1, j0, k0);        \
      pixelf##elem_size Ti0j1k0 =                                              \
          read_3d_##elem_size##_##cuda_address_mode(image, i0, j1, k0);        \
      pixelf##elem_size Ti1j1k0 =                                              \
          read_3d_##elem_size##_##cuda_address_mode(image, i1, j1, k0);        \
      pixelf##elem_size Ti0j0k1 =                                              \
          read_3d_##elem_size##_##cuda_address_mode(image, i0, j0, k1);        \
      pixelf##elem_size Ti1j0k1 =                                              \
          read_3d_##elem_size##_##cuda_address_mode(image, i1, j0, k1);        \
      pixelf##elem_size Ti0j1k1 =                                              \
          read_3d_##elem_size##_##cuda_address_mode(image, i0, j1, k1);        \
      pixelf##elem_size Ti1j1k1 =                                              \
          read_3d_##elem_size##_##cuda_address_mode(image, i1, j1, k1);        \
                                                                               \
      return cast_pixelf##elem_size##_to_##elem_t##4(                          \
          (1 - a) * (1 - b) * (1 - c) * Ti0j0k0 +                              \
          a * (1 - b) * (1 - c) * Ti1j0k0 + (1 - a) * b * (1 - c) * Ti0j1k0 +  \
          a * b * (1 - c) * Ti1j1k0 + (1 - a) * (1 - b) * c * Ti0j0k1 +        \
          a * (1 - b) * c * Ti1j0k1 + (1 - a) * b * c * Ti0j1k1 +              \
          a * b * c * Ti1j1k1);                                                \
    }                                                                          \
  }

#define _DEFINE_REPEAT_SAMPLED_LOAD_1D(elem_t, elem_size)                      \
  elem_t##4 repeat_address_mode_1d_##elem_t##4(float coord, long image,        \
                                               int sampler) {                  \
    int width = __nvvm_suq_width(image);                                       \
                                                                               \
    float u = (coord - __spirv_ocl_floor(coord)) * width;                      \
                                                                               \
    if (is_nearest_filter_mode(sampler)) {                                     \
      int i = (int)__spirv_ocl_floor(u);                                       \
      if (i > width - 1) {                                                     \
        i = i - width;                                                         \
      }                                                                        \
      return as_##elem_t##4(__nvvm_suld_1d_v4i##elem_size##_trap_s(            \
          image, i * sizeof(elem_t##4)));                                      \
    } else {                                                                   \
      int i0, i1;                                                              \
      fp##elem_size a =                                                        \
          get_common_linear_fract_and_coords_fp##elem_size(u, &i0, &i1);       \
                                                                               \
      if (i0 < 0) {                                                            \
        i0 = width + i0;                                                       \
      }                                                                        \
      if (i1 > width - 1) {                                                    \
        i1 = i1 - width;                                                       \
      }                                                                        \
                                                                               \
      pixelf##elem_size Ti0 = read_1d_##elem_size##_trap(image, i0);           \
      pixelf##elem_size Ti1 = read_1d_##elem_size##_trap(image, i1);           \
                                                                               \
      return cast_pixelf##elem_size##_to_##elem_t##4((1 - a) * Ti0 + a * Ti1); \
    }                                                                          \
  }

#define _DEFINE_REPEAT_SAMPLED_LOAD_2D(elem_t, elem_size)                      \
  elem_t##4 repeat_address_mode_2d_##elem_t##4(float2 coord, long image,       \
                                               int sampler) {                  \
    int width = __nvvm_suq_width(image);                                       \
    int height = __nvvm_suq_height(image);                                     \
                                                                               \
    float u = (coord.x - __spirv_ocl_floor(coord.x)) * width;                  \
    float v = (coord.y - __spirv_ocl_floor(coord.y)) * height;                 \
                                                                               \
    if (is_nearest_filter_mode(sampler)) {                                     \
      int i = (int)__spirv_ocl_floor(u);                                       \
      int j = (int)__spirv_ocl_floor(v);                                       \
      if (i > width - 1) {                                                     \
        i = i - width;                                                         \
      }                                                                        \
      if (j > height - 1) {                                                    \
        j = j - height;                                                        \
      }                                                                        \
      return as_##elem_t##4(__nvvm_suld_2d_v4i##elem_size##_trap_s(            \
          image, i * sizeof(elem_t##4), j));                                   \
    } else {                                                                   \
      int i0, i1, j0, j1;                                                      \
      fp##elem_size a =                                                        \
          get_common_linear_fract_and_coords_fp##elem_size(u, &i0, &i1);       \
      fp##elem_size b =                                                        \
          get_common_linear_fract_and_coords_fp##elem_size(v, &j0, &j1);       \
                                                                               \
      if (i0 < 0) {                                                            \
        i0 = width + i0;                                                       \
      }                                                                        \
      if (i1 > width - 1) {                                                    \
        i1 = i1 - width;                                                       \
      }                                                                        \
      if (j0 < 0) {                                                            \
        j0 = height + j0;                                                      \
      }                                                                        \
      if (j1 > height - 1) {                                                   \
        j1 = j1 - height;                                                      \
      }                                                                        \
                                                                               \
      pixelf##elem_size Ti0j0 = read_2d_##elem_size##_trap(image, i0, j0);     \
      pixelf##elem_size Ti1j0 = read_2d_##elem_size##_trap(image, i1, j0);     \
      pixelf##elem_size Ti0j1 = read_2d_##elem_size##_trap(image, i0, j1);     \
      pixelf##elem_size Ti1j1 = read_2d_##elem_size##_trap(image, i1, j1);     \
                                                                               \
      return cast_pixelf##elem_size##_to_##elem_t##4(                          \
          (1 - a) * (1 - b) * Ti0j0 + a * (1 - b) * Ti1j0 +                    \
          (1 - a) * b * Ti0j1 + a * b * Ti1j1);                                \
    }                                                                          \
  }

#define _DEFINE_REPEAT_SAMPLED_LOAD_3D(elem_t, elem_size)                      \
  elem_t##4 repeat_address_mode_3d_##elem_t##4(float4 coord, long image,       \
                                               int sampler) {                  \
    int width = __nvvm_suq_width(image);                                       \
    int height = __nvvm_suq_height(image);                                     \
    int depth = __nvvm_suq_depth(image);                                       \
                                                                               \
    float v = (coord.y - __spirv_ocl_floor(coord.y)) * height;                 \
    float u = (coord.x - __spirv_ocl_floor(coord.x)) * width;                  \
    float w = (coord.z - __spirv_ocl_floor(coord.z)) * depth;                  \
                                                                               \
    if (is_nearest_filter_mode(sampler)) {                                     \
      int i = (int)__spirv_ocl_floor(u);                                       \
      int j = (int)__spirv_ocl_floor(v);                                       \
      int k = (int)__spirv_ocl_floor(w);                                       \
      if (i > width - 1) {                                                     \
        i = i - width;                                                         \
      }                                                                        \
      if (j > height - 1) {                                                    \
        j = j - height;                                                        \
      }                                                                        \
      if (k > depth - 1) {                                                     \
        k = k - depth;                                                         \
      }                                                                        \
      return as_##elem_t##4(__nvvm_suld_3d_v4i##elem_size##_trap_s(            \
          image, i * sizeof(elem_t##4), j, k));                                \
    } else {                                                                   \
      int i0, i1, j0, j1, k0, k1;                                              \
      fp##elem_size a =                                                        \
          get_common_linear_fract_and_coords_fp##elem_size(u, &i0, &i1);       \
      fp##elem_size b =                                                        \
          get_common_linear_fract_and_coords_fp##elem_size(v, &j0, &j1);       \
      fp##elem_size c =                                                        \
          get_common_linear_fract_and_coords_fp##elem_size(w, &k0, &k1);       \
                                                                               \
      if (i0 < 0) {                                                            \
        i0 = width + i0;                                                       \
      }                                                                        \
      if (i1 > width - 1) {                                                    \
        i1 = i1 - width;                                                       \
      }                                                                        \
      if (j0 < 0) {                                                            \
        j0 = height + j0;                                                      \
      }                                                                        \
      if (j1 > height - 1) {                                                   \
        j1 = j1 - height;                                                      \
      }                                                                        \
      if (k0 < 0) {                                                            \
        k0 = depth + k0;                                                       \
      }                                                                        \
      if (k1 > depth - 1) {                                                    \
        k1 = k1 - depth;                                                       \
      }                                                                        \
                                                                               \
      pixelf##elem_size Ti0j0k0 =                                              \
          read_3d_##elem_size##_trap(image, i0, j0, k0);                       \
      pixelf##elem_size Ti1j0k0 =                                              \
          read_3d_##elem_size##_trap(image, i1, j0, k0);                       \
      pixelf##elem_size Ti0j1k0 =                                              \
          read_3d_##elem_size##_trap(image, i0, j1, k0);                       \
      pixelf##elem_size Ti1j1k0 =                                              \
          read_3d_##elem_size##_trap(image, i1, j1, k0);                       \
      pixelf##elem_size Ti0j0k1 =                                              \
          read_3d_##elem_size##_trap(image, i0, j0, k1);                       \
      pixelf##elem_size Ti1j0k1 =                                              \
          read_3d_##elem_size##_trap(image, i1, j0, k1);                       \
      pixelf##elem_size Ti0j1k1 =                                              \
          read_3d_##elem_size##_trap(image, i0, j1, k1);                       \
      pixelf##elem_size Ti1j1k1 =                                              \
          read_3d_##elem_size##_trap(image, i1, j1, k1);                       \
                                                                               \
      return cast_pixelf##elem_size##_to_##elem_t##4(                          \
          (1 - a) * (1 - b) * (1 - c) * Ti0j0k0 +                              \
          a * (1 - b) * (1 - c) * Ti1j0k0 + (1 - a) * b * (1 - c) * Ti0j1k0 +  \
          a * b * (1 - c) * Ti1j1k0 + (1 - a) * (1 - b) * c * Ti0j0k1 +        \
          a * (1 - b) * c * Ti1j0k1 + (1 - a) * b * c * Ti0j1k1 +              \
          a * b * c * Ti1j1k1);                                                \
    }                                                                          \
  }

#define _DEFINE_MIRRORED_REPEAT_SAMPLED_LOAD_1D(elem_t, elem_size)             \
  elem_t##4 mirrored_repeat_address_mode_1d_##elem_t##4(                       \
      float coord, long image, int sampler) {                                  \
    int width = __nvvm_suq_width(image);                                       \
                                                                               \
    float sp = 2.0f * __spirv_ocl_rint(0.5f * coord);                          \
    sp = __spirv_ocl_fabs(coord - sp);                                         \
    float u = sp * width;                                                      \
                                                                               \
    if (is_nearest_filter_mode(sampler)) {                                     \
      int i = (int)__spirv_ocl_floor(u);                                       \
      i = __spirv_ocl_s_min(i, width - 1);                                     \
                                                                               \
      return as_##elem_t##4(__nvvm_suld_1d_v4i##elem_size##_trap_s(            \
          image, i * sizeof(elem_t##4)));                                      \
    } else {                                                                   \
      int i0, i1;                                                              \
      fp##elem_size a =                                                        \
          get_common_linear_fract_and_coords_fp##elem_size(u, &i0, &i1);       \
                                                                               \
      i0 = __spirv_ocl_s_max(i0, 0);                                           \
      i1 = __spirv_ocl_s_min(i1, width - 1);                                   \
                                                                               \
      pixelf##elem_size Ti0 = read_1d_##elem_size##_trap(image, i0);           \
      pixelf##elem_size Ti1 = read_1d_##elem_size##_trap(image, i1);           \
                                                                               \
      return cast_pixelf##elem_size##_to_##elem_t##4((1 - a) * Ti0 + a * Ti1); \
    }                                                                          \
  }

#define _DEFINE_MIRRORED_REPEAT_SAMPLED_LOAD_2D(elem_t, elem_size)             \
  elem_t##4 mirrored_repeat_address_mode_2d_##elem_t##4(                       \
      float2 coord, long image, int sampler) {                                 \
    int width = __nvvm_suq_width(image);                                       \
    int height = __nvvm_suq_height(image);                                     \
                                                                               \
    float sp = 2.0f * __spirv_ocl_rint(0.5f * coord.x);                        \
    float tp = 2.0f * __spirv_ocl_rint(0.5f * coord.y);                        \
    sp = __spirv_ocl_fabs(coord.x - sp);                                       \
    tp = __spirv_ocl_fabs(coord.y - tp);                                       \
    float u = sp * width;                                                      \
    float v = tp * height;                                                     \
                                                                               \
    if (is_nearest_filter_mode(sampler)) {                                     \
      int i = (int)__spirv_ocl_floor(u);                                       \
      i = __spirv_ocl_s_min(i, width - 1);                                     \
      int j = (int)__spirv_ocl_floor(v);                                       \
      j = __spirv_ocl_s_min(j, height - 1);                                    \
                                                                               \
      return as_##elem_t##4(__nvvm_suld_2d_v4i##elem_size##_trap_s(            \
          image, i * sizeof(elem_t##4), j));                                   \
    } else {                                                                   \
      int i0, i1, j0, j1;                                                      \
      fp##elem_size a =                                                        \
          get_common_linear_fract_and_coords_fp##elem_size(u, &i0, &i1);       \
      fp##elem_size b =                                                        \
          get_common_linear_fract_and_coords_fp##elem_size(v, &j0, &j1);       \
                                                                               \
      i0 = __spirv_ocl_s_max(i0, 0);                                           \
      i1 = __spirv_ocl_s_min(i1, width - 1);                                   \
      j0 = __spirv_ocl_s_max(j0, 0);                                           \
      j1 = __spirv_ocl_s_min(j1, height - 1);                                  \
                                                                               \
      pixelf##elem_size Ti0j0 = read_2d_##elem_size##_trap(image, i0, j0);     \
      pixelf##elem_size Ti1j0 = read_2d_##elem_size##_trap(image, i1, j0);     \
      pixelf##elem_size Ti0j1 = read_2d_##elem_size##_trap(image, i0, j1);     \
      pixelf##elem_size Ti1j1 = read_2d_##elem_size##_trap(image, i1, j1);     \
                                                                               \
      return cast_pixelf##elem_size##_to_##elem_t##4(                          \
          (1 - a) * (1 - b) * Ti0j0 + a * (1 - b) * Ti1j0 +                    \
          (1 - a) * b * Ti0j1 + a * b * Ti1j1);                                \
    }                                                                          \
  }

#define _DEFINE_MIRRORED_REPEAT_SAMPLED_LOAD_3D(elem_t, elem_size)             \
  elem_t##4 mirrored_repeat_address_mode_3d_##elem_t##4(                       \
      float4 coord, long image, int sampler) {                                 \
    int width = __nvvm_suq_width(image);                                       \
    int height = __nvvm_suq_height(image);                                     \
    int depth = __nvvm_suq_depth(image);                                       \
                                                                               \
    float sp = 2.0f * __spirv_ocl_rint(0.5f * coord.x);                        \
    float tp = 2.0f * __spirv_ocl_rint(0.5f * coord.y);                        \
    float rp = 2.0f * __spirv_ocl_rint(0.5f * coord.z);                        \
    sp = __spirv_ocl_fabs(coord.x - sp);                                       \
    tp = __spirv_ocl_fabs(coord.y - tp);                                       \
    rp = __spirv_ocl_fabs(coord.z - rp);                                       \
    float u = sp * width;                                                      \
    float v = tp * height;                                                     \
    float w = rp * depth;                                                      \
                                                                               \
    if (is_nearest_filter_mode(sampler)) {                                     \
      int i = (int)__spirv_ocl_floor(u);                                       \
      i = __spirv_ocl_s_min(i, width - 1);                                     \
      int j = (int)__spirv_ocl_floor(v);                                       \
      j = __spirv_ocl_s_min(j, height - 1);                                    \
      int k = (int)__spirv_ocl_floor(w);                                       \
      k = __spirv_ocl_s_min(k, depth - 1);                                     \
                                                                               \
      return as_##elem_t##4(__nvvm_suld_3d_v4i##elem_size##_trap_s(            \
          image, i * sizeof(elem_t##4), j, k));                                \
    } else {                                                                   \
      int i0, i1, j0, j1, k0, k1;                                              \
      fp##elem_size a =                                                        \
          get_common_linear_fract_and_coords_fp##elem_size(u, &i0, &i1);       \
      fp##elem_size b =                                                        \
          get_common_linear_fract_and_coords_fp##elem_size(v, &j0, &j1);       \
      fp##elem_size c =                                                        \
          get_common_linear_fract_and_coords_fp##elem_size(w, &k0, &k1);       \
                                                                               \
      i0 = __spirv_ocl_s_max(i0, 0);                                           \
      i1 = __spirv_ocl_s_min(i1, width - 1);                                   \
      j0 = __spirv_ocl_s_max(j0, 0);                                           \
      j1 = __spirv_ocl_s_min(j1, height - 1);                                  \
      k0 = __spirv_ocl_s_max(k0, 0);                                           \
      k1 = __spirv_ocl_s_min(k1, depth - 1);                                   \
                                                                               \
      pixelf##elem_size Ti0j0k0 =                                              \
          read_3d_##elem_size##_trap(image, i0, j0, k0);                       \
      pixelf##elem_size Ti1j0k0 =                                              \
          read_3d_##elem_size##_trap(image, i1, j0, k0);                       \
      pixelf##elem_size Ti0j1k0 =                                              \
          read_3d_##elem_size##_trap(image, i0, j1, k0);                       \
      pixelf##elem_size Ti1j1k0 =                                              \
          read_3d_##elem_size##_trap(image, i1, j1, k0);                       \
      pixelf##elem_size Ti0j0k1 =                                              \
          read_3d_##elem_size##_trap(image, i0, j0, k1);                       \
      pixelf##elem_size Ti1j0k1 =                                              \
          read_3d_##elem_size##_trap(image, i1, j0, k1);                       \
      pixelf##elem_size Ti0j1k1 =                                              \
          read_3d_##elem_size##_trap(image, i0, j1, k1);                       \
      pixelf##elem_size Ti1j1k1 =                                              \
          read_3d_##elem_size##_trap(image, i1, j1, k1);                       \
                                                                               \
      return cast_pixelf##elem_size##_to_##elem_t##4(                          \
          (1 - a) * (1 - b) * (1 - c) * Ti0j0k0 +                              \
          a * (1 - b) * (1 - c) * Ti1j0k0 + (1 - a) * b * (1 - c) * Ti0j1k0 +  \
          a * b * (1 - c) * Ti1j1k0 + (1 - a) * (1 - b) * c * Ti0j0k1 +        \
          a * (1 - b) * c * Ti1j0k1 + (1 - a) * b * c * Ti0j1k1 +              \
          a * b * c * Ti1j1k1);                                                \
    }                                                                          \
  }

#define _DEFINE_SAMPLED_LOADS(elem_t, elem_size)                               \
  _DEFINE_COMMON_SAMPLED_LOAD_1D(elem_t, elem_size, none, zero)                \
  _DEFINE_COMMON_SAMPLED_LOAD_2D(elem_t, elem_size, none, zero)                \
  _DEFINE_COMMON_SAMPLED_LOAD_3D(elem_t, elem_size, none, zero)                \
  _DEFINE_COMMON_SAMPLED_LOAD_1D(elem_t, elem_size, clamp, zero)               \
  _DEFINE_COMMON_SAMPLED_LOAD_2D(elem_t, elem_size, clamp, zero)               \
  _DEFINE_COMMON_SAMPLED_LOAD_3D(elem_t, elem_size, clamp, zero)               \
  _DEFINE_COMMON_SAMPLED_LOAD_1D(elem_t, elem_size, clamp_to_edge, clamp)      \
  _DEFINE_COMMON_SAMPLED_LOAD_2D(elem_t, elem_size, clamp_to_edge, clamp)      \
  _DEFINE_COMMON_SAMPLED_LOAD_3D(elem_t, elem_size, clamp_to_edge, clamp)      \
  _DEFINE_REPEAT_SAMPLED_LOAD_1D(elem_t, elem_size)                            \
  _DEFINE_REPEAT_SAMPLED_LOAD_2D(elem_t, elem_size)                            \
  _DEFINE_REPEAT_SAMPLED_LOAD_3D(elem_t, elem_size)                            \
  _DEFINE_MIRRORED_REPEAT_SAMPLED_LOAD_1D(elem_t, elem_size)                   \
  _DEFINE_MIRRORED_REPEAT_SAMPLED_LOAD_2D(elem_t, elem_size)                   \
  _DEFINE_MIRRORED_REPEAT_SAMPLED_LOAD_3D(elem_t, elem_size)

_DEFINE_SAMPLED_LOADS(float, 32)
_DEFINE_SAMPLED_LOADS(int, 32)
_DEFINE_SAMPLED_LOADS(uint, 32)

#ifdef cl_khr_fp16
_DEFINE_SAMPLED_LOADS(half, 16)
#endif

#undef _DEFINE_SAMPLED_LOADS
#undef _DEFINE_COMMON_SAMPLED_LOAD_1D
#undef _DEFINE_COMMON_SAMPLED_LOAD_2D
#undef _DEFINE_COMMON_SAMPLED_LOAD_3D
#undef _DEFINE_REPEAT_SAMPLED_LOAD_1D
#undef _DEFINE_REPEAT_SAMPLED_LOAD_2D
#undef _DEFINE_REPEAT_SAMPLED_LOAD_3D
#undef _DEFINE_MIRRORED_REPEAT_SAMPLED_LOAD_1D
#undef _DEFINE_MIRRORED_REPEAT_SAMPLED_LOAD_2D
#undef _DEFINE_MIRRORED_REPEAT_SAMPLED_LOAD_3D

#define _CLC_DEFINE_IMAGE_SAMPLED_READ_BUILTIN(                                                                                                   \
    elem_t, elem_t_mangled, dims, input_coord_t, input_coord_t_mangled,                                                                           \
    sampling_coord_t)                                                                                                                             \
  _CLC_DEF                                                                                                                                        \
  elem_t##4 _Z30__spirv_ImageSampleExplicitLodI32__spirv_SampledImage__image##dims##d_roDv4_##elem_t_mangled##input_coord_t_mangled##ET0_T_T1_if( \
      __ocl_sampled_image##dims##d_ro_t sampled_image,                                                                                            \
      input_coord_t input_coord, int operands, float lod) {                                                                                       \
    long image = __clc__sampled_image##dims##d_unpack_image(sampled_image);                                                                       \
    int sampler = __clc__sampled_image##dims##d_unpack_sampler(sampled_image);                                                                    \
    /* Sampling algorithms are implemented assu__spirv_ocl_s_ming an                                                                              \
     * unnormalized floating point coordinate as input. Need to transform as                                                                      \
     * appropriate. */                                                                                                                            \
    sampling_coord_t sampling_coord =                                                                                                             \
        cast_##input_coord_t##_to_##sampling_coord_t(input_coord);                                                                                \
    if (is_normalized_coords(sampler)) {                                                                                                          \
      sampling_coord = unnormalized_coord_##dims##d(sampling_coord, image);                                                                       \
    }                                                                                                                                             \
    if (is_address_mode(sampler, 0)) { /* ADDRESS_NONE */                                                                                         \
      return none_address_mode_##dims##d_##elem_t##4(sampling_coord, image,                                                                       \
                                                     sampler);                                                                                    \
    }                                                                                                                                             \
    if (is_address_mode(sampler, 1)) { /* ADDRESS_CLAMP_TO_EDGE */                                                                                \
      return clamp_to_edge_address_mode_##dims##d_##elem_t##4(sampling_coord,                                                                     \
                                                              image, sampler);                                                                    \
    }                                                                                                                                             \
    if (is_address_mode(sampler, 2)) { /* ADDRESS_CLAMP */                                                                                        \
      return clamp_address_mode_##dims##d_##elem_t##4(sampling_coord, image,                                                                      \
                                                      sampler);                                                                                   \
    }                                                                                                                                             \
    if (is_address_mode(sampler, 3)) { /* ADDRESS_REPEAT */                                                                                       \
      return repeat_address_mode_##dims##d_##elem_t##4(sampling_coord, image,                                                                     \
                                                       sampler);                                                                                  \
    }                                                                                                                                             \
    /* ADDRESS_MIRRORED_REPEAT */                                                                                                                 \
    return mirrored_repeat_address_mode_##dims##d_##elem_t##4(sampling_coord,                                                                     \
                                                              image, sampler);                                                                    \
  }

_CLC_DEFINE_IMAGE_SAMPLED_READ_BUILTIN(float, f, 1, float, f, float)
_CLC_DEFINE_IMAGE_SAMPLED_READ_BUILTIN(float, f, 2, float2, Dv2_f, float2)
_CLC_DEFINE_IMAGE_SAMPLED_READ_BUILTIN(float, f, 3, float4, S1_, float4)
_CLC_DEFINE_IMAGE_SAMPLED_READ_BUILTIN(float, f, 1, int, i, float)
_CLC_DEFINE_IMAGE_SAMPLED_READ_BUILTIN(float, f, 2, int2, Dv2_i, float2)
_CLC_DEFINE_IMAGE_SAMPLED_READ_BUILTIN(float, f, 3, int4, Dv4_i, float4)

_CLC_DEFINE_IMAGE_SAMPLED_READ_BUILTIN(int, i, 1, float, f, float)
_CLC_DEFINE_IMAGE_SAMPLED_READ_BUILTIN(int, i, 2, float2, Dv2_f, float2)
_CLC_DEFINE_IMAGE_SAMPLED_READ_BUILTIN(int, i, 3, float4, Dv4_f, float4)
_CLC_DEFINE_IMAGE_SAMPLED_READ_BUILTIN(int, i, 1, int, i, float)
_CLC_DEFINE_IMAGE_SAMPLED_READ_BUILTIN(int, i, 2, int2, Dv2_i, float2)
_CLC_DEFINE_IMAGE_SAMPLED_READ_BUILTIN(int, i, 3, int4, S1_, float4)

_CLC_DEFINE_IMAGE_SAMPLED_READ_BUILTIN(uint, j, 1, float, f, float)
_CLC_DEFINE_IMAGE_SAMPLED_READ_BUILTIN(uint, j, 2, float2, Dv2_f, float2)
_CLC_DEFINE_IMAGE_SAMPLED_READ_BUILTIN(uint, j, 3, float4, Dv4_f, float4)
_CLC_DEFINE_IMAGE_SAMPLED_READ_BUILTIN(uint, j, 1, int, i, float)
_CLC_DEFINE_IMAGE_SAMPLED_READ_BUILTIN(uint, j, 2, int2, Dv2_i, float2)
_CLC_DEFINE_IMAGE_SAMPLED_READ_BUILTIN(uint, j, 3, int4, Dv4_i, float4)

#ifdef cl_khr_fp16
_CLC_DEFINE_IMAGE_SAMPLED_READ_BUILTIN(half, DF16_, 1, float, f, float)
_CLC_DEFINE_IMAGE_SAMPLED_READ_BUILTIN(half, DF16_, 2, float2, Dv2_f, float2)
_CLC_DEFINE_IMAGE_SAMPLED_READ_BUILTIN(half, DF16_, 3, float4, Dv4_f, float4)
_CLC_DEFINE_IMAGE_SAMPLED_READ_BUILTIN(half, DF16_, 1, int, i, float)
_CLC_DEFINE_IMAGE_SAMPLED_READ_BUILTIN(half, DF16_, 2, int2, Dv2_i, float2)
_CLC_DEFINE_IMAGE_SAMPLED_READ_BUILTIN(half, DF16_, 3, int4, Dv4_i, float4)
#endif

#undef _CLC_DEFINE_IMAGE1D_READ_BUILTIN
#undef _CLC_DEFINE_IMAGE2D_READ_BUILTIN
#undef _CLC_DEFINE_IMAGE3D_READ_BUILTIN
#undef _CLC_DEFINE_IMAGE1D_WRITE_BUILTIN
#undef _CLC_DEFINE_IMAGE2D_WRITE_BUILTIN
#undef _CLC_DEFINE_IMAGE3D_WRITE_BUILTIN
#undef _CLC_DEFINE_IMAGE_SAMPLED_READ_BUILTIN

// Size Queries
_CLC_DEF int _Z22__spirv_ImageQuerySizeIDv1_i14ocl_image1d_roET_T0_(
    read_only image1d_t image) {
  return __nvvm_suq_width_1i(image);
}

_CLC_DEF int2 _Z22__spirv_ImageQuerySizeIDv2_i14ocl_image2d_roET_T0_(
    read_only image2d_t image) {
  int width = __nvvm_suq_width_2i(image);
  int height = __nvvm_suq_height_2i(image);
  return (int2)(width, height);
}

_CLC_DEF int3 _Z22__spirv_ImageQuerySizeIDv3_i14ocl_image3d_roET_T0_(
    read_only image3d_t image) {
  int width = __nvvm_suq_width_3i(image);
  int height = __nvvm_suq_height_3i(image);
  int depth = __nvvm_suq_depth_3i(image);
  return (int3)(width, height, depth);
}

// <--- BINDLESS IMAGES PROTOTYPE --->

// UNSAMPLED IMAGES

// Generated funcs -- READS
// int -- int4 already defined
int __nvvm_suld_1d_i32_clamp_s(long, int) __asm("llvm.nvvm.suld.1d.i32.clamp");
int __nvvm_suld_2d_i32_clamp_s(long, int,
                               int) __asm("llvm.nvvm.suld.2d.i32.clamp");
int __nvvm_suld_3d_i32_clamp_s(long, int, int,
                               int) __asm("llvm.nvvm.suld.3d.i32.clamp");

int2 __nvvm_suld_1d_v2i32_clamp_s(long imageHandle, int coord) {
  int4 a = __nvvm_suld_1d_v4i32_clamp_s(imageHandle, coord);
  return cast_int4_to_int2(a);
}

int2 __nvvm_suld_2d_v2i32_clamp_s(long imageHandle, int x, int y) {
  int4 a = __nvvm_suld_2d_v4i32_clamp_s(imageHandle, x, y);
  return cast_int4_to_int2(a);
}

int2 __nvvm_suld_3d_v2i32_clamp_s(long imageHandle, int x, int y, int z) {
  int4 a = __nvvm_suld_3d_v4i32_clamp_s(imageHandle, x, y, z);
  return cast_int4_to_int2(a);
}

// unsigned int
unsigned int __nvvm_suld_1d_j32_clamp_s(long imageHandle, int coord) {
  return as_uint(__nvvm_suld_1d_i32_clamp_s(imageHandle, coord));
}
unsigned int __nvvm_suld_2d_j32_clamp_s(long imageHandle, int x, int y) {
  return as_uint(__nvvm_suld_2d_i32_clamp_s(imageHandle, x, y));
}
unsigned int __nvvm_suld_3d_j32_clamp_s(long imageHandle, int x, int y, int z) {
  return as_uint(__nvvm_suld_3d_i32_clamp_s(imageHandle, x, y, z));
}
uint2 __nvvm_suld_1d_v2j32_clamp_s(long imageHandle, int coord) {
  return as_uint2(__nvvm_suld_1d_v2i32_clamp_s(imageHandle, coord));
}
uint2 __nvvm_suld_2d_v2j32_clamp_s(long imageHandle, int x, int y) {
  return as_uint2(__nvvm_suld_2d_v2i32_clamp_s(imageHandle, x, y));
}
uint2 __nvvm_suld_3d_v2j32_clamp_s(long imageHandle, int x, int y, int z) {
  return as_uint2(__nvvm_suld_3d_v2i32_clamp_s(imageHandle, x, y, z));
}
uint4 __nvvm_suld_1d_v4j32_clamp_s(long imageHandle, int coord) {
  return as_uint4(__nvvm_suld_1d_v4i32_clamp_s(imageHandle, coord));
}
uint4 __nvvm_suld_2d_v4j32_clamp_s(long imageHandle, int x, int y) {
  return as_uint4(__nvvm_suld_2d_v4i32_clamp_s(imageHandle, x, y));
}
uint4 __nvvm_suld_3d_v4j32_clamp_s(long imageHandle, int x, int y, int z) {
  return as_uint4(__nvvm_suld_3d_v4i32_clamp_s(imageHandle, x, y, z));
}

// short -- short4 already defined
short __nvvm_suld_1d_i16_clamp_s(long,
                                 int) __asm("llvm.nvvm.suld.1d.i16.clamp");
short __nvvm_suld_2d_i16_clamp_s(long, int,
                                 int) __asm("llvm.nvvm.suld.2d.i16.clamp");
short __nvvm_suld_3d_i16_clamp_s(long, int, int,
                                 int) __asm("llvm.nvvm.suld.3d.i16.clamp");

short2 __nvvm_suld_1d_v2i16_clamp_s(long imageHandle, int coord) {
  short4 a = __nvvm_suld_1d_v4i16_clamp_s(imageHandle, coord);
  return cast_short4_to_short2(a);
}

short2 __nvvm_suld_2d_v2i16_clamp_s(long imageHandle, int x, int y) {
  short4 a = __nvvm_suld_2d_v4i16_clamp_s(imageHandle, x, y);
  return cast_short4_to_short2(a);
}

short2 __nvvm_suld_3d_v2i16_clamp_s(long imageHandle, int x, int y, int z) {
  short4 a = __nvvm_suld_3d_v4i16_clamp_s(imageHandle, x, y, z);
  return cast_short4_to_short2(a);
}

// unsigned short
unsigned short __nvvm_suld_1d_t16_clamp_s(long imageHandle, int coord) {
  return as_ushort(__nvvm_suld_1d_i16_clamp_s(imageHandle, coord));
}
unsigned short __nvvm_suld_2d_t16_clamp_s(long imageHandle, int x, int y) {
  return as_ushort(__nvvm_suld_2d_i16_clamp_s(imageHandle, x, y));
}
unsigned short __nvvm_suld_3d_t16_clamp_s(long imageHandle, int x, int y,
                                          int z) {
  return as_ushort(__nvvm_suld_3d_i16_clamp_s(imageHandle, x, y, z));
}
ushort2 __nvvm_suld_1d_v2t16_clamp_s(long imageHandle, int coord) {
  return as_ushort2(__nvvm_suld_1d_v2i16_clamp_s(imageHandle, coord));
}
ushort2 __nvvm_suld_2d_v2t16_clamp_s(long imageHandle, int x, int y) {
  return as_ushort2(__nvvm_suld_2d_v2i16_clamp_s(imageHandle, x, y));
}
ushort2 __nvvm_suld_3d_v2t16_clamp_s(long imageHandle, int x, int y, int z) {
  return as_ushort2(__nvvm_suld_3d_v2i16_clamp_s(imageHandle, x, y, z));
}
ushort4 __nvvm_suld_1d_v4t16_clamp_s(long imageHandle, int coord) {
  return as_ushort4(__nvvm_suld_1d_v4i16_clamp_s(imageHandle, coord));
}
ushort4 __nvvm_suld_2d_v4t16_clamp_s(long imageHandle, int x, int y) {
  return as_ushort4(__nvvm_suld_2d_v4i16_clamp_s(imageHandle, x, y));
}
ushort4 __nvvm_suld_3d_v4t16_clamp_s(long imageHandle, int x, int y, int z) {
  return as_ushort4(__nvvm_suld_3d_v4i16_clamp_s(imageHandle, x, y, z));
}

// signed char
short __nvvm_suld_1d_i8_clamp_s_helper(long,
                                       int) __asm("llvm.nvvm.suld.1d.i8.clamp");
signed char __nvvm_suld_1d_i8_clamp_s(long imageHandle, int coord) {
  return as_char(
      (signed char)__nvvm_suld_1d_i8_clamp_s_helper(imageHandle, coord));
}

short __nvvm_suld_2d_i8_clamp_s_helper(long, int,
                                       int) __asm("llvm.nvvm.suld.2d.i8.clamp");
signed char __nvvm_suld_2d_i8_clamp_s(long imageHandle, int x, int y) {
  return as_char(
      (signed char)__nvvm_suld_2d_i8_clamp_s_helper(imageHandle, x, y));
}

short __nvvm_suld_3d_i8_clamp_s_helper(long, int, int,
                                       int) __asm("llvm.nvvm.suld.3d.i8.clamp");
signed char __nvvm_suld_3d_i8_clamp_s(long imageHandle, int x, int y, int z) {
  return as_char(
      (signed char)__nvvm_suld_3d_i8_clamp_s_helper(imageHandle, x, y, z));
}

short2 __nvvm_suld_1d_v2i8_clamp_s_helper(long, int) __asm(
    "__clc_llvm_nvvm_suld_1d_v2i8_clamp");
char2 __nvvm_suld_1d_v2i8_clamp_s(long imageHandle, int coord) {
  short2 a = __nvvm_suld_1d_v2i8_clamp_s_helper(imageHandle, coord);
  return cast_short2_to_char2(a);
}

short2 __nvvm_suld_2d_v2i8_clamp_s_helper(long, int, int) __asm(
    "__clc_llvm_nvvm_suld_2d_v2i8_clamp");
char2 __nvvm_suld_2d_v2i8_clamp_s(long imageHandle, int x, int y) {
  short2 a = __nvvm_suld_2d_v2i8_clamp_s_helper(imageHandle, x, y);
  return cast_short2_to_char2(a);
}

short2 __nvvm_suld_3d_v2i8_clamp_s_helper(long, int, int, int) __asm(
    "__clc_llvm_nvvm_suld_3d_v2i8_clamp");
char2 __nvvm_suld_3d_v2i8_clamp_s(long imageHandle, int x, int y, int z) {
  short2 a = __nvvm_suld_3d_v2i8_clamp_s_helper(imageHandle, x, y, z);
  return cast_short2_to_char2(a);
}

short4 __nvvm_suld_1d_v4i8_clamp_s_helper(long, int) __asm(
    "__clc_llvm_nvvm_suld_1d_v4i8_clamp");
char4 __nvvm_suld_1d_v4i8_clamp_s(long imageHandle, int coord) {
  short4 a = __nvvm_suld_1d_v4i8_clamp_s_helper(imageHandle, coord);
  return cast_short4_to_char4(a);
}

short4 __nvvm_suld_2d_v4i8_clamp_s_helper(long, int, int) __asm(
    "__clc_llvm_nvvm_suld_2d_v4i8_clamp");
char4 __nvvm_suld_2d_v4i8_clamp_s(long imageHandle, int x, int y) {
  short4 a = __nvvm_suld_2d_v4i8_clamp_s_helper(imageHandle, x, y);
  return cast_short4_to_char4(a);
}

short4 __nvvm_suld_3d_v4i8_clamp_s_helper(long, int, int, int) __asm(
    "__clc_llvm_nvvm_suld_3d_v4i8_clamp");
char4 __nvvm_suld_3d_v4i8_clamp_s(long imageHandle, int x, int y, int z) {
  short4 a = __nvvm_suld_3d_v4i8_clamp_s_helper(imageHandle, x, y, z);
  return cast_short4_to_char4(a);
}

// unsigned char
unsigned short
__nvvm_suld_1d_h8_clamp_s_helper(long, int) __asm("llvm.nvvm.suld.1d.i8.clamp");
unsigned char __nvvm_suld_1d_h8_clamp_s(long imageHandle, int coord) {
  return as_uchar(
      (unsigned char)__nvvm_suld_1d_h8_clamp_s_helper(imageHandle, coord));
}

unsigned short
__nvvm_suld_2d_h8_clamp_s_helper(long, int,
                                 int) __asm("llvm.nvvm.suld.2d.i8.clamp");
unsigned char __nvvm_suld_2d_h8_clamp_s(long imageHandle, int x, int y) {
  return as_uchar(
      (unsigned char)__nvvm_suld_2d_h8_clamp_s_helper(imageHandle, x, y));
}

unsigned short
__nvvm_suld_3d_h8_clamp_s_helper(long, int, int,
                                 int) __asm("llvm.nvvm.suld.3d.i8.clamp");

unsigned char __nvvm_suld_3d_h8_clamp_s(long imageHandle, int x, int y, int z) {
  return as_uchar(
      (unsigned char)__nvvm_suld_3d_h8_clamp_s_helper(imageHandle, x, y, z));
}

uchar2 __nvvm_suld_1d_v2h8_clamp_s(long imageHandle, int coord) {
  short2 a = __nvvm_suld_1d_v2i8_clamp_s_helper(imageHandle, coord);
  return cast_short2_to_uchar2(a);
}

uchar2 __nvvm_suld_2d_v2h8_clamp_s(long imageHandle, int x, int y) {
  short2 a = __nvvm_suld_2d_v2i8_clamp_s_helper(imageHandle, x, y);
  return cast_short2_to_uchar2(a);
}

uchar2 __nvvm_suld_3d_v2h8_clamp_s(long imageHandle, int x, int y, int z) {
  short2 a = __nvvm_suld_3d_v2i8_clamp_s_helper(imageHandle, x, y, z);
  return cast_short2_to_uchar2(a);
}

uchar4 __nvvm_suld_1d_v4h8_clamp_s(long imageHandle, int coord) {
  short4 a = __nvvm_suld_1d_v4i8_clamp_s_helper(imageHandle, coord);
  return cast_short4_to_uchar4(a);
}

uchar4 __nvvm_suld_2d_v4h8_clamp_s(long imageHandle, int x, int y) {
  short4 a = __nvvm_suld_2d_v4i8_clamp_s_helper(imageHandle, x, y);
  return cast_short4_to_uchar4(a);
}

uchar4 __nvvm_suld_3d_v4h8_clamp_s(long imageHandle, int x, int y, int z) {
  short4 a = __nvvm_suld_3d_v4i8_clamp_s_helper(imageHandle, x, y, z);
  return cast_short4_to_uchar4(a);
}

// float
float __nvvm_suld_1d_f32_clamp_s(long imageHandle, int coord) {
  return as_float(__nvvm_suld_1d_i32_clamp_s(imageHandle, coord));
}
float __nvvm_suld_2d_f32_clamp_s(long imageHandle, int x, int y) {
  return as_float(__nvvm_suld_2d_i32_clamp_s(imageHandle, x, y));
}
float __nvvm_suld_3d_f32_clamp_s(long imageHandle, int x, int y, int z) {
  return as_float(__nvvm_suld_3d_i32_clamp_s(imageHandle, x, y, z));
}
float2 __nvvm_suld_1d_v2f32_clamp_s(long imageHandle, int coord) {
  return as_float2(__nvvm_suld_1d_v2i32_clamp_s(imageHandle, coord));
}
float2 __nvvm_suld_2d_v2f32_clamp_s(long imageHandle, int x, int y) {
  return as_float2(__nvvm_suld_2d_v2i32_clamp_s(imageHandle, x, y));
}
float2 __nvvm_suld_3d_v2f32_clamp_s(long imageHandle, int x, int y, int z) {
  return as_float2(__nvvm_suld_3d_v2i32_clamp_s(imageHandle, x, y, z));
}
float4 __nvvm_suld_1d_v4f32_clamp_s(long imageHandle, int coord) {
  return as_float4(__nvvm_suld_1d_v4i32_clamp_s(imageHandle, coord));
}
float4 __nvvm_suld_2d_v4f32_clamp_s(long imageHandle, int x, int y) {
  return as_float4(__nvvm_suld_2d_v4i32_clamp_s(imageHandle, x, y));
}
float4 __nvvm_suld_3d_v4f32_clamp_s(long imageHandle, int x, int y, int z) {
  return as_float4(__nvvm_suld_3d_v4i32_clamp_s(imageHandle, x, y, z));
}

// half
half __nvvm_suld_1d_f16_clamp_s(long imageHandle, int coord) {
  return as_half(__nvvm_suld_1d_i16_clamp_s(imageHandle, coord));
}
half __nvvm_suld_2d_f16_clamp_s(long imageHandle, int x, int y) {
  return as_half(__nvvm_suld_2d_i16_clamp_s(imageHandle, x, y));
}
half __nvvm_suld_3d_f16_clamp_s(long imageHandle, int x, int y, int z) {
  return as_half(__nvvm_suld_3d_i16_clamp_s(imageHandle, x, y, z));
}
half2 __nvvm_suld_1d_v2f16_clamp_s(long imageHandle, int coord) {
  return as_half2(__nvvm_suld_1d_v2i16_clamp_s(imageHandle, coord));
}
half2 __nvvm_suld_2d_v2f16_clamp_s(long imageHandle, int x, int y) {
  return as_half2(__nvvm_suld_2d_v2i16_clamp_s(imageHandle, x, y));
}
half2 __nvvm_suld_3d_v2f16_clamp_s(long imageHandle, int x, int y, int z) {
  return as_half2(__nvvm_suld_3d_v2i16_clamp_s(imageHandle, x, y, z));
}
half4 __nvvm_suld_1d_v4f16_clamp_s(long imageHandle, int coord) {
  return as_half4(__nvvm_suld_1d_v4i16_clamp_s(imageHandle, coord));
}
half4 __nvvm_suld_2d_v4f16_clamp_s(long imageHandle, int x, int y) {
  return as_half4(__nvvm_suld_2d_v4i16_clamp_s(imageHandle, x, y));
}
half4 __nvvm_suld_3d_v4f16_clamp_s(long imageHandle, int x, int y, int z) {
  return as_half4(__nvvm_suld_3d_v4i16_clamp_s(imageHandle, x, y, z));
}

// Generated funcs -- WRITES

// int
void __nvvm_sust_1d_i32_clamp_s(unsigned long, int,
                                int) __asm("llvm.nvvm.sust.b.1d.i32.clamp");
void __nvvm_sust_2d_i32_clamp_s(unsigned long, int, int,
                                int) __asm("llvm.nvvm.sust.b.2d.i32.clamp");
void __nvvm_sust_3d_i32_clamp_s(unsigned long, int, int, int,
                                int) __asm("llvm.nvvm.sust.b.3d.i32.clamp");
void __nvvm_sust_1d_v2i32_clamp_s(unsigned long, int, int,
                                  int) __asm("llvm.nvvm.sust.b.1d.v2i32.clamp");
void __nvvm_sust_2d_v2i32_clamp_s(unsigned long, int, int, int,
                                  int) __asm("llvm.nvvm.sust.b.2d.v2i32.clamp");
void __nvvm_sust_3d_v2i32_clamp_s(unsigned long, int, int, int, int,
                                  int) __asm("llvm.nvvm.sust.b.3d.v2i32.clamp");
void __nvvm_sust_1d_v4i32_clamp_s(unsigned long, int, int, int, int,
                                  int) __asm("llvm.nvvm.sust.b.1d.v4i32.clamp");
void __nvvm_sust_2d_v4i32_clamp_s(unsigned long, int, int, int, int, int,
                                  int) __asm("llvm.nvvm.sust.b.2d.v4i32.clamp");
void __nvvm_sust_3d_v4i32_clamp_s(unsigned long, int, int, int, int, int, int,
                                  int) __asm("llvm.nvvm.sust.b.3d.v4i32.clamp");

// unsigned int
void __nvvm_sust_1d_j32_clamp_s(unsigned long imageHandle, int coord,
                                unsigned int a) {
  return __nvvm_sust_1d_i32_clamp_s(imageHandle, coord, as_int(a));
}
void __nvvm_sust_2d_j32_clamp_s(unsigned long imageHandle, int x, int y,
                                unsigned int a) {
  return __nvvm_sust_2d_i32_clamp_s(imageHandle, x, y, as_int(a));
}
void __nvvm_sust_3d_j32_clamp_s(unsigned long imageHandle, int x, int y, int z,
                                unsigned int a) {
  return __nvvm_sust_3d_i32_clamp_s(imageHandle, x, y, z, as_int(a));
}
void __nvvm_sust_1d_v2j32_clamp_s(unsigned long imageHandle, int coord,
                                  unsigned int a, unsigned int b) {
  return __nvvm_sust_1d_v2i32_clamp_s(imageHandle, coord, as_int(a), as_int(b));
}
void __nvvm_sust_2d_v2j32_clamp_s(unsigned long imageHandle, int x, int y,
                                  unsigned int a, unsigned int b) {
  return __nvvm_sust_2d_v2i32_clamp_s(imageHandle, x, y, as_int(a), as_int(b));
}
void __nvvm_sust_3d_v2j32_clamp_s(unsigned long imageHandle, int x, int y,
                                  int z, unsigned int a, unsigned int b) {
  return __nvvm_sust_3d_v2i32_clamp_s(imageHandle, x, y, z, as_int(a),
                                      as_int(b));
}
void __nvvm_sust_1d_v4j32_clamp_s(unsigned long imageHandle, int coord,
                                  unsigned int a, unsigned int b,
                                  unsigned int c, unsigned int d) {
  return __nvvm_sust_1d_v4i32_clamp_s(imageHandle, coord, as_int(a), as_int(b),
                                      as_int(c), as_int(d));
}
void __nvvm_sust_2d_v4j32_clamp_s(unsigned long imageHandle, int x, int y,
                                  unsigned int a, unsigned int b,
                                  unsigned int c, unsigned int d) {
  return __nvvm_sust_2d_v4i32_clamp_s(imageHandle, x, y, as_int(a), as_int(b),
                                      as_int(c), as_int(d));
}
void __nvvm_sust_3d_v4j32_clamp_s(unsigned long imageHandle, int x, int y,
                                  int z, unsigned int a, unsigned int b,
                                  unsigned int c, unsigned int d) {
  return __nvvm_sust_3d_v4i32_clamp_s(imageHandle, x, y, z, as_int(a),
                                      as_int(b), as_int(c), as_int(d));
}

// short
void __nvvm_sust_1d_i16_clamp_s(unsigned long, int,
                                short) __asm("llvm.nvvm.sust.b.1d.i16.clamp");
void __nvvm_sust_2d_i16_clamp_s(unsigned long, int, int,
                                short) __asm("llvm.nvvm.sust.b.2d.i16.clamp");
void __nvvm_sust_3d_i16_clamp_s(unsigned long, int, int, int,
                                short) __asm("llvm.nvvm.sust.b.3d.i16.clamp");
void __nvvm_sust_1d_v2i16_clamp_s(unsigned long, int, short, short) __asm(
    "llvm.nvvm.sust.b.1d.v2i16.clamp");
void __nvvm_sust_2d_v2i16_clamp_s(unsigned long, int, int, short, short) __asm(
    "llvm.nvvm.sust.b.2d.v2i16.clamp");
void __nvvm_sust_3d_v2i16_clamp_s(
    unsigned long, int, int, int, short,
    short) __asm("llvm.nvvm.sust.b.3d.v2i16.clamp");
void __nvvm_sust_1d_v4i16_clamp_s(
    unsigned long, int, short, short, short,
    short) __asm("llvm.nvvm.sust.b.1d.v4i16.clamp");
void __nvvm_sust_2d_v4i16_clamp_s(
    unsigned long, int, int, short, short, short,
    short) __asm("llvm.nvvm.sust.b.2d.v4i16.clamp");
void __nvvm_sust_3d_v4i16_clamp_s(
    unsigned long, int, int, int, short, short, short,
    short) __asm("llvm.nvvm.sust.b.3d.v4i16.clamp");

// unsigned short
void __nvvm_sust_1d_t16_clamp_s(unsigned long imageHandle, int coord,
                                unsigned short a) {
  return __nvvm_sust_1d_i16_clamp_s(imageHandle, coord, as_ushort(a));
}
void __nvvm_sust_2d_t16_clamp_s(unsigned long imageHandle, int x, int y,
                                unsigned short a) {
  return __nvvm_sust_2d_i16_clamp_s(imageHandle, x, y, as_ushort(a));
}
void __nvvm_sust_3d_t16_clamp_s(unsigned long imageHandle, int x, int y, int z,
                                unsigned short a) {
  return __nvvm_sust_3d_i16_clamp_s(imageHandle, x, y, z, as_ushort(a));
}
void __nvvm_sust_1d_v2t16_clamp_s(unsigned long imageHandle, int coord,
                                  unsigned short a, unsigned short b) {
  return __nvvm_sust_1d_v2i16_clamp_s(imageHandle, coord, as_ushort(a),
                                      as_ushort(b));
}
void __nvvm_sust_2d_v2t16_clamp_s(unsigned long imageHandle, int x, int y,
                                  unsigned short a, unsigned short b) {
  return __nvvm_sust_2d_v2i16_clamp_s(imageHandle, x, y, as_ushort(a),
                                      as_ushort(b));
}
void __nvvm_sust_3d_v2t16_clamp_s(unsigned long imageHandle, int x, int y,
                                  int z, unsigned short a, unsigned short b) {
  return __nvvm_sust_3d_v2i16_clamp_s(imageHandle, x, y, z, as_ushort(a),
                                      as_ushort(b));
}
void __nvvm_sust_1d_v4t16_clamp_s(unsigned long imageHandle, int coord,
                                  unsigned short a, unsigned short b,
                                  unsigned short c, unsigned short d) {
  return __nvvm_sust_1d_v4i16_clamp_s(imageHandle, coord, as_ushort(a),
                                      as_ushort(b), as_short(c), as_ushort(d));
}
void __nvvm_sust_2d_v4t16_clamp_s(unsigned long imageHandle, int x, int y,
                                  unsigned short a, unsigned short b,
                                  unsigned short c, unsigned short d) {
  return __nvvm_sust_2d_v4i16_clamp_s(imageHandle, x, y, as_ushort(a),
                                      as_ushort(b), as_short(c), as_ushort(d));
}
void __nvvm_sust_3d_v4t16_clamp_s(unsigned long imageHandle, int x, int y,
                                  int z, unsigned short a, unsigned short b,
                                  unsigned short c, unsigned short d) {
  return __nvvm_sust_3d_v4i16_clamp_s(imageHandle, x, y, z, as_ushort(a),
                                      as_ushort(b), as_short(c), as_ushort(d));
}

// char  -- i8 intrinsic returns i16, requires helper
void __nvvm_sust_1d_i8_clamp_s_helper(unsigned long, int, short) __asm(
    "llvm.nvvm.sust.b.1d.i8.clamp");
void __nvvm_sust_1d_i8_clamp_s(unsigned long imageHandle, int coord, char c) {
  return __nvvm_sust_1d_i8_clamp_s_helper(imageHandle, coord, (short)c);
}

void __nvvm_sust_2d_i8_clamp_s_helper(unsigned long, int, int, short) __asm(
    "llvm.nvvm.sust.b.2d.i8.clamp");
void __nvvm_sust_2d_i8_clamp_s(unsigned long imageHandle, int x, int y,
                               char c) {
  return __nvvm_sust_2d_i8_clamp_s_helper(imageHandle, x, y, (short)c);
}

void __nvvm_sust_3d_i8_clamp_s_helper(
    unsigned long, int, int, int, short) __asm("llvm.nvvm.sust.b.3d.i8.clamp");
void __nvvm_sust_3d_i8_clamp_s(unsigned long imageHandle, int x, int y, int z,
                               char c) {
  return __nvvm_sust_3d_i8_clamp_s_helper(imageHandle, x, y, z, (short)c);
}

void __nvvm_sust_1d_v2i8_clamp_s_helper(unsigned long, int, short, short) __asm(
    "llvm.nvvm.sust.b.1d.v2i8.clamp");
void __nvvm_sust_1d_v2i8_clamp_s(unsigned long imageHandle, int coord, char a,
                                 char b) {
  return __nvvm_sust_1d_v2i8_clamp_s_helper(imageHandle, coord, (short)a,
                                            (short)b);
}

void __nvvm_sust_2d_v2i8_clamp_s_helper(
    unsigned long, int, int, short,
    short) __asm("llvm.nvvm.sust.b.2d.v2i8.clamp");
void __nvvm_sust_2d_v2i8_clamp_s(unsigned long imageHandle, int x, int y,
                                 char a, char b) {
  return __nvvm_sust_2d_v2i8_clamp_s_helper(imageHandle, x, y, (short)a,
                                            (short)b);
}

void __nvvm_sust_3d_v2i8_clamp_s_helper(
    unsigned long, int, int, int, short,
    short) __asm("llvm.nvvm.sust.b.3d.v2i8.clamp");
void __nvvm_sust_3d_v2i8_clamp_s(unsigned long imageHandle, int x, int y, int z,
                                 char a, char b) {
  return __nvvm_sust_3d_v2i8_clamp_s_helper(imageHandle, x, y, z, (short)a,
                                            (short)b);
}

void __nvvm_sust_1d_v4i8_clamp_s_helper(
    unsigned long, int, short, short, short,
    short) __asm("llvm.nvvm.sust.b.1d.v4i8.clamp");
void __nvvm_sust_1d_v4i8_clamp_s(unsigned long imageHandle, int coord, char a,
                                 char b, char c, char d) {
  return __nvvm_sust_1d_v4i8_clamp_s_helper(imageHandle, coord, (short)a,
                                            (short)b, (short)c, (short)d);
}

void __nvvm_sust_2d_v4i8_clamp_s_helper(
    unsigned long, int, int, short, short, short,
    short) __asm("llvm.nvvm.sust.b.2d.v4i8.clamp");
void __nvvm_sust_2d_v4i8_clamp_s(unsigned long imageHandle, int x, int y,
                                 char a, char b, char c, char d) {
  return __nvvm_sust_2d_v4i8_clamp_s_helper(imageHandle, x, y, (short)a,
                                            (short)b, (short)c, (short)d);
}

void __nvvm_sust_3d_v4i8_clamp_s_helper(
    unsigned long, int, int, int, short, short, short,
    short) __asm("llvm.nvvm.sust.b.3d.v4i8.clamp");
void __nvvm_sust_3d_v4i8_clamp_s(unsigned long imageHandle, int x, int y, int z,
                                 char a, char b, char c, char d) {
  return __nvvm_sust_3d_v4i8_clamp_s_helper(imageHandle, x, y, z, (short)a,
                                            (short)b, (short)c, (short)d);
}

// unsigned char  -- i8 intrinsic returns i16, requires helper
void __nvvm_sust_1d_h8_clamp_s(unsigned long imageHandle, int coord,
                               unsigned char c) {
  return __nvvm_sust_1d_i8_clamp_s_helper(imageHandle, coord,
                                          (unsigned short)c);
}
void __nvvm_sust_2d_h8_clamp_s(unsigned long imageHandle, int x, int y,
                               unsigned char c) {
  return __nvvm_sust_2d_i8_clamp_s_helper(imageHandle, x, y, (unsigned short)c);
}
void __nvvm_sust_3d_h8_clamp_s(unsigned long imageHandle, int x, int y, int z,
                               unsigned char c) {
  return __nvvm_sust_3d_i8_clamp_s_helper(imageHandle, x, y, z,
                                          (unsigned short)c);
}
void __nvvm_sust_1d_v2h8_clamp_s(unsigned long imageHandle, int coord, uchar a,
                                 uchar b) {
  return __nvvm_sust_1d_v2i8_clamp_s_helper(imageHandle, coord, (ushort)a,
                                            (ushort)b);
}
void __nvvm_sust_2d_v2h8_clamp_s(unsigned long imageHandle, int x, int y,
                                 uchar a, uchar b) {
  return __nvvm_sust_2d_v2i8_clamp_s_helper(imageHandle, x, y, (ushort)a,
                                            (ushort)b);
}
void __nvvm_sust_3d_v2h8_clamp_s(unsigned long imageHandle, int x, int y, int z,
                                 uchar a, uchar b) {
  return __nvvm_sust_3d_v2i8_clamp_s_helper(imageHandle, x, y, z, (ushort)a,
                                            (ushort)b);
}
void __nvvm_sust_1d_v4h8_clamp_s(unsigned long imageHandle, int coord, uchar a,
                                 uchar b, uchar c, uchar d) {
  return __nvvm_sust_1d_v4i8_clamp_s_helper(imageHandle, coord, (ushort)a,
                                            (ushort)b, (ushort)c, (ushort)d);
}
void __nvvm_sust_2d_v4h8_clamp_s(unsigned long imageHandle, int x, int y,
                                 uchar a, uchar b, uchar c, uchar d) {
  return __nvvm_sust_2d_v4i8_clamp_s_helper(imageHandle, x, y, (ushort)a,
                                            (ushort)b, (ushort)c, (ushort)d);
}
void __nvvm_sust_3d_v4h8_clamp_s(unsigned long imageHandle, int x, int y, int z,
                                 uchar a, uchar b, uchar c, uchar d) {
  return __nvvm_sust_3d_v4i8_clamp_s_helper(imageHandle, x, y, z, (ushort)a,
                                            (ushort)b, (ushort)c, (ushort)d);
}

// float
void __nvvm_sust_1d_f32_clamp_s(unsigned long imageHandle, int coord, float a) {
  return __nvvm_sust_1d_i32_clamp_s(imageHandle, coord, as_int(a));
}
void __nvvm_sust_2d_f32_clamp_s(unsigned long imageHandle, int x, int y,
                                float a) {
  return __nvvm_sust_2d_i32_clamp_s(imageHandle, x, y, as_int(a));
}
void __nvvm_sust_3d_f32_clamp_s(unsigned long imageHandle, int x, int y, int z,
                                float a) {
  return __nvvm_sust_3d_i32_clamp_s(imageHandle, x, y, z, as_int(a));
}
void __nvvm_sust_1d_v2f32_clamp_s(unsigned long imageHandle, int coord, float a,
                                  float b) {
  return __nvvm_sust_1d_v2i32_clamp_s(imageHandle, coord, as_int(a), as_int(b));
}
void __nvvm_sust_2d_v2f32_clamp_s(unsigned long imageHandle, int x, int y,
                                  float a, float b) {
  return __nvvm_sust_2d_v2i32_clamp_s(imageHandle, x, y, as_int(a), as_int(b));
}
void __nvvm_sust_3d_v2f32_clamp_s(unsigned long imageHandle, int x, int y,
                                  int z, float a, float b) {
  return __nvvm_sust_3d_v2i32_clamp_s(imageHandle, x, y, z, as_int(a),
                                      as_int(b));
}
void __nvvm_sust_1d_v4f32_clamp_s(unsigned long imageHandle, int coord, float a,
                                  float b, float c, float d) {
  return __nvvm_sust_1d_v4i32_clamp_s(imageHandle, coord, as_int(a), as_int(b),
                                      as_int(c), as_int(d));
}
void __nvvm_sust_2d_v4f32_clamp_s(unsigned long imageHandle, int x, int y,
                                  float a, float b, float c, float d) {
  return __nvvm_sust_2d_v4i32_clamp_s(imageHandle, x, y, as_int(a), as_int(b),
                                      as_int(c), as_int(d));
}
void __nvvm_sust_3d_v4f32_clamp_s(unsigned long imageHandle, int x, int y,
                                  int z, float a, float b, float c, float d) {
  return __nvvm_sust_3d_v4i32_clamp_s(imageHandle, x, y, z, as_int(a),
                                      as_int(b), as_int(c), as_int(d));
}

// half
void __nvvm_sust_1d_f16_clamp_s(unsigned long imageHandle, int coord, half a) {
  return __nvvm_sust_1d_i16_clamp_s(imageHandle, coord, as_short(a));
}
void __nvvm_sust_2d_f16_clamp_s(unsigned long imageHandle, int x, int y,
                                half a) {
  return __nvvm_sust_2d_i16_clamp_s(imageHandle, x, y, as_short(a));
}
void __nvvm_sust_3d_f16_clamp_s(unsigned long imageHandle, int x, int y, int z,
                                half a) {
  return __nvvm_sust_3d_i16_clamp_s(imageHandle, x, y, z, as_short(a));
}
void __nvvm_sust_1d_v2f16_clamp_s(unsigned long imageHandle, int coord, half a,
                                  half b) {
  return __nvvm_sust_1d_v2i16_clamp_s(imageHandle, coord, as_short(a),
                                      as_short(b));
}
void __nvvm_sust_2d_v2f16_clamp_s(unsigned long imageHandle, int x, int y,
                                  half a, half b) {
  return __nvvm_sust_2d_v2i16_clamp_s(imageHandle, x, y, as_short(a),
                                      as_short(b));
}
void __nvvm_sust_3d_v2f16_clamp_s(unsigned long imageHandle, int x, int y,
                                  int z, half a, half b) {
  return __nvvm_sust_3d_v2i16_clamp_s(imageHandle, x, y, z, as_short(a),
                                      as_short(b));
}
void __nvvm_sust_1d_v4f16_clamp_s(unsigned long imageHandle, int coord, half a,
                                  half b, half c, half d) {
  return __nvvm_sust_1d_v4i16_clamp_s(imageHandle, coord, as_short(a),
                                      as_short(b), as_short(c), as_short(d));
}
void __nvvm_sust_2d_v4f16_clamp_s(unsigned long imageHandle, int x, int y,
                                  half a, half b, half c, half d) {
  return __nvvm_sust_2d_v4i16_clamp_s(imageHandle, x, y, as_short(a),
                                      as_short(b), as_short(c), as_short(d));
}
void __nvvm_sust_3d_v4f16_clamp_s(unsigned long imageHandle, int x, int y,
                                  int z, half a, half b, half c, half d) {
  return __nvvm_sust_3d_v4i16_clamp_s(imageHandle, x, y, z, as_short(a),
                                      as_short(b), as_short(c), as_short(d));
}

#define _CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(                               \
    elem_t, dimension, elem_t_mangled, vec_size, coord_mangled, coord_input,   \
    ...)                                                                       \
  _CLC_DEF elem_t MANGLE_FUNC_IMG_HANDLE(                                      \
      17, __spirv_ImageRead, I##elem_t_mangled,                                \
      coord_mangled##ET_T0_T1_)(ulong imageHandle, coord_input) {              \
    return __nvvm_suld_##dimension##d_##vec_size##_clamp_s(imageHandle,        \
                                                           __VA_ARGS__);       \
  }

#define _CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(                              \
    elem_t, dimension, write_mangled, elem_t_mangled, vec_size, coord_input,   \
    ...)                                                                       \
  _CLC_DEF void MANGLE_FUNC_IMG_HANDLE(                                        \
      18, __spirv_ImageWrite, I, write_mangled##elem_t_mangled##EvT_T0_T1_)(   \
      ulong imageHandle, coord_input, elem_t c) {                              \
    __nvvm_sust_##dimension##d_##vec_size##_clamp_s(imageHandle, __VA_ARGS__); \
  }

// READS
// Int
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(int, 1, i, i32, i, int x, x * sizeof(int))
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(int, 2, i, i32, Dv2_i, int2 coord, coord.x * sizeof(int), coord.y)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(int, 3, i, i32, Dv4_i, int4 coord, coord.x * sizeof(int), coord.y, coord.z)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(int2, 1, Dv2_i, v2i32, i, int x, x * sizeof(int2))
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(int2, 2, Dv2_i, v2i32, S0_, int2 coord, coord.x * sizeof(int2), coord.y)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(int2, 3, Dv2_i, v2i32, Dv4_i, int4 coord, coord.x * sizeof(int2), coord.y, coord.z)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(int4, 1, Dv4_i, v4i32, i, int x, x * sizeof(int4))
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(int4, 2, Dv4_i, v4i32, Dv2_i, int2 coord, coord.x * sizeof(int4), coord.y)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(int4, 3, Dv4_i, v4i32, S0_, int4 coord, coord.x * sizeof(int4), coord.y, coord.z)

// Unsigned Int
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(unsigned int, 1, j, j32, i, int x, x * sizeof(unsigned int))
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(unsigned int, 2, j, j32, Dv2_i, int2 coord, coord.x * sizeof(unsigned int), coord.y)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(unsigned int, 3, j, j32, Dv4_i, int4 coord, coord.x * sizeof(unsigned int), coord.y, coord.z)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(uint2, 1, Dv2_j, v2j32, i, int x, x * sizeof(uint2))
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(uint2, 2, Dv2_j, v2j32, Dv2_i, int2 coord, coord.x * sizeof(uint2), coord.y)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(uint2, 3, Dv2_j, v2j32, Dv4_i, int4 coord, coord.x * sizeof(uint2), coord.y, coord.z)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(uint4, 1, Dv4_j, v4j32, i, int x, x * sizeof(uint4))
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(uint4, 2, Dv4_j, v4j32, Dv2_i, int2 coord, coord.x * sizeof(uint4), coord.y)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(uint4, 3, Dv4_j, v4j32, Dv4_i, int4 coord, coord.x * sizeof(uint4), coord.y, coord.z)

// Short
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(short, 1, s, i16, i, int x, x * sizeof(short))
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(short, 2, s, i16, Dv2_i, int2 coord, coord.x * sizeof(short), coord.y)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(short, 3, s, i16, Dv4_i, int4 coord, coord.x * sizeof(short), coord.y, coord.z)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(short2, 1, Dv2_s, v2i16, i, int x, x * sizeof(short2))
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(short2, 2, Dv2_s, v2i16, Dv2_i, int2 coord, coord.x * sizeof(short2), coord.y)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(short2, 3, Dv2_s, v2i16, Dv4_i, int4 coord, coord.x * sizeof(short2), coord.y, coord.z)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(short4, 1, Dv4_s, v4i16, i, int x, x * sizeof(short4))
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(short4, 2, Dv4_s, v4i16, Dv2_i, int2 coord, coord.x * sizeof(short4), coord.y)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(short4, 3, Dv4_s, v4i16, Dv4_i, int4 coord, coord.x * sizeof(short4), coord.y, coord.z)

// Unsigned Short
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(ushort, 1, t, t16, i, int x, x * sizeof(ushort))
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(ushort, 2, t, t16, Dv2_i, int2 coord, coord.x * sizeof(ushort), coord.y)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(ushort, 3, t, t16, Dv4_i, int4 coord, coord.x * sizeof(ushort), coord.y, coord.z)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(ushort2, 1, Dv2_t, v2t16, i, int x, x * sizeof(ushort2))
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(ushort2, 2, Dv2_t, v2t16, Dv2_i, int2 coord, coord.x * sizeof(ushort2), coord.y)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(ushort2, 3, Dv2_t, v2t16, Dv4_i, int4 coord, coord.x * sizeof(ushort2), coord.y, coord.z)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(ushort4, 1, Dv4_t, v4t16, i, int x, x * sizeof(ushort4))
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(ushort4, 2, Dv4_t, v4t16, Dv2_i, int2 coord, coord.x * sizeof(ushort4), coord.y)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(ushort4, 3, Dv4_t, v4t16, Dv4_i, int4 coord, coord.x * sizeof(ushort4), coord.y, coord.z)

// Char
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(char, 1, a, i8, i, int x, x * sizeof(char))
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(char, 2, a, i8, Dv2_i, int2 coord, coord.x * sizeof(char), coord.y)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(char, 3, a, i8, Dv4_i, int4 coord, coord.x * sizeof(char), coord.y, coord.z)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(char2, 1, Dv2_a, v2i8, i, int x, x * sizeof(char2))
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(char2, 2, Dv2_a, v2i8, Dv2_i, int2 coord, coord.x * sizeof(char2), coord.y)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(char2, 3, Dv2_a, v2i8, Dv4_i, int4 coord, coord.x * sizeof(char2), coord.y, coord.z)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(char4, 1, Dv4_a, v4i8, i, int x, x * sizeof(char4))
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(char4, 2, Dv4_a, v4i8, Dv2_i, int2 coord, coord.x * sizeof(char4), coord.y)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(char4, 3, Dv4_a, v4i8, Dv4_i, int4 coord, coord.x * sizeof(char4), coord.y, coord.z)

// Unsigned Char
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(uchar, 1, h, h8, i, int x, x * sizeof(uchar))
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(uchar, 2, h, h8, Dv2_i, int2 coord, coord.x * sizeof(uchar), coord.y)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(uchar, 3, h, h8, Dv4_i, int4 coord, coord.x * sizeof(uchar), coord.y, coord.z)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(uchar2, 1, Dv2_h, v2h8, i, int x, x * sizeof(uchar2))
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(uchar2, 2, Dv2_h, v2h8, Dv2_i, int2 coord, coord.x * sizeof(uchar2), coord.y)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(uchar2, 3, Dv2_h, v2h8, Dv4_i, int4 coord, coord.x * sizeof(uchar2), coord.y, coord.z)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(uchar4, 1, Dv4_h, v4h8, i, int x, x * sizeof(uchar4))
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(uchar4, 2, Dv4_h, v4h8, Dv2_i, int2 coord, coord.x * sizeof(uchar4), coord.y)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(uchar4, 3, Dv4_h, v4h8, Dv4_i, int4 coord, coord.x * sizeof(uchar4), coord.y, coord.z)

// Float
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(float, 1, f, f32, i, int x, x * sizeof(float))
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(float, 2, f, f32, Dv2_i, int2 coord, coord.x * sizeof(float), coord.y)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(float, 3, f, f32, Dv4_i, int4 coord, coord.x * sizeof(float), coord.y, coord.z)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(float2, 1, Dv2_f, v2f32, i, int x, x * sizeof(float2))
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(float2, 2, Dv2_f, v2f32, Dv2_i, int2 coord, coord.x * sizeof(float2), coord.y)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(float2, 3, Dv2_f, v2f32, Dv4_i, int4 coord, coord.x * sizeof(float2), coord.y, coord.z)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(float4, 1, Dv4_f, v4f32, i, int x, x * sizeof(float4))
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(float4, 2, Dv4_f, v4f32, Dv2_i, int2 coord, coord.x * sizeof(float4), coord.y)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(float4, 3, Dv4_f, v4f32, Dv4_i, int4 coord, coord.x * sizeof(float4), coord.y, coord.z)

// Half
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(half, 1, DF16_, f16, i, int x, x * sizeof(half))
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(half, 2, DF16_, f16, Dv2_i, int2 coord, coord.x * sizeof(half), coord.y)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(half, 3, DF16_, f16, Dv4_i, int4 coord, coord.x * sizeof(half), coord.y, coord.z)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(half2, 1, Dv2_DF16_, v2f16, i, int x, x * sizeof(half2))
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(half2, 2, Dv2_DF16_, v2f16, Dv2_i, int2 coord, coord.x * sizeof(half2), coord.y)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(half2, 3, Dv2_DF16_, v2f16, Dv4_i, int4 coord, coord.x * sizeof(half2), coord.y, coord.z)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(half4, 1, Dv4_DF16_, v4f16, i, int x, x * sizeof(half4))
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(half4, 2, Dv4_DF16_, v4f16, Dv2_i, int2 coord, coord.x * sizeof(half4), coord.y)
_CLC_DEFINE_IMAGE_BINDLESS_READ_BUILTIN(half4, 3, Dv4_DF16_, v4f16, Dv4_i, int4 coord, coord.x * sizeof(half4), coord.y, coord.z)

// WRITES
// Int
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(int, 1, i, i, i32, int x, x * sizeof(int), c)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(int, 2, Dv2_i, i, i32, int2 coord, coord.x * sizeof(int), coord.y, c)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(int, 3, Dv4_i, i, i32, int4 coord, coord.x * sizeof(int), coord.y, coord.z, c)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(int2, 1, i, Dv2_i, v2i32, int x, x * sizeof(int2), c.x, c.y)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(int2, 2, Dv2_i, S0_, v2i32, int2 coord, coord.x * sizeof(int2), coord.y, c.x, c.y)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(int2, 3, Dv4_i, Dv2_i, v2i32, int4 coord, coord.x * sizeof(int2), coord.y, coord.z, c.x, c.y)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(int4, 1, i, Dv4_i, v4i32, int x, x * sizeof(int4), c.x, c.y, c.z, c.w)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(int4, 2, Dv2_i, Dv4_i, v4i32, int2 coord, coord.x * sizeof(int4), coord.y, c.x, c.y, c.z, c.w)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(int4, 3, Dv4_i, S0_, v4i32, int4 coord, coord.x * sizeof(int4), coord.y, coord.z, c.x, c.y, c.z, c.w)

// Unsigned Int
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(unsigned int, 1, i, j, j32, int x, x * sizeof(unsigned int), c)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(unsigned int, 2, Dv2_i, j, j32, int2 coord, coord.x * sizeof(unsigned int), coord.y, c)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(unsigned int, 3, Dv4_i, j, j32, int4 coord, coord.x * sizeof(unsigned int), coord.y, coord.z, c)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(uint2, 1, i, Dv2_j, v2j32, int x, x * sizeof(uint2), c.x, c.y)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(uint2, 2, Dv2_i, Dv2_j, v2j32, int2 coord, coord.x * sizeof(uint2), coord.y, c.x, c.y)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(uint2, 3, Dv4_i, Dv2_j, v2j32, int4 coord, coord.x * sizeof(uint2), coord.y, coord.z, c.x, c.y)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(uint4, 1, i, Dv4_j, v4j32, int x, x * sizeof(uint4), c.x, c.y, c.z, c.w)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(uint4, 2, Dv2_i, Dv4_j, v4j32, int2 coord, coord.x * sizeof(uint4), coord.y, c.x, c.y, c.z, c.w)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(uint4, 3, Dv4_i, Dv4_j, v4j32, int4 coord, coord.x * sizeof(uint4), coord.y, coord.z, c.x, c.y, c.z, c.w)

// Short
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(short, 1, i, s, i16, int x, x * sizeof(short), c)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(short, 2, Dv2_i, s, i16, int2 coord, coord.x * sizeof(short), coord.y, c)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(short, 3, Dv4_i, s, i16, int4 coord, coord.x * sizeof(short), coord.y, coord.z, c)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(short2, 1, i, Dv2_s, v2i16, int x, x * sizeof(short2), c.x, c.y)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(short2, 2, Dv2_i, Dv2_s, v2i16, int2 coord, coord.x * sizeof(short2), coord.y, c.x, c.y)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(short2, 3, Dv4_i, Dv2_s, v2i16, int4 coord, coord.x * sizeof(short2), coord.y, coord.z, c.x, c.y)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(short4, 1, i, Dv4_s, v4i16, int x, x * sizeof(short4), c.x, c.y, c.z, c.w)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(short4, 2, Dv2_i, Dv4_s, v4i16, int2 coord, coord.x * sizeof(short4), coord.y, c.x, c.y, c.z, c.w)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(short4, 3, Dv4_i, Dv4_s, v4i16, int4 coord, coord.x * sizeof(short4), coord.y, coord.z, c.x, c.y, c.z, c.w)

// Unsigned Short
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(ushort, 1, i, t, t16, int x, x * sizeof(ushort), c)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(ushort, 2, Dv2_i, t, t16, int2 coord, coord.x * sizeof(ushort), coord.y, c)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(ushort, 3, Dv4_i, t, t16, int4 coord, coord.x * sizeof(ushort), coord.y, coord.z, c)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(ushort2, 1, i, Dv2_t, v2t16, int x, x * sizeof(ushort2), c.x, c.y)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(ushort2, 2, Dv2_i, Dv2_t, v2t16, int2 coord, coord.x * sizeof(ushort2), coord.y, c.x, c.y)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(ushort2, 3, Dv4_i, Dv2_t, v2t16, int4 coord, coord.x * sizeof(ushort2), coord.y, coord.z, c.x, c.y)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(ushort4, 1, i, Dv4_t, v4t16, int x, x * sizeof(ushort4), c.x, c.y, c.z, c.w)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(ushort4, 2, Dv2_i, Dv4_t, v4t16, int2 coord, coord.x * sizeof(ushort4), coord.y, c.x, c.y, c.z, c.w)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(ushort4, 3, Dv4_i, Dv4_t, v4t16, int4 coord, coord.x * sizeof(ushort4), coord.y, coord.z, c.x, c.y, c.z, c.w)

// Char
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(char, 1, i, a, i8, int x, x * sizeof(char), c)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(char, 2, Dv2_i, a, i8, int2 coord, coord.x * sizeof(char), coord.y, c)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(char, 3, Dv4_i, a, i8, int4 coord, coord.x * sizeof(char), coord.y, coord.z, c)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(char2, 1, i, Dv2_a, v2i8, int x, x * sizeof(char2), c.x, c.y)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(char2, 2, Dv2_i, Dv2_a, v2i8, int2 coord, coord.x * sizeof(char2), coord.y, c.x, c.y)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(char2, 3, Dv4_i, Dv2_a, v2i8, int4 coord, coord.x * sizeof(char2), coord.y, coord.z, c.x, c.y)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(char4, 1, i, Dv4_a, v4i8, int x, x * sizeof(char4), c.x, c.y, c.z, c.w)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(char4, 2, Dv2_i, Dv4_a, v4i8, int2 coord, coord.x * sizeof(char4), coord.y, c.x, c.y, c.z, c.w)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(char4, 3, Dv4_i, Dv4_a, v4i8, int4 coord, coord.x * sizeof(char4), coord.y, coord.z, c.x, c.y, c.z, c.w)

// Unsigned Char
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(uchar, 1, i, h, h8, int x, x * sizeof(uchar), c)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(uchar, 2, Dv2_i, h, h8, int2 coord, coord.x * sizeof(uchar), coord.y, c)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(uchar, 3, Dv4_i, h, h8, int4 coord, coord.x * sizeof(uchar), coord.y, coord.z, c)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(uchar2, 1, i, Dv2_h, v2h8, int x, x * sizeof(uchar2), c.x, c.y)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(uchar2, 2, Dv2_i, Dv2_h, v2h8, int2 coord, coord.x * sizeof(uchar2), coord.y, c.x, c.y)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(uchar2, 3, Dv4_i, Dv2_h, v2h8, int4 coord, coord.x * sizeof(uchar2), coord.y, coord.z, c.x, c.y)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(uchar4, 1, i, Dv4_h, v4h8, int x, x * sizeof(uchar4), c.x, c.y, c.z, c.w)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(uchar4, 2, Dv2_i, Dv4_h, v4h8, int2 coord, coord.x * sizeof(uchar4), coord.y, c.x, c.y, c.z, c.w)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(uchar4, 3, Dv4_i, Dv4_h, v4h8, int4 coord, coord.x * sizeof(uchar4), coord.y, coord.z, c.x, c.y, c.z, c.w)

// Float
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(float, 1, i, f, f32, int x, x * sizeof(float), c)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(float, 2, Dv2_i, f, f32, int2 coord, coord.x * sizeof(float), coord.y, c)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(float, 3, Dv4_i, f, f32, int4 coord, coord.x * sizeof(float), coord.y, coord.z, c)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(float2, 1, i, Dv2_f, v2f32, int x, x * sizeof(float2), c.x, c.y)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(float2, 2, Dv2_i, Dv2_f, v2f32, int2 coord, coord.x * sizeof(float2), coord.y, c.x, c.y)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(float2, 3, Dv4_i, Dv2_f, v2f32, int4 coord, coord.x * sizeof(float2), coord.y, coord.z, c.x, c.y)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(float4, 1, i, Dv4_f, v4f32, int x, x * sizeof(float4), c.x, c.y, c.z, c.w)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(float4, 2, Dv2_i, Dv4_f, v4f32, int2 coord, coord.x * sizeof(float4), coord.y, c.x, c.y, c.z, c.w)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(float4, 3, Dv4_i, Dv4_f, v4f32, int4 coord, coord.x * sizeof(float4), coord.y, coord.z, c.x, c.y, c.z, c.w)

// Half
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(half, 1, i, DF16_, f16, int x, x * sizeof(half), c)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(half, 2, Dv2_i, DF16_, f16, int2 coord, coord.x * sizeof(half), coord.y, c)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(half, 3, Dv4_i, DF16_, f16, int4 coord, coord.x * sizeof(half), coord.y, coord.z, c)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(half2, 1, i, Dv2_DF16_, v2f16, int x, x * sizeof(half2), c.x, c.y)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(half2, 2, Dv2_i, Dv2_DF16_, v2f16, int2 coord, coord.x * sizeof(half2), coord.y, c.x, c.y)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(half2, 3, Dv4_i, Dv2_DF16_, v2f16, int4 coord, coord.x * sizeof(half2), coord.y, coord.z, c.x, c.y)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(half4, 1, i, Dv4_DF16_, v4f16, int x, x * sizeof(half4), c.x, c.y, c.z, c.w)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(half4, 2, Dv2_i, Dv4_DF16_, v4f16, int2 coord, coord.x * sizeof(half4), coord.y, c.x, c.y, c.z, c.w)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(half4, 3, Dv4_i, Dv4_DF16_, v4f16, int4 coord, coord.x * sizeof(half4), coord.y, coord.z, c.x, c.y, c.z, c.w)


// <--- TEXTURES --->

// <--- Texture sampling (float coords) --->

// Int
int4 __nvvm_tex_1d_v4i32_f32(unsigned long,
                             float) __asm("__clc_llvm_nvvm_tex_1d_v4i32_f32");
int4 __nvvm_tex_2d_v4i32_f32(unsigned long, float,
                             float) __asm("__clc_llvm_nvvm_tex_2d_v4i32_f32");
int4 __nvvm_tex_3d_v4i32_f32(unsigned long, float, float,
                             float) __asm("__clc_llvm_nvvm_tex_3d_v4i32_f32");
int2 __nvvm_tex_1d_v2i32_f32(unsigned long imageHandle, float x) {
  int4 a = __nvvm_tex_1d_v4i32_f32(imageHandle, x);
  return cast_int4_to_int2(a);
}
int2 __nvvm_tex_2d_v2i32_f32(unsigned long imageHandle, float x, float y) {
  int4 a = __nvvm_tex_2d_v4i32_f32(imageHandle, x, y);
  return cast_int4_to_int2(a);
}
int2 __nvvm_tex_3d_v2i32_f32(unsigned long imageHandle, float x, float y,
                             float z) {
  int4 a = __nvvm_tex_3d_v4i32_f32(imageHandle, x, y, z);
  return cast_int4_to_int2(a);
}
int __nvvm_tex_1d_i32_f32(unsigned long imageHandle, float x) {
  return __nvvm_tex_1d_v4i32_f32(imageHandle, x)[0];
}
int __nvvm_tex_2d_i32_f32(unsigned long imageHandle, float x, float y) {
  return __nvvm_tex_2d_v4i32_f32(imageHandle, x, y)[0];
}
int __nvvm_tex_3d_i32_f32(unsigned long imageHandle, float x, float y,
                          float z) {
  return __nvvm_tex_3d_v4i32_f32(imageHandle, x, y, z)[0];
}

// Unsigned int
uint4 __nvvm_tex_1d_v4j32_f32(unsigned long,
                              float) __asm("__clc_llvm_nvvm_tex_1d_v4j32_f32");
uint4 __nvvm_tex_2d_v4j32_f32(unsigned long, float,
                              float) __asm("__clc_llvm_nvvm_tex_2d_v4j32_f32");
uint4 __nvvm_tex_3d_v4j32_f32(unsigned long, float, float,
                              float) __asm("__clc_llvm_nvvm_tex_3d_v4j32_f32");

uint2 __nvvm_tex_1d_v2j32_f32(unsigned long imageHandle, float x) {
  uint4 a = __nvvm_tex_1d_v4j32_f32(imageHandle, x);
  return cast_uint4_to_uint2(a);
}

uint2 __nvvm_tex_2d_v2j32_f32(unsigned long imageHandle, float x, float y) {
  uint4 a = __nvvm_tex_2d_v4j32_f32(imageHandle, x, y);
  return cast_uint4_to_uint2(a);
}

uint2 __nvvm_tex_3d_v2j32_f32(unsigned long imageHandle, float x, float y,
                              float z) {
  uint4 a = __nvvm_tex_3d_v4j32_f32(imageHandle, x, y, z);
  return cast_uint4_to_uint2(a);
}

uint __nvvm_tex_1d_j32_f32(unsigned long imageHandle, float x) {
  return __nvvm_tex_1d_v4j32_f32(imageHandle, x)[0];
}

uint __nvvm_tex_2d_j32_f32(unsigned long imageHandle, float x, float y) {
  return __nvvm_tex_2d_v4j32_f32(imageHandle, x, y)[0];
}

uint __nvvm_tex_3d_j32_f32(unsigned long imageHandle, float x, float y,
                           float z) {
  return __nvvm_tex_3d_v4j32_f32(imageHandle, x, y, z)[0];
}

// Short
short4 __nvvm_tex_1d_v4i16_f32(unsigned long imageHandle, float x) {
  int4 a = __nvvm_tex_1d_v4i32_f32(imageHandle, x);
  return cast_int4_to_short4(a);
}

short4 __nvvm_tex_2d_v4i16_f32(unsigned long imageHandle, float x, float y) {
  int4 a = __nvvm_tex_2d_v4i32_f32(imageHandle, x, y);
  return cast_int4_to_short4(a);
}

short4 __nvvm_tex_3d_v4i16_f32(unsigned long imageHandle, float x, float y,
                               float z) {
  int4 a = __nvvm_tex_3d_v4i32_f32(imageHandle, x, y, z);
  return cast_int4_to_short4(a);
}

short2 __nvvm_tex_1d_v2i16_f32(unsigned long imageHandle, float x) {
  int4 a = __nvvm_tex_1d_v4i32_f32(imageHandle, x);
  return cast_int4_to_short2(a);
}

short2 __nvvm_tex_2d_v2i16_f32(unsigned long imageHandle, float x, float y) {
  int4 a = __nvvm_tex_2d_v4i32_f32(imageHandle, x, y);
  return cast_int4_to_short2(a);
}

short2 __nvvm_tex_3d_v2i16_f32(unsigned long imageHandle, float x, float y,
                               float z) {
  int4 a = __nvvm_tex_3d_v4i32_f32(imageHandle, x, y, z);
  return cast_int4_to_short2(a);
}

short __nvvm_tex_1d_i16_f32(unsigned long imageHandle, float x) {
  return (short)(__nvvm_tex_1d_v4i32_f32(imageHandle, x)[0]);
}

short __nvvm_tex_2d_i16_f32(unsigned long imageHandle, float x, float y) {
  return (short)(__nvvm_tex_2d_v4i32_f32(imageHandle, x, y)[0]);
}

short __nvvm_tex_3d_i16_f32(unsigned long imageHandle, float x, float y,
                            float z) {
  return (short)(__nvvm_tex_3d_v4i32_f32(imageHandle, x, y, z)[0]);
}

// Unsigned Short
ushort4 __nvvm_tex_1d_v4t16_f32(unsigned long imageHandle, float x) {
  uint4 a = __nvvm_tex_1d_v4j32_f32(imageHandle, x);
  return cast_uint4_to_ushort4(a);
}

ushort4 __nvvm_tex_2d_v4t16_f32(unsigned long imageHandle, float x, float y) {
  uint4 a = __nvvm_tex_2d_v4j32_f32(imageHandle, x, y);
  return cast_uint4_to_ushort4(a);
}

ushort4 __nvvm_tex_3d_v4t16_f32(unsigned long imageHandle, float x, float y,
                                float z) {
  uint4 a = __nvvm_tex_3d_v4j32_f32(imageHandle, x, y, z);
  return cast_uint4_to_ushort4(a);
}

ushort2 __nvvm_tex_1d_v2t16_f32(unsigned long imageHandle, float x) {
  uint4 a = __nvvm_tex_1d_v4j32_f32(imageHandle, x);
  return cast_uint4_to_ushort2(a);
}

ushort2 __nvvm_tex_2d_v2t16_f32(unsigned long imageHandle, float x, float y) {
  uint4 a = __nvvm_tex_2d_v4j32_f32(imageHandle, x, y);
  return cast_uint4_to_ushort2(a);
}

ushort2 __nvvm_tex_3d_v2t16_f32(unsigned long imageHandle, float x, float y,
                                float z) {
  uint4 a = __nvvm_tex_3d_v4j32_f32(imageHandle, x, y, z);
  return cast_uint4_to_ushort2(a);
}

ushort __nvvm_tex_1d_t16_f32(unsigned long imageHandle, float x) {
  return (ushort)(__nvvm_tex_1d_v4j32_f32(imageHandle, x)[0]);
}

ushort __nvvm_tex_2d_t16_f32(unsigned long imageHandle, float x, float y) {
  return (ushort)(__nvvm_tex_2d_v4j32_f32(imageHandle, x, y)[0]);
}

ushort __nvvm_tex_3d_t16_f32(unsigned long imageHandle, float x, float y,
                             float z) {
  return (ushort)(__nvvm_tex_3d_v4j32_f32(imageHandle, x, y, z)[0]);
}

// Char
char4 __nvvm_tex_1d_v4i8_f32(unsigned long imageHandle, float x) {
  int4 a = __nvvm_tex_1d_v4i32_f32(imageHandle, x);
  return cast_int4_to_char4(a);
}

char4 __nvvm_tex_2d_v4i8_f32(unsigned long imageHandle, float x, float y) {
  int4 a = __nvvm_tex_2d_v4i32_f32(imageHandle, x, y);
  return cast_int4_to_char4(a);
}

char4 __nvvm_tex_3d_v4i8_f32(unsigned long imageHandle, float x, float y,
                             float z) {
  int4 a = __nvvm_tex_3d_v4i32_f32(imageHandle, x, y, z);
  return cast_int4_to_char4(a);
}

char2 __nvvm_tex_1d_v2i8_f32(unsigned long imageHandle, float x) {
  int4 a = __nvvm_tex_1d_v4i32_f32(imageHandle, x);
  return cast_int4_to_char2(a);
}

char2 __nvvm_tex_2d_v2i8_f32(unsigned long imageHandle, float x, float y) {
  int4 a = __nvvm_tex_2d_v4i32_f32(imageHandle, x, y);
  return cast_int4_to_char2(a);
}

char2 __nvvm_tex_3d_v2i8_f32(unsigned long imageHandle, float x, float y,
                             float z) {
  int4 a = __nvvm_tex_3d_v4i32_f32(imageHandle, x, y, z);
  return cast_int4_to_char2(a);
}

char __nvvm_tex_1d_i8_f32(unsigned long imageHandle, float x) {
  return (char)(__nvvm_tex_1d_v4i32_f32(imageHandle, x)[0]);
}

char __nvvm_tex_2d_i8_f32(unsigned long imageHandle, float x, float y) {
  return (char)(__nvvm_tex_2d_v4i32_f32(imageHandle, x, y)[0]);
}

char __nvvm_tex_3d_i8_f32(unsigned long imageHandle, float x, float y,
                          float z) {
  return (char)(__nvvm_tex_3d_v4i32_f32(imageHandle, x, y, z)[0]);
}

// Unsigned Char
uchar4 __nvvm_tex_1d_v4h8_f32(unsigned long imageHandle, float x) {
  uint4 a = __nvvm_tex_1d_v4j32_f32(imageHandle, x);
  return cast_uint4_to_uchar4(a);
}

uchar4 __nvvm_tex_2d_v4h8_f32(unsigned long imageHandle, float x, float y) {
  uint4 a = __nvvm_tex_2d_v4j32_f32(imageHandle, x, y);
  return cast_uint4_to_uchar4(a);
}

uchar4 __nvvm_tex_3d_v4h8_f32(unsigned long imageHandle, float x, float y,
                              float z) {
  uint4 a = __nvvm_tex_3d_v4j32_f32(imageHandle, x, y, z);
  return cast_uint4_to_uchar4(a);
}

uchar2 __nvvm_tex_1d_v2h8_f32(unsigned long imageHandle, float x) {
  uint4 a = __nvvm_tex_1d_v4j32_f32(imageHandle, x);
  return cast_uint4_to_uchar2(a);
}

uchar2 __nvvm_tex_2d_v2h8_f32(unsigned long imageHandle, float x, float y) {
  uint4 a = __nvvm_tex_2d_v4j32_f32(imageHandle, x, y);
  return cast_uint4_to_uchar2(a);
}

uchar2 __nvvm_tex_3d_v2h8_f32(unsigned long imageHandle, float x, float y,
                              float z) {
  uint4 a = __nvvm_tex_3d_v4j32_f32(imageHandle, x, y, z);
  return cast_uint4_to_uchar2(a);
}

uchar __nvvm_tex_1d_h8_f32(unsigned long imageHandle, float x) {
  return (uchar)(__nvvm_tex_1d_v4j32_f32(imageHandle, x)[0]);
}

uchar __nvvm_tex_2d_h8_f32(unsigned long imageHandle, float x, float y) {
  return (uchar)(__nvvm_tex_2d_v4j32_f32(imageHandle, x, y)[0]);
}

uchar __nvvm_tex_3d_h8_f32(unsigned long imageHandle, float x, float y,
                           float z) {
  return (uchar)(__nvvm_tex_3d_v4j32_f32(imageHandle, x, y, z)[0]);
}

// Float
float4 __nvvm_tex_1d_v4f32_f32(unsigned long,
                               float) __asm("__clc_llvm_nvvm_tex_1d_v4f32_f32");
float4 __nvvm_tex_2d_v4f32_f32(unsigned long, float,
                               float) __asm("__clc_llvm_nvvm_tex_2d_v4f32_f32");
float4 __nvvm_tex_3d_v4f32_f32(unsigned long, float, float,
                               float) __asm("__clc_llvm_nvvm_tex_3d_v4f32_f32");

float2 __nvvm_tex_1d_v2f32_f32(unsigned long imageHandle, float x) {
  float4 a = __nvvm_tex_1d_v4f32_f32(imageHandle, x);
  return cast_float4_to_float2(a);
}

float2 __nvvm_tex_2d_v2f32_f32(unsigned long imageHandle, float x, float y) {
  float4 a = __nvvm_tex_2d_v4f32_f32(imageHandle, x, y);
  return cast_float4_to_float2(a);
}

float2 __nvvm_tex_3d_v2f32_f32(unsigned long imageHandle, float x, float y,
                               float z) {
  float4 a = __nvvm_tex_3d_v4f32_f32(imageHandle, x, y, z);
  return cast_float4_to_float2(a);
}

float __nvvm_tex_1d_f32_f32(unsigned long imageHandle, float x) {
  return __nvvm_tex_1d_v4f32_f32(imageHandle, x)[0];
}

float __nvvm_tex_2d_f32_f32(unsigned long imageHandle, float x, float y) {
  return __nvvm_tex_2d_v4f32_f32(imageHandle, x, y)[0];
}

float __nvvm_tex_3d_f32_f32(unsigned long imageHandle, float x, float y,
                            float z) {
  return __nvvm_tex_3d_v4f32_f32(imageHandle, x, y, z)[0];
}

// Half
half4 __nvvm_tex_1d_v4f16_f32(unsigned long imageHandle, float x) {
  float4 a = __nvvm_tex_1d_v4f32_f32(imageHandle, x);
  return cast_float4_to_half4(a);
}

half4 __nvvm_tex_2d_v4f16_f32(unsigned long imageHandle, float x, float y) {
  float4 a = __nvvm_tex_2d_v4f32_f32(imageHandle, x, y);
  return cast_float4_to_half4(a);
}

half4 __nvvm_tex_3d_v4f16_f32(unsigned long imageHandle, float x, float y,
                              float z) {
  float4 a = __nvvm_tex_1d_v4f32_f32(imageHandle, x);
  return cast_float4_to_half4(a);
}

half2 __nvvm_tex_1d_v2f16_f32(unsigned long imageHandle, float x) {
  float4 a = __nvvm_tex_1d_v4f32_f32(imageHandle, x);
  return cast_float4_to_half2(a);
}

half2 __nvvm_tex_2d_v2f16_f32(unsigned long imageHandle, float x, float y) {
  float4 a = __nvvm_tex_2d_v4f32_f32(imageHandle, x, y);
  return cast_float4_to_half2(a);
}

half2 __nvvm_tex_3d_v2f16_f32(unsigned long imageHandle, float x, float y,
                              float z) {
  float4 a = __nvvm_tex_3d_v4f32_f32(imageHandle, x, y, z);
  return cast_float4_to_half2(a);
}

half __nvvm_tex_1d_f16_f32(unsigned long imageHandle, float x) {
  return (half)__nvvm_tex_1d_v4f32_f32(imageHandle, x)[0];
}

half __nvvm_tex_2d_f16_f32(unsigned long imageHandle, float x, float y) {
  return (half)__nvvm_tex_2d_v4f32_f32(imageHandle, x, y)[0];
}

half __nvvm_tex_3d_f16_f32(unsigned long imageHandle, float x, float y,
                           float z) {
  return (half)__nvvm_tex_3d_v4f32_f32(imageHandle, x, y, z)[0];
}

#define _CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(                        \
    elem_t, dimension, elem_t_mangled, vec_size, coord_mangled, coord_input,   \
    ...)                                                                       \
  _CLC_DEF elem_t MANGLE_FUNC_IMG_HANDLE(                                      \
      17, __spirv_ImageRead, I##elem_t_mangled,                                \
      coord_mangled##ET_T0_T1_)(ulong imageHandle, coord_input) {              \
    return __nvvm_tex_##dimension##d_##vec_size##_f32(imageHandle,             \
                                                      __VA_ARGS__);            \
  }

// Int
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(int, 1, i, i32, f, float x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(int, 2, i, i32, Dv2_f, float2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(int, 3, i, i32, Dv4_f, float4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(int2, 1, Dv2_i, v2i32, f, float x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(int2, 2, Dv2_i, v2i32, Dv2_f, float2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(int2, 3, Dv2_i, v2i32, Dv4_f, float4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(int4, 1, Dv4_i, v4i32, f, float x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(int4, 2, Dv4_i, v4i32, Dv2_f, float2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(int4, 3, Dv4_i, v4i32, Dv4_f, float4 coord, coord.x, coord.y, coord.z)

// Unsigned int
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(uint, 1, j, j32, f, float x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(uint, 2, j, j32, Dv2_f, float2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(uint, 3, j, j32, Dv4_f, float4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(uint2, 1, Dv2_j, v2j32, f, float x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(uint2, 2, Dv2_j, v2j32, Dv2_f, float2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(uint2, 3, Dv2_j, v2j32, Dv4_f, float4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(uint4, 1, Dv4_j, v4j32, f, float x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(uint4, 2, Dv4_j, v4j32, Dv2_f, float2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(uint4, 3, Dv4_j, v4j32, Dv4_f, float4 coord, coord.x, coord.y, coord.z)

// Short
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(short, 1, s, i16, f, float x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(short, 2, s, i16, Dv2_f, float2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(short, 3, s, i16, Dv4_f, float4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(short2, 1, Dv2_s, v2i16, f, float x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(short2, 2, Dv2_s, v2i16, Dv2_f, float2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(short2, 3, Dv2_s, v2i16, Dv4_f, float4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(short4, 1, Dv4_s, v4i16, f, float x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(short4, 2, Dv4_s, v4i16, Dv2_f, float2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(short4, 3, Dv4_s, v4i16, Dv4_f, float4 coord, coord.x, coord.y, coord.z)

// Unsigned short
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(ushort, 1, t, t16, f, float x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(ushort, 2, t, t16, Dv2_f, float2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(ushort, 3, t, t16, Dv4_f, float4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(ushort2, 1, Dv2_t, v2t16, f, float x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(ushort2, 2, Dv2_t, v2t16, Dv2_f, float2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(ushort2, 3, Dv2_t, v2t16, Dv4_f, float4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(ushort4, 1, Dv4_t, v4t16, f, float x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(ushort4, 2, Dv4_t, v4t16, Dv2_f, float2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(ushort4, 3, Dv4_t, v4t16, Dv4_f, float4 coord, coord.x, coord.y, coord.z)

// Char
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(char, 1, a, i8, f, float x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(char, 2, a, i8, Dv2_f, float2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(char, 3, a, i8, Dv4_f, float4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(char2, 1, Dv2_a, v2i8, f, float x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(char2, 2, Dv2_a, v2i8, Dv2_f, float2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(char2, 3, Dv2_a, v2i8, Dv4_f, float4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(char4, 1, Dv4_a, v4i8, f, float x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(char4, 2, Dv4_a, v4i8, Dv2_f, float2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(char4, 3, Dv4_a, v4i8, Dv4_f, float4 coord, coord.x, coord.y, coord.z)

// Unsigned Char
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(uchar, 1, h, h8, f, float x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(uchar, 2, h, h8, Dv2_f, float2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(uchar, 3, h, h8, Dv4_f, float4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(uchar2, 1, Dv2_h, v2h8, f, float x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(uchar2, 2, Dv2_h, v2h8, Dv2_f, float2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(uchar2, 3, Dv2_h, v2h8, Dv4_f, float4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(uchar4, 1, Dv4_h, v4h8, f, float x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(uchar4, 2, Dv4_h, v4h8, Dv2_f, float2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(uchar4, 3, Dv4_h, v4h8, Dv4_f, float4 coord, coord.x, coord.y, coord.z)

// Float
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(float, 1, f, f32, f, float x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(float, 2, f, f32, Dv2_f, float2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(float, 3, f, f32, Dv4_f, float4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(float2, 1, Dv2_f, v2f32, f, float x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(float2, 2, Dv2_f, v2f32, S0_, float2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(float2, 3, Dv2_f, v2f32, Dv4_f, float4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(float4, 1, Dv4_f, v4f32, f, float x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(float4, 2, Dv4_f, v4f32, Dv2_f, float2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(float4, 3, Dv4_f, v4f32, S0_, float4 coord, coord.x, coord.y, coord.z)

// Half
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(half, 1, DF16_, f16, f, float x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(half, 2, DF16_, f16, Dv2_f, float2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(half, 3, DF16_, f16, Dv4_f, float4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(half2, 1, Dv2_DF16_, v2f16, f, float x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(half2, 2, Dv2_DF16_, v2f16, Dv2_f, float2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(half2, 3, Dv2_DF16_, v2f16, Dv4_f, float4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(half4, 1, Dv4_DF16_, v4f16, f, float x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(half4, 2, Dv4_DF16_, v4f16, Dv2_f, float2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(half4, 3, Dv4_DF16_, v4f16, Dv4_f, float4 coord, coord.x, coord.y, coord.z)

// <--- Texture fetching (integer coords) --->

// Int
int4 __nvvm_tex_1d_v4i32_i32(unsigned long,
                             int) __asm("__clc_llvm_nvvm_tex_1d_v4i32_s32");
int4 __nvvm_tex_2d_v4i32_i32(unsigned long, int,
                             int) __asm("__clc_llvm_nvvm_tex_2d_v4i32_s32");
int4 __nvvm_tex_3d_v4i32_i32(unsigned long, int, int,
                             int) __asm("__clc_llvm_nvvm_tex_3d_v4i32_s32");

// Unsigned int
uint4 __nvvm_tex_1d_v4j32_i32(unsigned long,
                              int) __asm("__clc_llvm_nvvm_tex_1d_v4j32_s32");
uint4 __nvvm_tex_2d_v4j32_i32(unsigned long, int,
                              int) __asm("__clc_llvm_nvvm_tex_2d_v4j32_s32");
uint4 __nvvm_tex_3d_v4j32_i32(unsigned long, int, int,
                              int) __asm("__clc_llvm_nvvm_tex_3d_v4j32_s32");

// Float
float4 __nvvm_tex_1d_v4f32_i32(unsigned long,
                               int) __asm("__clc_llvm_nvvm_tex_1d_v4f32_s32");
float4 __nvvm_tex_2d_v4f32_i32(unsigned long, int,
                               int) __asm("__clc_llvm_nvvm_tex_2d_v4f32_s32");
float4 __nvvm_tex_3d_v4f32_i32(unsigned long, int, int,
                               int) __asm("__clc_llvm_nvvm_tex_3d_v4f32_s32");

// Macro to generate texture vec4 fetches
#define _CLC_DEFINE_BINDLESS_VEC4THUNK_TEXTURE_FETCH_BUILTIN(                  \
    elem_t, fetch_elem_t, dimension, vec_size, fetch_vec_size, coord_input,    \
    coord_parameter)                                                           \
  elem_t##4 __nvvm_tex_##dimension##d_##vec_size##_i32(                        \
      unsigned long imageHandle, coord_input) {                                \
    fetch_elem_t##4 a = __nvvm_tex_##dimension##d_##fetch_vec_size##_i32(      \
        imageHandle, coord_parameter);                                         \
    return cast_##fetch_elem_t##4_to_##elem_t##4(a);                           \
  }

// Macro to generate texture vec2 fetches
#define _CLC_DEFINE_BINDLESS_VEC2THUNK_TEXTURE_FETCH_BUILTIN(                  \
    elem_t, fetch_elem_t, dimension, vec_size, fetch_vec_size, coord_input,    \
    coord_parameter)                                                           \
  elem_t##2 __nvvm_tex_##dimension##d_##vec_size##_i32(                        \
      unsigned long imageHandle, coord_input) {                                \
    fetch_elem_t##4 a = __nvvm_tex_##dimension##d_##fetch_vec_size##_i32(      \
        imageHandle, coord_parameter);                                         \
    return cast_##fetch_elem_t##4_to_##elem_t##2(a);                           \
  }

// Macro to generate texture singular data type fetches
#define _CLC_DEFINE_BINDLESS_THUNK_TEXTURE_FETCH_BUILTIN(                      \
    elem_t, fetch_elem_t, dimension, vec_size, fetch_vec_size, coord_input,    \
    coord_parameter)                                                           \
  elem_t __nvvm_tex_##dimension##d_##vec_size##_i32(unsigned long imageHandle, \
                                                    coord_input) {             \
    return (elem_t)__nvvm_tex_##dimension##d_##fetch_vec_size##_i32(           \
        imageHandle, coord_parameter)[0];                                      \
  }

#define COORD_INPUT_1D int x
#define COORD_INPUT_2D int x, int y
#define COORD_INPUT_3D int x, int y, int z

#define COORD_PARAMS_1D x
#define COORD_PARAMS_2D x, y
#define COORD_PARAMS_3D x, y, z

#define _CLC_DEFINE_BINDLESS_VEC4THUNK_TEXTURE_FETCH_BUILTIN_ALL_DIMS(elem_t, fetch_elem_t, vec_size, fetch_vec_size) \
_CLC_DEFINE_BINDLESS_VEC4THUNK_TEXTURE_FETCH_BUILTIN(elem_t, fetch_elem_t, 1, vec_size, fetch_vec_size, COORD_INPUT_1D, COORD_PARAMS_1D) \
_CLC_DEFINE_BINDLESS_VEC4THUNK_TEXTURE_FETCH_BUILTIN(elem_t, fetch_elem_t, 2, vec_size, fetch_vec_size, COORD_INPUT_2D, COORD_PARAMS_2D) \
_CLC_DEFINE_BINDLESS_VEC4THUNK_TEXTURE_FETCH_BUILTIN(elem_t, fetch_elem_t, 3, vec_size, fetch_vec_size, COORD_INPUT_3D, COORD_PARAMS_3D)

_CLC_DEFINE_BINDLESS_VEC4THUNK_TEXTURE_FETCH_BUILTIN_ALL_DIMS(short, int, v4i16, v4i32)
_CLC_DEFINE_BINDLESS_VEC4THUNK_TEXTURE_FETCH_BUILTIN_ALL_DIMS(ushort, uint, v4t16, v4j32)
_CLC_DEFINE_BINDLESS_VEC4THUNK_TEXTURE_FETCH_BUILTIN_ALL_DIMS(char, int, v4i8, v4i32)
_CLC_DEFINE_BINDLESS_VEC4THUNK_TEXTURE_FETCH_BUILTIN_ALL_DIMS(uchar, uint, v4h8, v4j32)
_CLC_DEFINE_BINDLESS_VEC4THUNK_TEXTURE_FETCH_BUILTIN_ALL_DIMS(half, float, v4f16, v4f32)

#define _CLC_DEFINE_BINDLESS_VEC2THUNK_TEXTURE_FETCH_BUILTIN_ALL_DIMS(elem_t, fetch_elem_t, vec_size, fetch_vec_size) \
_CLC_DEFINE_BINDLESS_VEC2THUNK_TEXTURE_FETCH_BUILTIN(elem_t, fetch_elem_t, 1, vec_size, fetch_vec_size, COORD_INPUT_1D, COORD_PARAMS_1D) \
_CLC_DEFINE_BINDLESS_VEC2THUNK_TEXTURE_FETCH_BUILTIN(elem_t, fetch_elem_t, 2, vec_size, fetch_vec_size, COORD_INPUT_2D, COORD_PARAMS_2D) \
_CLC_DEFINE_BINDLESS_VEC2THUNK_TEXTURE_FETCH_BUILTIN(elem_t, fetch_elem_t, 3, vec_size, fetch_vec_size, COORD_INPUT_3D, COORD_PARAMS_3D)

_CLC_DEFINE_BINDLESS_VEC2THUNK_TEXTURE_FETCH_BUILTIN_ALL_DIMS(float, float, v2f32, v4f32)
_CLC_DEFINE_BINDLESS_VEC2THUNK_TEXTURE_FETCH_BUILTIN_ALL_DIMS(int, int, v2i32, v4i32)
_CLC_DEFINE_BINDLESS_VEC2THUNK_TEXTURE_FETCH_BUILTIN_ALL_DIMS(uint, uint, v2j32, v4j32)
_CLC_DEFINE_BINDLESS_VEC2THUNK_TEXTURE_FETCH_BUILTIN_ALL_DIMS(short, int, v2i16, v4i32)
_CLC_DEFINE_BINDLESS_VEC2THUNK_TEXTURE_FETCH_BUILTIN_ALL_DIMS(ushort, uint, v2t16, v4j32)
_CLC_DEFINE_BINDLESS_VEC2THUNK_TEXTURE_FETCH_BUILTIN_ALL_DIMS(char, int, v2i8, v4i32)
_CLC_DEFINE_BINDLESS_VEC2THUNK_TEXTURE_FETCH_BUILTIN_ALL_DIMS(uchar, uint, v2h8, v4j32)
_CLC_DEFINE_BINDLESS_VEC2THUNK_TEXTURE_FETCH_BUILTIN_ALL_DIMS(half, float, v2f16, v4f32)

#define _CLC_DEFINE_BINDLESS_THUNK_TEXTURE_FETCH_BUILTIN_ALL_DIMS(elem_t, fetch_elem_t, vec_size, fetch_vec_size) \
_CLC_DEFINE_BINDLESS_THUNK_TEXTURE_FETCH_BUILTIN(elem_t, fetch_elem_t, 1, vec_size, fetch_vec_size, COORD_INPUT_1D, COORD_PARAMS_1D) \
_CLC_DEFINE_BINDLESS_THUNK_TEXTURE_FETCH_BUILTIN(elem_t, fetch_elem_t, 2, vec_size, fetch_vec_size, COORD_INPUT_2D, COORD_PARAMS_2D) \
_CLC_DEFINE_BINDLESS_THUNK_TEXTURE_FETCH_BUILTIN(elem_t, fetch_elem_t, 3, vec_size, fetch_vec_size, COORD_INPUT_3D, COORD_PARAMS_3D)

_CLC_DEFINE_BINDLESS_THUNK_TEXTURE_FETCH_BUILTIN_ALL_DIMS(float, float, f32, v4f32)
_CLC_DEFINE_BINDLESS_THUNK_TEXTURE_FETCH_BUILTIN_ALL_DIMS(int, int, i32, v4i32)
_CLC_DEFINE_BINDLESS_THUNK_TEXTURE_FETCH_BUILTIN_ALL_DIMS(uint, uint, j32, v4j32)
_CLC_DEFINE_BINDLESS_THUNK_TEXTURE_FETCH_BUILTIN_ALL_DIMS(short, int, i16, v4i32)
_CLC_DEFINE_BINDLESS_THUNK_TEXTURE_FETCH_BUILTIN_ALL_DIMS(ushort, uint, t16, v4j32)
_CLC_DEFINE_BINDLESS_THUNK_TEXTURE_FETCH_BUILTIN_ALL_DIMS(char, int, i8, v4i32)
_CLC_DEFINE_BINDLESS_THUNK_TEXTURE_FETCH_BUILTIN_ALL_DIMS(uchar, uint, h8, v4j32)
_CLC_DEFINE_BINDLESS_THUNK_TEXTURE_FETCH_BUILTIN_ALL_DIMS(half, float, f16, v4f32)

#undef COORD_INPUT_1D
#undef COORD_INPUT_2D
#undef COORD_INPUT_3D

#undef COORD_PARAMS_1D
#undef COORD_PARAMS_2D
#undef COORD_PARAMS_3D

#undef _CLC_DEFINE_BINDLESS_THUNK_TEXTURE_FETCH_BUILTIN
#undef _CLC_DEFINE_BINDLESS_VEC2THUNK_TEXTURE_FETCH_BUILTIN
#undef _CLC_DEFINE_BINDLESS_VEC4THUNK_TEXTURE_FETCH_BUILTIN

#undef _CLC_DEFINE_BINDLESS_VEC4THUNK_TEXTURE_FETCH_BUILTIN_ALL_DIMS
#undef _CLC_DEFINE_BINDLESS_VEC2THUNK_TEXTURE_FETCH_BUILTIN_ALL_DIMS
#undef _CLC_DEFINE_BINDLESS_THUNK_TEXTURE_FETCH_BUILTIN_ALL_DIMS

#define _CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(                       \
    elem_t, dimension, elem_t_mangled, vec_size, coord_mangled, coord_input,   \
    ...)                                                                       \
  _CLC_DEF elem_t MANGLE_FUNC_IMG_HANDLE(                                      \
      18, __spirv_ImageFetch, I##elem_t_mangled,                               \
      coord_mangled##ET_T0_T1_)(ulong imageHandle, coord_input) {              \
    return __nvvm_tex_##dimension##d_##vec_size##_i32(imageHandle,             \
                                                      __VA_ARGS__);            \
  }

// Int
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(int, 1, i, i32, i, int x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(int, 2, i, i32, Dv2_i, int2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(int, 3, i, i32, Dv4_i, int4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(int2, 1, Dv2_i, v2i32, i, int x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(int2, 2, Dv2_i, v2i32, Dv2_i, int2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(int2, 3, Dv2_i, v2i32, Dv4_i, int4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(int4, 1, Dv4_i, v4i32, i, int x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(int4, 2, Dv4_i, v4i32, Dv2_i, int2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(int4, 3, Dv4_i, v4i32, Dv4_i, int4 coord, coord.x, coord.y, coord.z)

// Unsigned int
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(uint, 1, j, j32, i, int x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(uint, 2, j, j32, Dv2_i, int2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(uint, 3, j, j32, Dv4_i, int4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(uint2, 1, Dv2_j, v2j32, i, int x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(uint2, 2, Dv2_j, v2j32, Dv2_i, int2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(uint2, 3, Dv2_j, v2j32, Dv4_i, int4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(uint4, 1, Dv4_j, v4j32, i, int x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(uint4, 2, Dv4_j, v4j32, Dv2_i, int2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(uint4, 3, Dv4_j, v4j32, Dv4_i, int4 coord, coord.x, coord.y, coord.z)

// Short
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(short, 1, s, i16, i, int x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(short, 2, s, i16, Dv2_i, int2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(short, 3, s, i16, Dv4_i, int4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(short2, 1, Dv2_s, v2i16, i, int x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(short2, 2, Dv2_s, v2i16, Dv2_i, int2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(short2, 3, Dv2_s, v2i16, Dv4_i, int4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(short4, 1, Dv4_s, v4i16, i, int x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(short4, 2, Dv4_s, v4i16, Dv2_i, int2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(short4, 3, Dv4_s, v4i16, Dv4_i, int4 coord, coord.x, coord.y, coord.z)

// Unsigned short
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(ushort, 1, t, t16, i, int x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(ushort, 2, t, t16, Dv2_i, int2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(ushort, 3, t, t16, Dv4_i, int4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(ushort2, 1, Dv2_t, v2t16, i, int x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(ushort2, 2, Dv2_t, v2t16, Dv2_i, int2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(ushort2, 3, Dv2_t, v2t16, Dv4_i, int4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(ushort4, 1, Dv4_t, v4t16, i, int x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(ushort4, 2, Dv4_t, v4t16, Dv2_i, int2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(ushort4, 3, Dv4_t, v4t16, Dv4_i, int4 coord, coord.x, coord.y, coord.z)

// Char
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(char, 1, a, i8, i, int x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(char, 2, a, i8, Dv2_i, int2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(char, 3, a, i8, Dv4_i, int4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(char2, 1, Dv2_a, v2i8, i, int x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(char2, 2, Dv2_a, v2i8, Dv2_i, int2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(char2, 3, Dv2_a, v2i8, Dv4_i, int4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(char4, 1, Dv4_a, v4i8, i, int x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(char4, 2, Dv4_a, v4i8, Dv2_i, int2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(char4, 3, Dv4_a, v4i8, Dv4_i, int4 coord, coord.x, coord.y, coord.z)

// Unsigned Char
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(uchar, 1, h, h8, i, int x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(uchar, 2, h, h8, Dv2_i, int2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(uchar, 3, h, h8, Dv4_i, int4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(uchar2, 1, Dv2_h, v2h8, i, int x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(uchar2, 2, Dv2_h, v2h8, Dv2_i, int2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(uchar2, 3, Dv2_h, v2h8, Dv4_i, int4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(uchar4, 1, Dv4_h, v4h8, i, int x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(uchar4, 2, Dv4_h, v4h8, Dv2_i, int2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(uchar4, 3, Dv4_h, v4h8, Dv4_i, int4 coord, coord.x, coord.y, coord.z)

// Float
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(float, 1, f, f32, i, uint x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(float, 2, f, f32, Dv2_i, uint2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(float, 3, f, f32, Dv4_i, uint4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(float2, 1, Dv2_f, v2f32, i, uint x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(float2, 2, Dv2_f, v2f32, S0_, uint2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(float2, 3, Dv2_f, v2f32, Dv4_i, uint4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(float4, 1, Dv4_f, v4f32, i, uint x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(float4, 2, Dv4_f, v4f32, Dv2_i, uint2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(float4, 3, Dv4_f, v4f32, S0_, uint4 coord, coord.x, coord.y, coord.z)

// Half
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(half, 1, DF16_, f16, i, int x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(half, 2, DF16_, f16, Dv2_i, int2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(half, 3, DF16_, f16, Dv4_i, int4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(half2, 1, Dv2_DF16_, v2f16, i, int x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(half2, 2, Dv2_DF16_, v2f16, Dv2_i, int2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(half2, 3, Dv2_DF16_, v2f16, Dv4_i, int4 coord, coord.x, coord.y, coord.z)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(half4, 1, Dv4_DF16_, v4f16, i, int x, x)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(half4, 2, Dv4_DF16_, v4f16, Dv2_i, int2 coord, coord.x, coord.y)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_FETCH_BUILTIN(half4, 3, Dv4_DF16_, v4f16, Dv4_i, int4 coord, coord.x, coord.y, coord.z)


// <--- MIPMAP --->

// Define functions to call intrinsic
// Float
float4 __nvvm_tex_1d_level_v4f32_f32(unsigned long, float, float) __asm(
    "__clc_llvm_nvvm_tex_1d_level_v4f32_f32");
float4 __nvvm_tex_1d_grad_v4f32_f32(unsigned long, float, float, float) __asm(
    "__clc_llvm_nvvm_tex_1d_grad_v4f32_f32");
float4 __nvvm_tex_2d_level_v4f32_f32(unsigned long, float, float, float) __asm(
    "__clc_llvm_nvvm_tex_2d_level_v4f32_f32");
float4 __nvvm_tex_2d_grad_v4f32_f32(
    unsigned long, float, float, float, float, float,
    float) __asm("__clc_llvm_nvvm_tex_2d_grad_v4f32_f32");
float4 __nvvm_tex_3d_level_v4f32_f32(
    unsigned long, float, float, float,
    float) __asm("__clc_llvm_nvvm_tex_3d_level_v4f32_f32");
float4 __nvvm_tex_3d_grad_v4f32_f32(
    unsigned long, float, float, float, float, float, float, float, float,
    float) __asm("__clc_llvm_nvvm_tex_3d_grad_v4f32_f32");

// Int
int4 __nvvm_tex_1d_level_v4i32_f32(unsigned long, float, float) __asm(
    "__clc_llvm_nvvm_tex_1d_level_v4i32_f32");
int4 __nvvm_tex_1d_grad_v4i32_f32(unsigned long, float, float, float) __asm(
    "__clc_llvm_nvvm_tex_1d_grad_v4i32_f32");
int4 __nvvm_tex_2d_level_v4i32_f32(unsigned long, float, float, float) __asm(
    "__clc_llvm_nvvm_tex_2d_level_v4i32_f32");
int4 __nvvm_tex_2d_grad_v4i32_f32(
    unsigned long, float, float, float, float, float,
    float) __asm("__clc_llvm_nvvm_tex_2d_grad_v4i32_f32");
int4 __nvvm_tex_3d_level_v4i32_f32(
    unsigned long, float, float, float,
    float) __asm("__clc_llvm_nvvm_tex_3d_level_v4i32_f32");
int4 __nvvm_tex_3d_grad_v4i32_f32(
    unsigned long, float, float, float, float, float, float, float, float,
    float) __asm("__clc_llvm_nvvm_tex_3d_grad_v4i32_f32");

// UInt
uint4 __nvvm_tex_1d_level_v4j32_f32(unsigned long, float, float) __asm(
    "__clc_llvm_nvvm_tex_1d_level_v4j32_f32");
uint4 __nvvm_tex_1d_grad_v4j32_f32(unsigned long, float, float, float) __asm(
    "__clc_llvm_nvvm_tex_1d_grad_v4j32_f32");
uint4 __nvvm_tex_2d_level_v4j32_f32(unsigned long, float, float, float) __asm(
    "__clc_llvm_nvvm_tex_2d_level_v4j32_f32");
uint4 __nvvm_tex_2d_grad_v4j32_f32(
    unsigned long, float, float, float, float, float,
    float) __asm("__clc_llvm_nvvm_tex_2d_grad_v4j32_f32");
uint4 __nvvm_tex_3d_level_v4j32_f32(
    unsigned long, float, float, float,
    float) __asm("__clc_llvm_nvvm_tex_3d_level_v4j32_f32");
uint4 __nvvm_tex_3d_grad_v4j32_f32(
    unsigned long, float, float, float, float, float, float, float, float,
    float) __asm("__clc_llvm_nvvm_tex_3d_grad_v4j32_f32");

// Macro to generate mipmap vec4 fetches
#define _CLC_DEFINE_MIPMAP_BINDLESS_VEC4THUNK_READS_BUILTIN(                   \
    elem_t, fetch_elem_t, dimension, vec_size, fetch_vec_size, coord_input,    \
    coord_parameter, grad_input, ...)                                          \
  elem_t##4 __nvvm_tex_##dimension##d_level_##vec_size##_f32(                  \
      unsigned long imageHandle, coord_input, float level) {                   \
    fetch_elem_t##4 a =                                                        \
        __nvvm_tex_##dimension##d_level_##fetch_vec_size##_f32(                \
            imageHandle, coord_parameter, level);                              \
    return cast_##fetch_elem_t##4_to_##elem_t##4(a);                           \
  }                                                                            \
  elem_t##4 __nvvm_tex_##dimension##d_grad_##vec_size##_f32(                   \
      unsigned long imageHandle, coord_input, grad_input) {                    \
    fetch_elem_t##4 a = __nvvm_tex_##dimension##d_grad_##fetch_vec_size##_f32( \
        imageHandle, coord_parameter, __VA_ARGS__);                            \
    return cast_##fetch_elem_t##4_to_##elem_t##4(a);                           \
  }

#define COORD_INPUT_1D float x
#define COORD_INPUT_2D float x, float y
#define COORD_INPUT_3D float x, float y, float z

#define COORD_PARAMS_1D x
#define COORD_PARAMS_2D x, y
#define COORD_PARAMS_3D x, y, z

#define GRAD_INPUT_1D float dX, float dY
#define GRAD_INPUT_2D float dXx, float dXy, float dYx, float dYy
#define GRAD_INPUT_3D float dXx, float dXy, float dXz, float dYx, float dYy, float dYz

_CLC_DEFINE_MIPMAP_BINDLESS_VEC4THUNK_READS_BUILTIN(short, int, 1, v4i16, v4i32, COORD_INPUT_1D, COORD_PARAMS_1D, GRAD_INPUT_1D, dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC4THUNK_READS_BUILTIN(short, int, 2, v4i16, v4i32, COORD_INPUT_2D, COORD_PARAMS_2D, GRAD_INPUT_2D, dXx, dXy, dYx, dYy)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC4THUNK_READS_BUILTIN(short, int, 3, v4i16, v4i32, COORD_INPUT_3D, COORD_PARAMS_3D, GRAD_INPUT_3D, dXx, dXy, dXz, dYx, dYy, dYz)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC4THUNK_READS_BUILTIN(ushort, uint, 1, v4j16, v4j32, COORD_INPUT_1D, COORD_PARAMS_1D, GRAD_INPUT_1D, dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC4THUNK_READS_BUILTIN(ushort, uint, 2, v4j16, v4j32, COORD_INPUT_2D, COORD_PARAMS_2D, GRAD_INPUT_2D, dXx, dXy, dYx, dYy)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC4THUNK_READS_BUILTIN(ushort, uint, 3, v4j16, v4j32, COORD_INPUT_3D, COORD_PARAMS_3D, GRAD_INPUT_3D, dXx, dXy, dXz, dYx, dYy, dYz)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC4THUNK_READS_BUILTIN(char, int, 1, v4i8, v4i32, COORD_INPUT_1D, COORD_PARAMS_1D, GRAD_INPUT_1D, dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC4THUNK_READS_BUILTIN(char, int, 2, v4i8, v4i32, COORD_INPUT_2D, COORD_PARAMS_2D, GRAD_INPUT_2D, dXx, dXy, dYx, dYy)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC4THUNK_READS_BUILTIN(char, int, 3, v4i8, v4i32, COORD_INPUT_3D, COORD_PARAMS_3D, GRAD_INPUT_3D, dXx, dXy, dXz, dYx, dYy, dYz)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC4THUNK_READS_BUILTIN(uchar, uint, 1, v4j8, v4j32, COORD_INPUT_1D, COORD_PARAMS_1D, GRAD_INPUT_1D, dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC4THUNK_READS_BUILTIN(uchar, uint, 2, v4j8, v4j32, COORD_INPUT_2D, COORD_PARAMS_2D, GRAD_INPUT_2D, dXx, dXy, dYx, dYy)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC4THUNK_READS_BUILTIN(uchar, uint, 3, v4j8, v4j32, COORD_INPUT_3D, COORD_PARAMS_3D, GRAD_INPUT_3D, dXx, dXy, dXz, dYx, dYy, dYz)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC4THUNK_READS_BUILTIN(half, float, 1, v4f16, v4f32, COORD_INPUT_1D, COORD_PARAMS_1D, GRAD_INPUT_1D, dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC4THUNK_READS_BUILTIN(half, float, 2, v4f16, v4f32, COORD_INPUT_2D, COORD_PARAMS_2D, GRAD_INPUT_2D, dXx, dXy, dYx, dYy)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC4THUNK_READS_BUILTIN(half, float, 3, v4f16, v4f32, COORD_INPUT_3D, COORD_PARAMS_3D, GRAD_INPUT_3D, dXx, dXy, dXz, dYx, dYy, dYz)


// Macro to generate mipmap vec2 fetches
#define _CLC_DEFINE_MIPMAP_BINDLESS_VEC2THUNK_READS_BUILTIN(                   \
    elem_t, fetch_elem_t, dimension, vec_size, fetch_vec_size, coord_input,    \
    coord_parameter, grad_input, ...)                                          \
  elem_t##2 __nvvm_tex_##dimension##d_level_##vec_size##_f32(                  \
      unsigned long imageHandle, coord_input, float level) {                   \
    fetch_elem_t##4 a =                                                        \
        __nvvm_tex_##dimension##d_level_##fetch_vec_size##_f32(                \
            imageHandle, coord_parameter, level);                              \
    return cast_##fetch_elem_t##4_to_##elem_t##2(a);                           \
  }                                                                            \
  elem_t##2 __nvvm_tex_##dimension##d_grad_##vec_size##_f32(                   \
      unsigned long imageHandle, coord_input, grad_input) {                    \
    fetch_elem_t##4 a = __nvvm_tex_##dimension##d_grad_##fetch_vec_size##_f32( \
        imageHandle, coord_parameter, __VA_ARGS__);                            \
    return cast_##fetch_elem_t##4_to_##elem_t##2(a);                           \
  }

_CLC_DEFINE_MIPMAP_BINDLESS_VEC2THUNK_READS_BUILTIN(float, float, 1, v2f32, v4f32, COORD_INPUT_1D, COORD_PARAMS_1D, GRAD_INPUT_1D, dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC2THUNK_READS_BUILTIN(float, float, 2, v2f32, v4f32, COORD_INPUT_2D, COORD_PARAMS_2D, GRAD_INPUT_2D, dXx, dXy, dYx, dYy)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC2THUNK_READS_BUILTIN(float, float, 3, v2f32, v4f32, COORD_INPUT_3D, COORD_PARAMS_3D, GRAD_INPUT_3D, dXx, dXy, dXz, dYx, dYy, dYz)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC2THUNK_READS_BUILTIN(int, int, 1, v2i32, v4i32, COORD_INPUT_1D, COORD_PARAMS_1D, GRAD_INPUT_1D, dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC2THUNK_READS_BUILTIN(int, int, 2, v2i32, v4i32, COORD_INPUT_2D, COORD_PARAMS_2D, GRAD_INPUT_2D, dXx, dXy, dYx, dYy)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC2THUNK_READS_BUILTIN(int, int, 3, v2i32, v4i32, COORD_INPUT_3D, COORD_PARAMS_3D, GRAD_INPUT_3D, dXx, dXy, dXz, dYx, dYy, dYz)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC2THUNK_READS_BUILTIN(uint, uint, 1, v2j32, v4j32, COORD_INPUT_1D, COORD_PARAMS_1D, GRAD_INPUT_1D, dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC2THUNK_READS_BUILTIN(uint, uint, 2, v2j32, v4j32, COORD_INPUT_2D, COORD_PARAMS_2D, GRAD_INPUT_2D, dXx, dXy, dYx, dYy)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC2THUNK_READS_BUILTIN(uint, uint, 3, v2j32, v4j32, COORD_INPUT_3D, COORD_PARAMS_3D, GRAD_INPUT_3D, dXx, dXy, dXz, dYx, dYy, dYz)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC2THUNK_READS_BUILTIN(short, int, 1, v2i16, v4i32, COORD_INPUT_1D, COORD_PARAMS_1D, GRAD_INPUT_1D, dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC2THUNK_READS_BUILTIN(short, int, 2, v2i16, v4i32, COORD_INPUT_2D, COORD_PARAMS_2D, GRAD_INPUT_2D, dXx, dXy, dYx, dYy)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC2THUNK_READS_BUILTIN(short, int, 3, v2i16, v4i32, COORD_INPUT_3D, COORD_PARAMS_3D, GRAD_INPUT_3D, dXx, dXy, dXz, dYx, dYy, dYz)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC2THUNK_READS_BUILTIN(ushort, uint, 1, v2j16, v4j32, COORD_INPUT_1D, COORD_PARAMS_1D, GRAD_INPUT_1D, dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC2THUNK_READS_BUILTIN(ushort, uint, 2, v2j16, v4j32, COORD_INPUT_2D, COORD_PARAMS_2D, GRAD_INPUT_2D, dXx, dXy, dYx, dYy)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC2THUNK_READS_BUILTIN(ushort, uint, 3, v2j16, v4j32, COORD_INPUT_3D, COORD_PARAMS_3D, GRAD_INPUT_3D, dXx, dXy, dXz, dYx, dYy, dYz)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC2THUNK_READS_BUILTIN(char, int, 1, v2i8, v4i32, COORD_INPUT_1D, COORD_PARAMS_1D, GRAD_INPUT_1D, dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC2THUNK_READS_BUILTIN(char, int, 2, v2i8, v4i32, COORD_INPUT_2D, COORD_PARAMS_2D, GRAD_INPUT_2D, dXx, dXy, dYx, dYy)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC2THUNK_READS_BUILTIN(char, int, 3, v2i8, v4i32, COORD_INPUT_3D, COORD_PARAMS_3D, GRAD_INPUT_3D, dXx, dXy, dXz, dYx, dYy, dYz)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC2THUNK_READS_BUILTIN(uchar, uint, 1, v2j8, v4j32, COORD_INPUT_1D, COORD_PARAMS_1D, GRAD_INPUT_1D, dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC2THUNK_READS_BUILTIN(uchar, uint, 2, v2j8, v4j32, COORD_INPUT_2D, COORD_PARAMS_2D, GRAD_INPUT_2D, dXx, dXy, dYx, dYy)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC2THUNK_READS_BUILTIN(uchar, uint, 3, v2j8, v4j32, COORD_INPUT_3D, COORD_PARAMS_3D, GRAD_INPUT_3D, dXx, dXy, dXz, dYx, dYy, dYz)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC2THUNK_READS_BUILTIN(half, float, 1, v2f16, v4f32, COORD_INPUT_1D, COORD_PARAMS_1D, GRAD_INPUT_1D, dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC2THUNK_READS_BUILTIN(half, float, 2, v2f16, v4f32, COORD_INPUT_2D, COORD_PARAMS_2D, GRAD_INPUT_2D, dXx, dXy, dYx, dYy)
_CLC_DEFINE_MIPMAP_BINDLESS_VEC2THUNK_READS_BUILTIN(half, float, 3, v2f16, v4f32, COORD_INPUT_3D, COORD_PARAMS_3D, GRAD_INPUT_3D, dXx, dXy, dXz, dYx, dYy, dYz)


#undef _CLC_DEFINE_MIPMAP_BINDLESS_VEC2THUNK_READS_BUILTIN

// Macro to generate mipmap singular data type fetches
#define _CLC_DEFINE_MIPMAP_BINDLESS_THUNK_READS_BUILTIN(                       \
    elem_t, dimension, vec_size, fetch_vec_size, coord_input, coord_parameter, \
    grad_input, ...)                                                           \
  elem_t __nvvm_tex_##dimension##d_level_##vec_size##_f32(                     \
      unsigned long imageHandle, coord_input, float level) {                   \
    return (elem_t)__nvvm_tex_##dimension##d_level_##fetch_vec_size##_f32(     \
        imageHandle, coord_parameter, level)[0];                               \
  }                                                                            \
  elem_t __nvvm_tex_##dimension##d_grad_##vec_size##_f32(                      \
      unsigned long imageHandle, coord_input, grad_input) {                    \
    return (elem_t)__nvvm_tex_##dimension##d_grad_##fetch_vec_size##_f32(      \
        imageHandle, coord_parameter, __VA_ARGS__)[0];                         \
  }

_CLC_DEFINE_MIPMAP_BINDLESS_THUNK_READS_BUILTIN(float, 1, f32, v4f32, COORD_INPUT_1D, COORD_PARAMS_1D, GRAD_INPUT_1D, dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_THUNK_READS_BUILTIN(float, 2, f32, v4f32, COORD_INPUT_2D, COORD_PARAMS_2D, GRAD_INPUT_2D, dXx, dXy, dYx, dYy)
_CLC_DEFINE_MIPMAP_BINDLESS_THUNK_READS_BUILTIN(float, 3, f32, v4f32, COORD_INPUT_3D, COORD_PARAMS_3D, GRAD_INPUT_3D, dXx, dXy, dXz, dYx, dYy, dYz)
_CLC_DEFINE_MIPMAP_BINDLESS_THUNK_READS_BUILTIN(int, 1, i32, v4i32, COORD_INPUT_1D, COORD_PARAMS_1D, GRAD_INPUT_1D, dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_THUNK_READS_BUILTIN(int, 2, i32, v4i32, COORD_INPUT_2D, COORD_PARAMS_2D, GRAD_INPUT_2D, dXx, dXy, dYx, dYy)
_CLC_DEFINE_MIPMAP_BINDLESS_THUNK_READS_BUILTIN(int, 3, i32, v4i32, COORD_INPUT_3D, COORD_PARAMS_3D, GRAD_INPUT_3D, dXx, dXy, dXz, dYx, dYy, dYz)
_CLC_DEFINE_MIPMAP_BINDLESS_THUNK_READS_BUILTIN(uint, 1, j32, v4j32, COORD_INPUT_1D, COORD_PARAMS_1D, GRAD_INPUT_1D, dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_THUNK_READS_BUILTIN(uint, 2, j32, v4j32, COORD_INPUT_2D, COORD_PARAMS_2D, GRAD_INPUT_2D, dXx, dXy, dYx, dYy)
_CLC_DEFINE_MIPMAP_BINDLESS_THUNK_READS_BUILTIN(uint, 3, j32, v4j32, COORD_INPUT_3D, COORD_PARAMS_3D, GRAD_INPUT_3D, dXx, dXy, dXz, dYx, dYy, dYz)
_CLC_DEFINE_MIPMAP_BINDLESS_THUNK_READS_BUILTIN(short, 1, i16, v4i32, COORD_INPUT_1D, COORD_PARAMS_1D, GRAD_INPUT_1D, dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_THUNK_READS_BUILTIN(short, 2, i16, v4i32, COORD_INPUT_2D, COORD_PARAMS_2D, GRAD_INPUT_2D, dXx, dXy, dYx, dYy)
_CLC_DEFINE_MIPMAP_BINDLESS_THUNK_READS_BUILTIN(short, 3, i16, v4i32, COORD_INPUT_3D, COORD_PARAMS_3D, GRAD_INPUT_3D, dXx, dXy, dXz, dYx, dYy, dYz)
_CLC_DEFINE_MIPMAP_BINDLESS_THUNK_READS_BUILTIN(ushort, 1, j16, v4j32, COORD_INPUT_1D, COORD_PARAMS_1D, GRAD_INPUT_1D, dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_THUNK_READS_BUILTIN(ushort, 2, j16, v4j32, COORD_INPUT_2D, COORD_PARAMS_2D, GRAD_INPUT_2D, dXx, dXy, dYx, dYy)
_CLC_DEFINE_MIPMAP_BINDLESS_THUNK_READS_BUILTIN(ushort, 3, j16, v4j32, COORD_INPUT_3D, COORD_PARAMS_3D, GRAD_INPUT_3D, dXx, dXy, dXz, dYx, dYy, dYz)
_CLC_DEFINE_MIPMAP_BINDLESS_THUNK_READS_BUILTIN(char, 1, i8, v4i32, COORD_INPUT_1D, COORD_PARAMS_1D, GRAD_INPUT_1D, dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_THUNK_READS_BUILTIN(char, 2, i8, v4i32, COORD_INPUT_2D, COORD_PARAMS_2D, GRAD_INPUT_2D, dXx, dXy, dYx, dYy)
_CLC_DEFINE_MIPMAP_BINDLESS_THUNK_READS_BUILTIN(char, 3, i8, v4i32, COORD_INPUT_3D, COORD_PARAMS_3D, GRAD_INPUT_3D, dXx, dXy, dXz, dYx, dYy, dYz)
_CLC_DEFINE_MIPMAP_BINDLESS_THUNK_READS_BUILTIN(uchar, 1, j8, v4j32, COORD_INPUT_1D, COORD_PARAMS_1D, GRAD_INPUT_1D, dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_THUNK_READS_BUILTIN(uchar, 2, j8, v4j32, COORD_INPUT_2D, COORD_PARAMS_2D, GRAD_INPUT_2D, dXx, dXy, dYx, dYy)
_CLC_DEFINE_MIPMAP_BINDLESS_THUNK_READS_BUILTIN(uchar, 3, j8, v4j32, COORD_INPUT_3D, COORD_PARAMS_3D, GRAD_INPUT_3D, dXx, dXy, dXz, dYx, dYy, dYz)
_CLC_DEFINE_MIPMAP_BINDLESS_THUNK_READS_BUILTIN(half, 1, f16, v4j32, COORD_INPUT_1D, COORD_PARAMS_1D, GRAD_INPUT_1D, dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_THUNK_READS_BUILTIN(half, 2, f16, v4f32, COORD_INPUT_2D, COORD_PARAMS_2D, GRAD_INPUT_2D, dXx, dXy, dYx, dYy)
_CLC_DEFINE_MIPMAP_BINDLESS_THUNK_READS_BUILTIN(half, 3, f16, v4f32, COORD_INPUT_3D, COORD_PARAMS_3D, GRAD_INPUT_3D, dXx, dXy, dXz, dYx, dYy, dYz)


#undef _CLC_DEFINE_MIPMAP_BINDLESS_THUNK_READS_BUILTIN

#undef COORD_INPUT_1D
#undef COORD_INPUT_2D
#undef COORD_INPUT_3D
#undef COORD_PARAMS_1D
#undef COORD_PARAMS_2D
#undef COORD_PARAMS_3D
#undef GRAD_INPUT_1D
#undef GRAD_INPUT_2D
#undef GRAD_INPUT_3D

// Macro to generate the mangled names for mipmap fetches
#define _CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(                             \
    elem_t, dimension, elem_t_mangled, vec_size, coord_mangled, coord_input,   \
    coord_parameter, grad_mangled, grad_input, ...)                            \
  _CLC_DEF elem_t MANGLE_FUNC_IMG_HANDLE(                                      \
      30, __spirv_ImageSampleExplicitLod, I,                                   \
      elem_t_mangled##coord_mangled##ET0_T_T1_if)(                             \
      ulong imageHandle, coord_input, int type, float level) {                 \
    return __nvvm_tex_##dimension##d_level_##vec_size##_f32(                   \
        imageHandle, coord_parameter, level);                                  \
  }                                                                            \
  _CLC_DEF elem_t MANGLE_FUNC_IMG_HANDLE(                                      \
      30, __spirv_ImageSampleExplicitLod, I,                                   \
      elem_t_mangled##coord_mangled##ET0_T_T1_i##grad_mangled)(                \
      ulong imageHandle, coord_input, int type, float##grad_input dX,          \
      float##grad_input dY) {                                                  \
    return __nvvm_tex_##dimension##d_grad_##vec_size##_f32(                    \
        imageHandle, coord_parameter, __VA_ARGS__);                            \
  }

#define COORD_PARAMS_1D coord
#define COORD_PARAMS_2D coord.x, coord.y
#define COORD_PARAMS_3D coord.x, coord.y, coord.z

// Int
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(int, 1, i, i32, f, float coord, COORD_PARAMS_1D, S2_S2_, , dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(int, 2, i, i32, Dv2_f, float2 coord, COORD_PARAMS_2D, S3_S3_, 2, dX.x, dX.y, dY.x, dY.y)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(int, 3, i, i32, Dv4_f, float4 coord, COORD_PARAMS_3D, S3_S3_, 4, dX.x, dX.y, dX.z, dY.x, dY.y, dY.z)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(int2, 1, Dv2_i, v2i32, f, float coord, COORD_PARAMS_1D, S3_S3_, , dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(int2, 2, Dv2_i, v2i32, Dv2_f, float2 coord, COORD_PARAMS_2D, S4_S4_, 2, dX.x, dX.y, dY.x, dY.y)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(int2, 3, Dv2_i, v2i32, Dv4_f, float4 coord, COORD_PARAMS_3D, S4_S4_, 4, dX.x, dX.y, dX.z, dY.x, dY.y, dY.z)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(int4, 1, Dv4_i, v4i32, f, float coord, coord, S3_S3_, , dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(int4, 2, Dv4_i, v4i32, Dv2_f, float2 coord, COORD_PARAMS_2D, S4_S4_, 2, dX.x, dX.y, dY.x, dY.y)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(int4, 3, Dv4_i, v4i32, Dv4_f, float4 coord, COORD_PARAMS_3D, S4_S4_, 4, dX.x, dX.y, dX.z, dY.x, dY.y, dY.z)

// UInt
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(uint, 1, j, j32, f, float coord, COORD_PARAMS_1D, S2_S2_, , dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(uint, 2, j, j32, Dv2_f, float2 coord, COORD_PARAMS_2D, S3_S3_, 2, dX.x, dX.y, dY.x, dY.y)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(uint, 3, j, j32, Dv4_f, float4 coord, COORD_PARAMS_3D, S3_S3_, 4, dX.x, dX.y, dX.z, dY.x, dY.y, dY.z)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(uint2, 1, Dv2_j, v2j32, f, float coord, COORD_PARAMS_1D, S3_S3_, , dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(uint2, 2, Dv2_j, v2j32, Dv2_f, float2 coord, COORD_PARAMS_2D, S4_S4_, 2, dX.x, dX.y, dY.x, dY.y)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(uint2, 3, Dv2_j, v2j32, Dv4_f, float4 coord, COORD_PARAMS_3D, S4_S4_, 4, dX.x, dX.y, dX.z, dY.x, dY.y, dY.z)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(uint4, 1, Dv4_j, v4j32, f, float coord, COORD_PARAMS_1D, S3_S3_, , dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(uint4, 2, Dv4_j, v4j32, Dv2_f, float2 coord, COORD_PARAMS_2D, S4_S4_, 2, dX.x, dX.y, dY.x, dY.y)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(uint4, 3, Dv4_j, v4j32, Dv4_f, float4 coord, COORD_PARAMS_3D, S4_S4_, 4, dX.x, dX.y, dX.z, dY.x, dY.y, dY.z)

// Float
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(float, 1, f, f32, f, float coord, COORD_PARAMS_1D, S2_S2_, , dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(float, 2, f, f32, Dv2_f, float2 coord, COORD_PARAMS_2D, S3_S3_, 2, dX.x, dX.y, dY.x, dY.y)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(float, 3, f, f32, Dv4_f, float4 coord, COORD_PARAMS_3D, S3_S3_, 4, dX.x, dX.y, dX.z, dY.x, dY.y, dY.z)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(float2, 1, Dv2_f, v2f32, f, float coord, COORD_PARAMS_1D, S3_S3_, , dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(float2, 2, Dv2_f, v2f32, S0_, float2 coord, COORD_PARAMS_2D, S3_S3_, 2, dX.x, dX.y, dY.x, dY.y)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(float2, 3, Dv2_f, v2f32, Dv4_f, float4 coord, COORD_PARAMS_3D, S4_S4_, 4, dX.x, dX.y, dX.z, dY.x, dY.y, dY.z)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(float4, 1, Dv4_f, v4f32, f, float coord, COORD_PARAMS_1D, S3_S3_, , dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(float4, 2, Dv4_f, v4f32, Dv2_f, float2 coord, COORD_PARAMS_2D, S4_S4_, 2, dX.x, dX.y, dY.x, dY.y)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(float4, 3, Dv4_f, v4f32, S0_, float4 coord, COORD_PARAMS_3D, S3_S3_, 4, dX.x, dX.y, dX.z, dY.x, dY.y, dY.z)

// Short
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(short, 1, s, i16, f, float coord, COORD_PARAMS_1D, S2_S2_, , dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(short, 2, s, i16, Dv2_f, float2 coord, COORD_PARAMS_2D, S3_S3_, 2, dX.x, dX.y, dY.x, dY.y)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(short, 3, s, i16, Dv4_f, float4 coord, COORD_PARAMS_3D, S3_S3_, 4, dX.x, dX.y, dX.z, dY.x, dY.y, dY.z)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(short2, 1, Dv2_s, v2i16, f, float coord, COORD_PARAMS_1D, S3_S3_, , dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(short2, 2, Dv2_s, v2i16, Dv2_f, float2 coord, COORD_PARAMS_2D, S4_S4_, 2, dX.x, dX.y, dY.x, dY.y)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(short2, 3, Dv2_s, v2i16, Dv4_f, float4 coord, COORD_PARAMS_3D, S4_S4_, 4, dX.x, dX.y, dX.z, dY.x, dY.y, dY.z)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(short4, 1, Dv4_s, v4i16, f, float coord, COORD_PARAMS_1D, S3_S3_, , dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(short4, 2, Dv4_s, v4i16, Dv2_f, float2 coord, COORD_PARAMS_2D, S4_S4_, 2, dX.x, dX.y, dY.x, dY.y)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(short4, 3, Dv4_s, v4i16, Dv4_f, float4 coord, COORD_PARAMS_3D, S4_S4_, 4, dX.x, dX.y, dX.z, dY.x, dY.y, dY.z)

// Unsigned Short
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(ushort, 1, t, j16, f, float coord, COORD_PARAMS_1D, S2_S2_, , dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(ushort, 2, t, j16, Dv2_f, float2 coord, COORD_PARAMS_2D, S3_S3_, 2, dX.x, dX.y, dY.x, dY.y)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(ushort, 3, t, j16, Dv4_f, float4 coord, COORD_PARAMS_3D, S3_S3_, 4, dX.x, dX.y, dX.z, dY.x, dY.y, dY.z)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(ushort2, 1, Dv2_t, v2j16, f, float coord, COORD_PARAMS_1D, S3_S3_, , dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(ushort2, 2, Dv2_t, v2j16, Dv2_f, float2 coord, COORD_PARAMS_2D, S4_S4_, 2, dX.x, dX.y, dY.x, dY.y)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(ushort2, 3, Dv2_t, v2j16, Dv4_f, float4 coord, COORD_PARAMS_3D, S4_S4_, 4, dX.x, dX.y, dX.z, dY.x, dY.y, dY.z)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(ushort4, 1, Dv4_t, v4j16, f, float coord, COORD_PARAMS_1D, S3_S3_, , dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(ushort4, 2, Dv4_t, v4j16, Dv2_f, float2 coord, COORD_PARAMS_2D, S4_S4_, 2, dX.x, dX.y, dY.x, dY.y)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(ushort4, 3, Dv4_t, v4j16, Dv4_f, float4 coord, COORD_PARAMS_3D, S4_S4_, 4, dX.x, dX.y, dX.z, dY.x, dY.y, dY.z)

// Char
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(char, 1, a, i8, f, float coord, COORD_PARAMS_1D, S2_S2_, , dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(char, 2, a, i8, Dv2_f, float2 coord, COORD_PARAMS_2D, S3_S3_, 2, dX.x, dX.y, dY.x, dY.y)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(char, 3, a, i8, Dv4_f, float4 coord, COORD_PARAMS_3D, S3_S3_, 4, dX.x, dX.y, dX.z, dY.x, dY.y, dY.z)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(char2, 1, Dv2_a, v2i8, f, float coord, COORD_PARAMS_1D, S3_S3_, , dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(char2, 2, Dv2_a, v2i8, Dv2_f, float2 coord, COORD_PARAMS_2D, S4_S4_, 2, dX.x, dX.y, dY.x, dY.y)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(char2, 3, Dv2_a, v2i8, Dv4_f, float4 coord, COORD_PARAMS_3D, S4_S4_, 4, dX.x, dX.y, dX.z, dY.x, dY.y, dY.z)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(char4, 1, Dv4_a, v4i8, f, float coord, COORD_PARAMS_1D, S3_S3_, , dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(char4, 2, Dv4_a, v4i8, Dv2_f, float2 coord, COORD_PARAMS_2D, S4_S4_, 2, dX.x, dX.y, dY.x, dY.y)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(char4, 3, Dv4_a, v4i8, Dv4_f, float4 coord, COORD_PARAMS_3D, S4_S4_, 4, dX.x, dX.y, dX.z, dY.x, dY.y, dY.z)

// Unsigned Char
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(uchar, 1, h, j8, f, float coord, COORD_PARAMS_1D, S2_S2_, , dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(uchar, 2, h, j8, Dv2_f, float2 coord, COORD_PARAMS_2D, S3_S3_, 2, dX.x, dX.y, dY.x, dY.y)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(uchar, 3, h, j8, Dv4_f, float4 coord, COORD_PARAMS_3D, S3_S3_, 4, dX.x, dX.y, dX.z, dY.x, dY.y, dY.z)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(uchar2, 1, Dv2_h, v2j8, f, float coord, COORD_PARAMS_1D, S3_S3_, , dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(uchar2, 2, Dv2_h, v2j8, Dv2_f, float2 coord, COORD_PARAMS_2D, S4_S4_, 2, dX.x, dX.y, dY.x, dY.y)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(uchar2, 3, Dv2_h, v2j8, Dv4_f, float4 coord, COORD_PARAMS_3D, S4_S4_, 4, dX.x, dX.y, dX.z, dY.x, dY.y, dY.z)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(uchar4, 1, Dv4_h, v4j8, f, float coord, COORD_PARAMS_1D, S3_S3_, , dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(uchar4, 2, Dv4_h, v4j8, Dv2_f, float2 coord, COORD_PARAMS_2D, S4_S4_, 2, dX.x, dX.y, dY.x, dY.y)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(uchar4, 3, Dv4_h, v4j8, Dv4_f, float4 coord, COORD_PARAMS_3D, S4_S4_, 4, dX.x, dX.y, dX.z, dY.x, dY.y, dY.z)

// Half
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(half, 1, DF16_, f16, f, float coord, COORD_PARAMS_1D, S2_S2_, , dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(half, 2, DF16_, f16, Dv2_f, float2 coord, COORD_PARAMS_2D, S3_S3_, 2, dX.x, dX.y, dY.x, dY.y)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(half, 3, DF16_, f16, Dv4_f, float4 coord, COORD_PARAMS_3D, S3_S3_, 4, dX.x, dX.y, dX.z, dY.x, dY.y, dY.z)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(half2, 1, Dv2_DF16_, v2f16, f, float coord, COORD_PARAMS_1D, S3_S3_, , dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(half2, 2, Dv2_DF16_, v2f16, Dv2_f, float2 coord, COORD_PARAMS_2D, S4_S4_, 2, dX.x, dX.y, dY.x, dY.y)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(half2, 3, Dv2_DF16_, v2f16, Dv4_f, float4 coord, COORD_PARAMS_3D, S4_S4_, 4, dX.x, dX.y, dX.z, dY.x, dY.y, dY.z)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(half4, 1, Dv4_DF16_, v4f16, f, float coord, COORD_PARAMS_1D, S3_S3_, , dX, dY)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(half4, 2, Dv4_DF16_, v4f16, Dv2_f, float2 coord, COORD_PARAMS_2D, S4_S4_, 2, dX.x, dX.y, dY.x, dY.y)
_CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN(half4, 3, Dv4_DF16_, v4f16, Dv4_f, float4 coord, COORD_PARAMS_3D, S4_S4_, 4, dX.x, dX.y, dX.z, dY.x, dY.y, dY.z)

#undef COORD_PARAMS_1D
#undef COORD_PARAMS_2D
#undef COORD_PARAMS_3D

#undef _CLC_DEFINE_MIPMAP_BINDLESS_READS_BUILTIN
