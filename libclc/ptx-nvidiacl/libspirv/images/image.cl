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
struct out_16 {
  short x, y, z, w;
};
#endif

#ifdef cl_khr_3d_image_writes
#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable
#endif

struct out_32 {
  int x, y, z, w;
};

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
struct out_16
__nvvm_suld_1d_v4i16_trap_s(long, int) __asm("llvm.nvvm.suld.1d.v4i16.trap");
struct out_16
__nvvm_suld_2d_v4i16_trap_s(long, int,
                            int) __asm("llvm.nvvm.suld.2d.v4i16.trap");
struct out_16
__nvvm_suld_3d_v4i16_trap_s(long, int, int,
                            int) __asm("llvm.nvvm.suld.3d.v4i16.trap");
struct out_32
__nvvm_suld_1d_v4i32_trap_s(long, int) __asm("llvm.nvvm.suld.1d.v4i32.trap");
struct out_32
__nvvm_suld_2d_v4i32_trap_s(long, int,
                            int) __asm("llvm.nvvm.suld.2d.v4i32.trap");
struct out_32
__nvvm_suld_3d_v4i32_trap_s(long, int, int,
                            int) __asm("llvm.nvvm.suld.3d.v4i32.trap");

struct out_16
__nvvm_suld_1d_v4i16_clamp_s(long, int) __asm("llvm.nvvm.suld.1d.v4i16.clamp");
struct out_16
__nvvm_suld_2d_v4i16_clamp_s(long, int,
                             int) __asm("llvm.nvvm.suld.2d.v4i16.clamp");
struct out_16
__nvvm_suld_3d_v4i16_clamp_s(long, int, int,
                             int) __asm("llvm.nvvm.suld.3d.v4i16.clamp");
struct out_32
__nvvm_suld_1d_v4i32_clamp_s(long, int) __asm("llvm.nvvm.suld.1d.v4i32.clamp");
struct out_32
__nvvm_suld_2d_v4i32_clamp_s(long, int,
                             int) __asm("llvm.nvvm.suld.2d.v4i32.clamp");
struct out_32
__nvvm_suld_3d_v4i32_clamp_s(long, int, int,
                             int) __asm("llvm.nvvm.suld.3d.v4i32.clamp");

struct out_16
__nvvm_suld_1d_v4i16_zero_s(long, int) __asm("llvm.nvvm.suld.1d.v4i16.zero");
struct out_16
__nvvm_suld_2d_v4i16_zero_s(long, int,
                            int) __asm("llvm.nvvm.suld.2d.v4i16.zero");
struct out_16
__nvvm_suld_3d_v4i16_zero_s(long, int, int,
                            int) __asm("llvm.nvvm.suld.3d.v4i16.zero");
struct out_32
__nvvm_suld_1d_v4i32_zero_s(long, int) __asm("llvm.nvvm.suld.1d.v4i32.zero");
struct out_32
__nvvm_suld_2d_v4i32_zero_s(long, int,
                            int) __asm("llvm.nvvm.suld.2d.v4i32.zero");
struct out_32
__nvvm_suld_3d_v4i32_zero_s(long, int, int,
                            int) __asm("llvm.nvvm.suld.3d.v4i32.zero");

struct out_16
__nvvm_suld_1d_v4i16_clamp(read_only image1d_t,
                           int) __asm("llvm.nvvm.suld.1d.v4i16.clamp");
struct out_16
__nvvm_suld_2d_v4i16_clamp(read_only image2d_t, int,
                           int) __asm("llvm.nvvm.suld.2d.v4i16.clamp");
struct out_16
__nvvm_suld_3d_v4i16_clamp(read_only image3d_t, int, int,
                           int) __asm("llvm.nvvm.suld.3d.v4i16.clamp");
struct out_32
__nvvm_suld_1d_v4i32_clamp(read_only image1d_t,
                           int) __asm("llvm.nvvm.suld.1d.v4i32.clamp");
struct out_32
__nvvm_suld_2d_v4i32_clamp(read_only image2d_t, int,
                           int) __asm("llvm.nvvm.suld.2d.v4i32.clamp");
struct out_32
__nvvm_suld_3d_v4i32_clamp(read_only image3d_t, int, int,
                           int) __asm("llvm.nvvm.suld.3d.v4i32.clamp");

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
int __nvvm_suq_depth(long) __asm("llvm.nvvm.suq.depth");

int __nvvm_suq_width_1i(read_only image1d_t) __asm("llvm.nvvm.suq.width");
int __nvvm_suq_width_2i(read_only image2d_t) __asm("llvm.nvvm.suq.width");
int __nvvm_suq_width_3i(read_only image3d_t) __asm("llvm.nvvm.suq.width");
int __nvvm_suq_height_2i(read_only image2d_t) __asm("llvm.nvvm.suq.height");
int __nvvm_suq_height_3i(read_only image3d_t) __asm("llvm.nvvm.suq.height");
int __nvvm_suq_depth_3i(read_only image3d_t) __asm("llvm.nvvm.suq.depth");

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

#define _DEFINE_OUT_TYPE(elem_t, elem_size)                                    \
  inline elem_t##4 out_##elem_t(struct out_##elem_size out) {                  \
    return (elem_t##4)(as_##elem_t(out.x), as_##elem_t(out.y),                 \
                       as_##elem_t(out.z), as_##elem_t(out.w));                \
  }

#define _DEFINE_VEC4_CAST(from_t, to_t)                                        \
  inline to_t##4 cast_##from_t##4_to_##to_t##4(from_t##4 from) {               \
    return (to_t##4)((to_t)from.x, (to_t)from.y, (to_t)from.z, (to_t)from.w);  \
  }

#define _DEFINE_VEC2_CAST(from_t, to_t)                                        \
  inline to_t##2 cast_##from_t##2_to_##to_t##2(from_t##2 from) {               \
    return (to_t##2)((to_t)from.x, (to_t)from.y);                              \
  }

#define _DEFINE_CAST(from_t, to_t)                                             \
  inline to_t cast_##from_t##_to_##to_t(from_t from) { return (to_t)from; }

#define _DEFINE_PIXELF_CAST(pixelf_size, pixelf_base_t, to_t)                  \
  inline to_t cast_pixelf##pixelf_size##_to_##to_t(pixelf##pixelf_size from) { \
    return cast_##pixelf_base_t##_to_##to_t(from);                             \
  }

#define _DEFINE_OUT_PIXELF(pixelf_size, elem_t)                                \
  inline pixelf##pixelf_size out_pixelf##pixelf_size(                          \
      struct out_##pixelf_size out) {                                          \
    return (pixelf##pixelf_size)(as_##elem_t(out.x), as_##elem_t(out.y),       \
                                 as_##elem_t(out.z), as_##elem_t(out.w));      \
  }

#define _DEFINE_READ_1D_PIXELF(pixelf_size, cuda_address_mode)                 \
  pixelf##pixelf_size read_1d_##pixelf_size##_##cuda_address_mode(long image,  \
                                                                  int x) {     \
    struct out_##pixelf_size res =                                             \
        __nvvm_suld_1d_v4i##pixelf_size##_##cuda_address_mode##_s(             \
            image, x * sizeof(struct out_##pixelf_size));                      \
    return out_pixelf##pixelf_size(res);                                       \
  }

#define _DEFINE_READ_2D_PIXELF(pixelf_size, cuda_address_mode)                 \
  pixelf##pixelf_size read_2d_##pixelf_size##_##cuda_address_mode(             \
      long image, int x, int y) {                                              \
    struct out_##pixelf_size res =                                             \
        __nvvm_suld_2d_v4i##pixelf_size##_##cuda_address_mode##_s(             \
            image, x * sizeof(struct out_##pixelf_size), y);                   \
    return out_pixelf##pixelf_size(res);                                       \
  }

#define _DEFINE_READ_3D_PIXELF(pixelf_size, cuda_address_mode)                 \
  pixelf##pixelf_size read_3d_##pixelf_size##_##cuda_address_mode(             \
      long image, int x, int y, int z) {                                       \
    struct out_##pixelf_size res =                                             \
        __nvvm_suld_3d_v4i##pixelf_size##_##cuda_address_mode##_s(             \
            image, x * sizeof(struct out_##pixelf_size), y, z);                \
    return out_pixelf##pixelf_size(res);                                       \
  }

_DEFINE_OUT_TYPE(float, 32)
_DEFINE_OUT_TYPE(int, 32)
_DEFINE_OUT_TYPE(uint, 32)

_DEFINE_VEC4_CAST(float, int)
_DEFINE_VEC4_CAST(int, float)
_DEFINE_VEC4_CAST(float, uint)
_DEFINE_VEC4_CAST(uint, float)

_DEFINE_VEC2_CAST(int, float)

_DEFINE_CAST(int, float)
_DEFINE_CAST(float, float)
_DEFINE_CAST(float2, float2)
_DEFINE_CAST(float4, float4)
_DEFINE_CAST(pixelf32, float4)
_DEFINE_CAST(pixelf32, pixelf32)
_DEFINE_CAST(float4, pixelf32)

_DEFINE_OUT_PIXELF(32, float)

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
_DEFINE_OUT_TYPE(half, 16)
_DEFINE_OUT_PIXELF(16, half)
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

#undef _DEFINE_OUT_TYPE
#undef _DEFINE_VEC4_CAST
#undef _DEFINE_VEC2_CAST
#undef _DEFINE_CAST
#undef _DEFINE_OUT_PIXELF
#undef _DEFINE_READ_1D_PIXELF
#undef _DEFINE_READ_2D_PIXELF
#undef _DEFINE_READ_3D_PIXELF

// Builtins

// Unsampled images
#define _CLC_DEFINE_IMAGE1D_READ_BUILTIN(elem_t, elem_t_mangled, elem_size)         \
  _CLC_DECL                                                                         \
  elem_t##4 _Z17__spirv_ImageReadIDv4_##elem_t_mangled##14ocl_image1d_roiET_T0_T1_( \
      read_only image1d_t image, int x) {                                           \
    return out_##elem_t(                                                            \
        __nvvm_suld_1d_v4i##elem_size##_clamp(image, x * sizeof(elem_t##4)));       \
  }

#define _CLC_DEFINE_IMAGE2D_READ_BUILTIN(elem_t, elem_t_mangled, elem_size)             \
  _CLC_DECL                                                                             \
  elem_t##4 _Z17__spirv_ImageReadIDv4_##elem_t_mangled##14ocl_image2d_roDv2_iET_T0_T1_( \
      read_only image2d_t image, int2 coord) {                                          \
    return out_##elem_t(__nvvm_suld_2d_v4i##elem_size##_clamp(                          \
        image, coord.x * sizeof(elem_t##4), coord.y));                                  \
  }

#define _CLC_DEFINE_IMAGE3D_READ_BUILTIN(elem_t, elem_t_mangled, elem_size,                         \
                                         coord_mangled)                                             \
  _CLC_DECL                                                                                         \
  elem_t##4 _Z17__spirv_ImageReadIDv4_##elem_t_mangled##14ocl_image3d_ro##coord_mangled##ET_T0_T1_( \
      read_only image3d_t image, int4 coord) {                                                      \
    return out_##elem_t(__nvvm_suld_3d_v4i##elem_size##_clamp(                                      \
        image, coord.x * sizeof(elem_t##4), coord.y, coord.z));                                     \
  }

#define _CLC_DEFINE_IMAGE1D_WRITE_BUILTIN(elem_t, elem_t_mangled, elem_size,    \
                                          int_rep)                              \
  _CLC_DECL void                                                                \
      _Z18__spirv_ImageWriteI14ocl_image1d_woiDv4_##elem_t_mangled##EvT_T0_T1_( \
          write_only image1d_t image, int x, elem_t##4 c) {                     \
    __nvvm_sust_1d_v4i##elem_size##_clamp(                                      \
        image, x * sizeof(elem_t##4), as_##int_rep(c.x), as_##int_rep(c.y),     \
        as_##int_rep(c.z), as_##int_rep(c.w));                                  \
  }

#define _CLC_DEFINE_IMAGE2D_WRITE_BUILTIN(elem_t, elem_t_mangled, elem_size,        \
                                          int_rep)                                  \
  _CLC_DECL void                                                                    \
      _Z18__spirv_ImageWriteI14ocl_image2d_woDv2_iDv4_##elem_t_mangled##EvT_T0_T1_( \
          write_only image2d_t image, int2 coord, elem_t##4 c) {                    \
    __nvvm_sust_2d_v4i##elem_size##_clamp(                                          \
        image, coord.x * sizeof(elem_t##4), coord.y, as_##int_rep(c.x),             \
        as_##int_rep(c.y), as_##int_rep(c.z), as_##int_rep(c.w));                   \
  }

#define _CLC_DEFINE_IMAGE3D_WRITE_BUILTIN(elem_t, elem_t_mangled, elem_size,   \
                                          int_rep, val_mangled)                \
  _CLC_DECL void                                                               \
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
  _CLC_DECL __ocl_sampled_image##dims##d_ro_t                                                                       \
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
      return out_##elem_t(                                                     \
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
      return out_##elem_t(                                                     \
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
      return out_##elem_t(                                                     \
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
      return out_##elem_t(__nvvm_suld_1d_v4i##elem_size##_trap_s(              \
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
      return out_##elem_t(__nvvm_suld_2d_v4i##elem_size##_trap_s(              \
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
      return out_##elem_t(__nvvm_suld_3d_v4i##elem_size##_trap_s(              \
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
      return out_##elem_t(__nvvm_suld_1d_v4i##elem_size##_trap_s(              \
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
      return out_##elem_t(__nvvm_suld_2d_v4i##elem_size##_trap_s(              \
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
      return out_##elem_t(__nvvm_suld_3d_v4i##elem_size##_trap_s(              \
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
  _DEFINE_COMMON_SAMPLED_LOAD_1D(elem_t, elem_size, none, trap)                \
  _DEFINE_COMMON_SAMPLED_LOAD_2D(elem_t, elem_size, none, trap)                \
  _DEFINE_COMMON_SAMPLED_LOAD_3D(elem_t, elem_size, none, trap)                \
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
  _CLC_DECL                                                                                                                                       \
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
_CLC_DECL int _Z22__spirv_ImageQuerySizeIDv2_i14ocl_image1d_roET_T0_(
    read_only image1d_t image) {
  return __nvvm_suq_width_1i(image);
}

_CLC_DECL int2 _Z22__spirv_ImageQuerySizeIDv2_i14ocl_image2d_roET_T0_(
    read_only image2d_t image) {
  int width = __nvvm_suq_width_2i(image);
  int height = __nvvm_suq_height_2i(image);
  return (int2)(width, height);
}

_CLC_DECL int4 _Z22__spirv_ImageQuerySizeIDv2_i14ocl_image3d_roET_T0_(
    read_only image3d_t image) {
  int width = __nvvm_suq_width_3i(image);
  int height = __nvvm_suq_height_3i(image);
  int depth = __nvvm_suq_depth_3i(image);
  return (int4)(width, height, depth, 0);
}