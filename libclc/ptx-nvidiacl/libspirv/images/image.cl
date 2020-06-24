//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>
#include <spirv/spirv.h>

#define _CLC_DEFINE_SAMPLED_IMAGE_BUILTIN(dims)                                                                     \
  __ocl_sampled_image##dims##d_ro_t __clc__sampled_image##dims##d_pack(                                             \
      read_only image##dims##d_t,                                                                                   \
      sampler_t) __asm("__clc__sampled_image_pack");                                                                \
  _CLC_DECL __ocl_sampled_image##dims##d_ro_t                                                                       \
      _Z20__spirv_SampledImageI14ocl_image##dims##d_ro32__spirv_SampledImage__image##dims##d_roET0_T_11ocl_sampler( \
          read_only image##dims##d_t image, sampler_t sampler) {                                                    \
    return __clc__sampled_image##dims##d_pack(image, sampler);                                                      \
  }

_CLC_DEFINE_SAMPLED_IMAGE_BUILTIN(1)
_CLC_DEFINE_SAMPLED_IMAGE_BUILTIN(2)
_CLC_DEFINE_SAMPLED_IMAGE_BUILTIN(3)

#undef _CLC_DEFINE_SAMPLED_IMAGE_BUILTIN

struct out_32 {
  int x, y, z, w;
};

#define _CLC_DEFINE_IMAGE1D_READ_BUILTIN(elem_t, elem_t_mangled, elem_size)         \
  struct out_##elem_size __nvvm_suld_1d_v4i##elem_size##_clamp(                     \
      read_only image1d_t,                                                          \
      int) __asm("llvm.nvvm.suld.1d.v4i" #elem_size ".clamp");                      \
  _CLC_DECL                                                                         \
  elem_t##4 _Z17__spirv_ImageReadIDv4_##elem_t_mangled##14ocl_image1d_roiET_T0_T1_( \
      read_only image1d_t image, int x) {                                           \
    struct out_##elem_size res =                                                    \
        __nvvm_suld_1d_v4i##elem_size##_clamp(image, x * sizeof(elem_t##4));        \
    return (elem_t##4)(as_##elem_t(res.x), as_##elem_t(res.y),                      \
                       as_##elem_t(res.z), as_##elem_t(res.w));                     \
  }

#define _CLC_DEFINE_IMAGE2D_READ_BUILTIN(elem_t, elem_t_mangled, elem_size)             \
  struct out_##elem_size __nvvm_suld_2d_v4i##elem_size##_clamp(                         \
      read_only image2d_t, int,                                                         \
      int) __asm("llvm.nvvm.suld.2d.v4i" #elem_size ".clamp");                          \
  _CLC_DECL                                                                             \
  elem_t##4 _Z17__spirv_ImageReadIDv4_##elem_t_mangled##14ocl_image2d_roDv2_iET_T0_T1_( \
      read_only image2d_t image, int2 coord) {                                          \
    struct out_##elem_size res = __nvvm_suld_2d_v4i##elem_size##_clamp(                 \
        image, coord.x * sizeof(elem_t##4), coord.y);                                   \
    return (elem_t##4)(as_##elem_t(res.x), as_##elem_t(res.y),                          \
                       as_##elem_t(res.z), as_##elem_t(res.w));                         \
  }

#define _CLC_DEFINE_IMAGE3D_READ_BUILTIN(elem_t, elem_t_mangled, elem_size,                         \
                                         coord_mangled)                                             \
  struct out_##elem_size __nvvm_suld_3d_v4i##elem_size##_clamp(                                     \
      read_only image3d_t, int, int,                                                                \
      int) __asm("llvm.nvvm.suld.3d.v4i" #elem_size ".clamp");                                      \
  _CLC_DECL                                                                                         \
  elem_t##4 _Z17__spirv_ImageReadIDv4_##elem_t_mangled##14ocl_image3d_ro##coord_mangled##ET_T0_T1_( \
      read_only image3d_t image, int4 coord) {                                                      \
    struct out_##elem_size res = __nvvm_suld_3d_v4i##elem_size##_clamp(                             \
        image, coord.x * sizeof(elem_t##4), coord.y, coord.z);                                      \
    return (elem_t##4)(as_##elem_t(res.x), as_##elem_t(res.y),                                      \
                       as_##elem_t(res.z), as_##elem_t(res.w));                                     \
  }

#define _CLC_DEFINE_IMAGE1D_WRITE_BUILTIN(elem_t, elem_t_mangled, elem_size)    \
  void __nvvm_sust_1d_v4i##elem_size##_clamp(                                   \
      write_only image1d_t, int, int, int, int,                                 \
      int) __asm("llvm.nvvm.sust.b.1d.v4i" #elem_size ".clamp");                \
  _CLC_DECL void                                                                \
      _Z18__spirv_ImageWriteI14ocl_image1d_woiDv4_##elem_t_mangled##EvT_T0_T1_( \
          write_only image1d_t image, int x, elem_t##4 c) {                     \
    __nvvm_sust_1d_v4i##elem_size##_clamp(image, x * sizeof(elem_t##4),         \
                                          as_int(c.x), as_int(c.y),             \
                                          as_int(c.z), as_int(c.w));            \
  }

#define _CLC_DEFINE_IMAGE2D_WRITE_BUILTIN(elem_t, elem_t_mangled, elem_size)        \
  void __nvvm_sust_2d_v4i##elem_size##_clamp(                                       \
      write_only image2d_t, int, int, int, int, int,                                \
      int) __asm("llvm.nvvm.sust.b.2d.v4i" #elem_size ".clamp");                    \
  _CLC_DECL void                                                                    \
      _Z18__spirv_ImageWriteI14ocl_image2d_woDv2_iDv4_##elem_t_mangled##EvT_T0_T1_( \
          write_only image2d_t image, int2 coord, elem_t##4 c) {                    \
    __nvvm_sust_2d_v4i##elem_size##_clamp(image, coord.x * sizeof(elem_t##4),       \
                                          coord.y, as_int(c.x), as_int(c.y),        \
                                          as_int(c.z), as_int(c.w));                \
  }

#define _CLC_DEFINE_IMAGE3D_WRITE_BUILTIN(elem_t, elem_t_mangled, elem_size,   \
                                          val_mangled)                         \
  void __nvvm_sust_3d_v4i##elem_size##_clamp(                                  \
      write_only image3d_t, int, int, int, int, int, int,                      \
      int) __asm("llvm.nvvm.sust.b.3d.v4i" #elem_size ".clamp");               \
  _CLC_DECL void                                                               \
      _Z18__spirv_ImageWriteI14ocl_image3d_woDv4_i##val_mangled##EvT_T0_T1_(   \
          write_only image3d_t image, int4 coord, elem_t##4 c) {               \
    __nvvm_sust_3d_v4i##elem_size##_clamp(                                     \
        image, coord.x * sizeof(elem_t##4), coord.y, coord.z, as_int(c.x),     \
        as_int(c.y), as_int(c.z), as_int(c.w));                                \
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

_CLC_DEFINE_IMAGE1D_WRITE_BUILTIN(float, f, 32)
_CLC_DEFINE_IMAGE1D_WRITE_BUILTIN(int, i, 32)
_CLC_DEFINE_IMAGE1D_WRITE_BUILTIN(uint, j, 32)

_CLC_DEFINE_IMAGE2D_WRITE_BUILTIN(float, f, 32)
_CLC_DEFINE_IMAGE2D_WRITE_BUILTIN(int, i, 32)
_CLC_DEFINE_IMAGE2D_WRITE_BUILTIN(uint, j, 32)

#ifdef cl_khr_3d_image_writes
#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable
_CLC_DEFINE_IMAGE3D_WRITE_BUILTIN(float, f, 32, Dv4_f)
_CLC_DEFINE_IMAGE3D_WRITE_BUILTIN(int, i, 32, S1_)
_CLC_DEFINE_IMAGE3D_WRITE_BUILTIN(uint, j, 32, Dv4_j)
#endif

#undef _CLC_DEFINE_IMAGE1D_READ_BUILTIN
#undef _CLC_DEFINE_IMAGE2D_READ_BUILTIN
#undef _CLC_DEFINE_IMAGE3D_READ_BUILTIN
#undef _CLC_DEFINE_IMAGE1D_WRITE_BUILTIN
#undef _CLC_DEFINE_IMAGE2D_WRITE_BUILTIN
#undef _CLC_DEFINE_IMAGE3D_WRITE_BUILTIN