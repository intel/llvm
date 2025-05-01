#include "image_common.h"
#include <libspirv/spirv.h>

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#ifdef cl_khr_3d_image_writes
#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable
#endif

// Declare ockl functions/builtins that we link from the ROCm device libs.
float4 __ockl_image_load_1D(_CLC_CONST_AS unsigned int *tex, int coord);
float4 __ockl_image_load_2D(_CLC_CONST_AS unsigned int *tex, int2 coord);
float4 __ockl_image_load_3D(_CLC_CONST_AS unsigned int *tex, int3 coord);
half4 __ockl_image_loadh_1D(_CLC_CONST_AS unsigned int *tex, int coord);
half4 __ockl_image_loadh_2D(_CLC_CONST_AS unsigned int *tex, int2 coord);
half4 __ockl_image_loadh_3D(_CLC_CONST_AS unsigned int *tex, int3 coord);

void __ockl_image_store_1D(_CLC_CONST_AS unsigned int *tex, int coord,
                           float4 color);
void __ockl_image_store_2D(_CLC_CONST_AS unsigned int *tex, int2 coord,
                           float4 color);
void __ockl_image_store_3D(_CLC_CONST_AS unsigned int *tex, int3 coord,
                           float4 color);
void __ockl_image_storeh_1D(_CLC_CONST_AS unsigned int *tex, int coord,
                            half4 color);
void __ockl_image_storeh_2D(_CLC_CONST_AS unsigned int *tex, int2 coord,
                            half4 color);
void __ockl_image_storeh_3D(_CLC_CONST_AS unsigned int *tex, int3 coord,
                            half4 color);

float4 __ockl_image_sample_1D(_CLC_CONST_AS unsigned int *tex,
                              _CLC_CONST_AS unsigned int *samp, float coord);
float4 __ockl_image_sample_2D(_CLC_CONST_AS unsigned int *tex,
                              _CLC_CONST_AS unsigned int *samp, float2 coord);
float4 __ockl_image_sample_3D(_CLC_CONST_AS unsigned int *tex,
                              _CLC_CONST_AS unsigned int *samp, float3 coord);
half4 __ockl_image_sampleh_1D(_CLC_CONST_AS unsigned int *tex,
                              _CLC_CONST_AS unsigned int *samp, float coord);
half4 __ockl_image_sampleh_2D(_CLC_CONST_AS unsigned int *tex,
                              _CLC_CONST_AS unsigned int *samp, float2 coord);
half4 __ockl_image_sampleh_3D(_CLC_CONST_AS unsigned int *tex,
                              _CLC_CONST_AS unsigned int *samp, float3 coord);

//
// IMAGES
//

// Fetch Ops

#define _CLC_DEFINE_IMAGE_BINDLESS_FETCH_BUILTIN(                              \
    dimension, elem_t, elem_t_mangled, coord_t, coord_mangled, builtin_ret_t,  \
    builtin_ret_postfix)                                                       \
  _CLC_DEF elem_t _CLC_MANGLE_FUNC_IMG_HANDLE(                                 \
      18, __spirv_ImageFetch, I##elem_t_mangled,                               \
      coord_mangled##ET_T0_T1_)(ulong imageHandle, coord_t coord) {            \
    _CLC_CONST_AS unsigned int *tex =                                          \
        (_CLC_CONST_AS unsigned int *)imageHandle;                             \
    builtin_ret_t##4 color =                                                   \
        __ockl_image_load##builtin_ret_postfix##_##dimension##D(tex, coord);   \
    return __clc_cast_from_##builtin_ret_t##4_to_##elem_t(color);              \
  }

#define _CLC_DEFINE_IMAGE_BINDLESS_FETCH_32_BUILTIN(                           \
    dimension, elem_t, elem_t_mangled, coord_t, coord_mangled)                 \
  _CLC_DEFINE_IMAGE_BINDLESS_FETCH_BUILTIN(dimension, elem_t, elem_t_mangled,  \
                                           coord_t, coord_mangled, float, )

#define _CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN(                           \
    dimension, elem_t, elem_t_mangled, coord_t, coord_mangled)                 \
  _CLC_DEFINE_IMAGE_BINDLESS_FETCH_BUILTIN(dimension, elem_t, elem_t_mangled,  \
                                           coord_t, coord_mangled, half, h)

#define _CLC_DEFINE_IMAGE_BINDLESS_FETCH_8_BUILTIN(                            \
    dimension, elem_t, elem_t_mangled, coord_t, coord_mangled)                 \
  _CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN(                                 \
      dimension, elem_t, elem_t_mangled, coord_t, coord_mangled)

// Float
// return 1-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_32_BUILTIN(1, float, f, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_32_BUILTIN(2, float, f, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_32_BUILTIN(3, float, f, int3, Dv3_i)
// return 2-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_32_BUILTIN(1, float2, Dv2_f, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_32_BUILTIN(2, float2, Dv2_f, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_32_BUILTIN(3, float2, Dv2_f, int3, Dv3_i)
// return 4-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_32_BUILTIN(1, float4, Dv4_f, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_32_BUILTIN(2, float4, Dv4_f, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_32_BUILTIN(3, float4, Dv4_f, int3, Dv3_i)

// Half
#ifdef cl_khr_fp16
// return 1-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN(1, half, Dh, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN(2, half, Dh, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN(3, half, Dh, int3, Dv3_i)
// return 2-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN(1, half2, Dv2_Dh, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN(2, half2, Dv2_Dh, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN(3, half2, Dv2_Dh, int3, Dv3_i)
// return 4-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN(1, half4, Dv4_Dh, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN(2, half4, Dv4_Dh, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN(3, half4, Dv4_Dh, int3, Dv3_i)
#endif

// Int
// return 1-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_32_BUILTIN(1, int, i, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_32_BUILTIN(2, int, i, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_32_BUILTIN(3, int, i, int3, Dv3_i)
// return 2-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_32_BUILTIN(1, int2, Dv2_i, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_32_BUILTIN(2, int2, Dv2_i, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_32_BUILTIN(3, int2, Dv2_i, int3, Dv3_i)
// return 4-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_32_BUILTIN(1, int4, Dv4_i, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_32_BUILTIN(2, int4, Dv4_i, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_32_BUILTIN(3, int4, Dv4_i, int3, Dv3_i)

// Unsigned Int
// return 1-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_32_BUILTIN(1, uint, j, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_32_BUILTIN(2, uint, j, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_32_BUILTIN(3, uint, j, int3, Dv3_i)
// return 2-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_32_BUILTIN(1, uint2, Dv2_j, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_32_BUILTIN(2, uint2, Dv2_j, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_32_BUILTIN(3, uint2, Dv2_j, int3, Dv3_i)
// return 4-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_32_BUILTIN(1, uint4, Dv4_j, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_32_BUILTIN(2, uint4, Dv4_j, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_32_BUILTIN(3, uint4, Dv4_j, int3, Dv3_i)

// Short
// return 1-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN(1, short, s, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN(2, short, s, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN(3, short, s, int3, Dv3_i)
// return 2-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN(1, short2, Dv2_s, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN(2, short2, Dv2_s, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN(3, short2, Dv2_s, int3, Dv3_i)
// return 4-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN(1, short4, Dv4_s, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN(2, short4, Dv4_s, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN(3, short4, Dv4_s, int3, Dv3_i)

// Unsigned Short
// return 1-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN(1, ushort, t, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN(2, ushort, t, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN(3, ushort, t, int3, Dv3_i)
// return 2-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN(1, ushort2, Dv2_t, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN(2, ushort2, Dv2_t, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN(3, ushort2, Dv2_t, int3, Dv3_i)
// return 4-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN(1, ushort4, Dv4_t, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN(2, ushort4, Dv4_t, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN(3, ushort4, Dv4_t, int3, Dv3_i)

// Char
// return 1-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_8_BUILTIN(1, char, a, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_8_BUILTIN(2, char, a, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_8_BUILTIN(3, char, a, int3, Dv3_i)
// return 2-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_8_BUILTIN(1, char2, Dv2_a, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_8_BUILTIN(2, char2, Dv2_a, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_8_BUILTIN(3, char2, Dv2_a, int3, Dv3_i)
// return 4-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_8_BUILTIN(1, char4, Dv4_a, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_8_BUILTIN(2, char4, Dv4_a, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_8_BUILTIN(3, char4, Dv4_a, int3, Dv3_i)

// Unsigned Char
// return 1-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_8_BUILTIN(1, uchar, h, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_8_BUILTIN(2, uchar, h, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_8_BUILTIN(3, uchar, h, int3, Dv3_i)
// return 2-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_8_BUILTIN(1, uchar2, Dv2_h, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_8_BUILTIN(2, uchar2, Dv2_h, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_8_BUILTIN(3, uchar2, Dv2_h, int3, Dv3_i)
// return 4-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_8_BUILTIN(1, uchar4, Dv4_h, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_8_BUILTIN(2, uchar4, Dv4_h, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_FETCH_8_BUILTIN(3, uchar4, Dv4_h, int3, Dv3_i)

#undef _CLC_DEFINE_IMAGE_BINDLESS_FETCH_8_BUILTIN
#undef _CLC_DEFINE_IMAGE_BINDLESS_FETCH_16_BUILTIN
#undef _CLC_DEFINE_IMAGE_BINDLESS_FETCH_32_BUILTIN

// Write Ops

#define _CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(                              \
    dimension, elem_t, elem_t_mangled, coord_t, coord_mangled, builtin_ret_t,  \
    builtin_ret_postfix)                                                       \
  _CLC_DEF void _CLC_MANGLE_FUNC_IMG_HANDLE(                                   \
      18, __spirv_ImageWrite, I, coord_mangled##elem_t_mangled##EvT_T0_T1_)(   \
      ulong imageHandle, coord_t coord, elem_t color) {                        \
    _CLC_CONST_AS unsigned int *tex =                                          \
        (_CLC_CONST_AS unsigned int *)imageHandle;                             \
    builtin_ret_t##4 outColor =                                                \
        __clc_cast_from_##elem_t##_to_##builtin_ret_t##4(color);               \
    __ockl_image_store##builtin_ret_postfix##_##dimension##D(tex, coord,       \
                                                             outColor);        \
  }

#define _CLC_DEFINE_IMAGE_BINDLESS_WRITE_32_BUILTIN(                           \
    dimension, elem_t, elem_t_mangled, coord_t, coord_mangled)                 \
  _CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(dimension, elem_t, elem_t_mangled,  \
                                           coord_t, coord_mangled, float, )

#define _CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN(                           \
    dimension, elem_t, elem_t_mangled, coord_t, coord_mangled)                 \
  _CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN(dimension, elem_t, elem_t_mangled,  \
                                           coord_t, coord_mangled, half, h)

#define _CLC_DEFINE_IMAGE_BINDLESS_WRITE_8_BUILTIN(                            \
    dimension, elem_t, elem_t_mangled, coord_t, coord_mangled)                 \
  _CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN(                                 \
      dimension, elem_t, elem_t_mangled, coord_t, coord_mangled)

// Float
// write 1-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_32_BUILTIN(1, float, f, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_32_BUILTIN(2, float, f, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_32_BUILTIN(3, float, f, int3, Dv3_i)
// write 2-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_32_BUILTIN(1, float2, Dv2_f, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_32_BUILTIN(2, float2, Dv2_f, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_32_BUILTIN(3, float2, Dv2_f, int3, Dv3_i)
// write 4-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_32_BUILTIN(1, float4, Dv4_f, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_32_BUILTIN(2, float4, Dv4_f, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_32_BUILTIN(3, float4, Dv4_f, int3, Dv3_i)

// Half
#ifdef cl_khr_fp16
// write 1-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN(1, half, Dh, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN(2, half, Dh, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN(3, half, Dh, int3, Dv3_i)
// write 2-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN(1, half2, Dv2_Dh, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN(2, half2, Dv2_Dh, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN(3, half2, Dv2_Dh, int3, Dv3_i)
// write 4-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN(1, half4, Dv4_Dh, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN(2, half4, Dv4_Dh, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN(3, half4, Dv4_Dh, int3, Dv3_i)
#endif

// Int
// write 1-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_32_BUILTIN(1, int, i, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_32_BUILTIN(2, int, i, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_32_BUILTIN(3, int, i, int3, Dv3_i)
// write 2-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_32_BUILTIN(1, int2, Dv2_i, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_32_BUILTIN(2, int2, Dv2_i, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_32_BUILTIN(3, int2, Dv2_i, int3, Dv3_i)
// write 4-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_32_BUILTIN(1, int4, Dv4_i, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_32_BUILTIN(2, int4, Dv4_i, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_32_BUILTIN(3, int4, Dv4_i, int3, Dv3_i)

// Unsigned Int
// write 1-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_32_BUILTIN(1, uint, j, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_32_BUILTIN(2, uint, j, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_32_BUILTIN(3, uint, j, int3, Dv3_i)
// write 2-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_32_BUILTIN(1, uint2, Dv2_j, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_32_BUILTIN(2, uint2, Dv2_j, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_32_BUILTIN(3, uint2, Dv2_j, int3, Dv3_i)
// write 4-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_32_BUILTIN(1, uint4, Dv4_j, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_32_BUILTIN(2, uint4, Dv4_j, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_32_BUILTIN(3, uint4, Dv4_j, int3, Dv3_i)

// Short
// write 1-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN(1, short, s, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN(2, short, s, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN(3, short, s, int3, Dv3_i)
// write 2-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN(1, short2, Dv2_s, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN(2, short2, Dv2_s, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN(3, short2, Dv2_s, int3, Dv3_i)
// write 4-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN(1, short4, Dv4_s, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN(2, short4, Dv4_s, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN(3, short4, Dv4_s, int3, Dv3_i)

// Unsigned Short
// write 1-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN(1, ushort, t, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN(2, ushort, t, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN(3, ushort, t, int3, Dv3_i)
// write 2-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN(1, ushort2, Dv2_t, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN(2, ushort2, Dv2_t, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN(3, ushort2, Dv2_t, int3, Dv3_i)
// write 4-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN(1, ushort4, Dv4_t, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN(2, ushort4, Dv4_t, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN(3, ushort4, Dv4_t, int3, Dv3_i)

// Char
// write 1-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_8_BUILTIN(1, char, a, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_8_BUILTIN(2, char, a, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_8_BUILTIN(3, char, a, int3, Dv3_i)
// write 2-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_8_BUILTIN(1, char2, Dv2_a, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_8_BUILTIN(2, char2, Dv2_a, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_8_BUILTIN(3, char2, Dv2_a, int3, Dv3_i)
// write 4-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_8_BUILTIN(1, char4, Dv4_a, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_8_BUILTIN(2, char4, Dv4_a, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_8_BUILTIN(3, char4, Dv4_a, int3, Dv3_i)

// Unsigned Char
// write 1-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_8_BUILTIN(1, uchar, h, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_8_BUILTIN(2, uchar, h, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_8_BUILTIN(3, uchar, h, int3, Dv3_i)
// write 2-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_8_BUILTIN(1, uchar2, Dv2_h, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_8_BUILTIN(2, uchar2, Dv2_h, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_8_BUILTIN(3, uchar2, Dv2_h, int3, Dv3_i)
// write 4-channel color data
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_8_BUILTIN(1, uchar4, Dv4_h, int, i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_8_BUILTIN(2, uchar4, Dv4_h, int2, Dv2_i)
_CLC_DEFINE_IMAGE_BINDLESS_WRITE_8_BUILTIN(3, uchar4, Dv4_h, int3, Dv3_i)

#undef _CLC_DEFINE_IMAGE_BINDLESS_WRITE_8_BUILTIN
#undef _CLC_DEFINE_IMAGE_BINDLESS_WRITE_16_BUILTIN
#undef _CLC_DEFINE_IMAGE_BINDLESS_WRITE_32_BUILTIN

//
// SAMPLED IMAGES
//

// Read Ops

#define _CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(                        \
    dimension, elem_t, elem_t_mangled, coord_t, coord_mangled, builtin_ret_t,  \
    builtin_ret_postfix)                                                       \
  _CLC_DEF elem_t _CLC_MANGLE_FUNC_IMG_HANDLE(                                 \
      17, __spirv_ImageRead, I##elem_t_mangled,                                \
      coord_mangled##ET_T0_T1_)(ulong imageHandle, coord_t coord) {            \
    _CLC_CONST_AS unsigned int *tex =                                          \
        (_CLC_CONST_AS unsigned int *)imageHandle;                             \
    _CLC_CONST_AS unsigned int *samp = tex + SAMPLER_OBJECT_OFFSET_DWORD;      \
    builtin_ret_t##4 color =                                                   \
        __ockl_image_sample##builtin_ret_postfix##_##dimension##D(tex, samp,   \
                                                                  coord);      \
    return __clc_cast_from_##builtin_ret_t##4_to_##elem_t(color);              \
  }

#define _CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_32_BUILTIN(                     \
    dimension, elem_t, elem_t_mangled, coord_t, coord_mangled)                 \
  _CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(                              \
      dimension, elem_t, elem_t_mangled, coord_t, coord_mangled, float, )

#define _CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN(                     \
    dimension, elem_t, elem_t_mangled, coord_t, coord_mangled)                 \
  _CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN(                              \
      dimension, elem_t, elem_t_mangled, coord_t, coord_mangled, half, h)

#define _CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_8_BUILTIN(                      \
    dimension, elem_t, elem_t_mangled, coord_t, coord_mangled)                 \
  _CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN(                           \
      dimension, elem_t, elem_t_mangled, coord_t, coord_mangled)

// Float
// return 1 channel color data
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_32_BUILTIN(1, float, f, float, f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_32_BUILTIN(2, float, f, float2, Dv2_f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_32_BUILTIN(3, float, f, float3, Dv3_f)
// return 2-channel color data
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_32_BUILTIN(1, float2, Dv2_f, float, f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_32_BUILTIN(2, float2, Dv2_f, float2,
                                                  Dv2_f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_32_BUILTIN(3, float2, Dv2_f, float3,
                                                  Dv3_f)
// return 4-channel color data
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_32_BUILTIN(1, float4, Dv4_f, float, f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_32_BUILTIN(2, float4, Dv4_f, float2,
                                                  Dv2_f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_32_BUILTIN(3, float4, Dv4_f, float3,
                                                  Dv3_f)

// Half
#ifdef cl_khr_fp16
// return 1 channel color data
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN(1, half, Dh, float, f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN(2, half, Dh, float2, Dv2_f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN(3, half, Dh, float3, Dv3_f)
// return 2-channel color data
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN(1, half2, Dv2_Dh, float, f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN(2, half2, Dv2_Dh, float2,
                                                  Dv2_f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN(3, half2, Dv2_Dh, float3,
                                                  Dv3_f)
// return 4-channel color data
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN(1, half4, Dv4_Dh, float, f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN(2, half4, Dv4_Dh, float2,
                                                  Dv2_f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN(3, half4, Dv4_Dh, float3,
                                                  Dv3_f)
#endif

// Int
// return 1-channel color data
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_32_BUILTIN(1, int, i, float, f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_32_BUILTIN(2, int, i, float2, Dv2_f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_32_BUILTIN(3, int, i, float3, Dv3_f)
// return 2-channel color data
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_32_BUILTIN(1, int2, Dv2_i, float, f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_32_BUILTIN(2, int2, Dv2_i, float2, Dv2_f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_32_BUILTIN(3, int2, Dv2_i, float3, Dv3_f)
// return 4-channel color data
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_32_BUILTIN(1, int4, Dv4_i, float, f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_32_BUILTIN(2, int4, Dv4_i, float2, Dv2_f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_32_BUILTIN(3, int4, Dv4_i, float3, Dv3_f)

// Unsigned Int
// return 1-channel color data
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_32_BUILTIN(1, uint, j, float, f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_32_BUILTIN(2, uint, j, float2, Dv2_f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_32_BUILTIN(3, uint, j, float3, Dv3_f)
// return 2-channel color data
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_32_BUILTIN(1, uint2, Dv2_j, float, f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_32_BUILTIN(2, uint2, Dv2_j, float2,
                                                  Dv2_f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_32_BUILTIN(3, uint2, Dv2_j, float3,
                                                  Dv3_f)
// return 4-channel color data
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_32_BUILTIN(1, uint4, Dv4_j, float, f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_32_BUILTIN(2, uint4, Dv4_j, float2,
                                                  Dv2_f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_32_BUILTIN(3, uint4, Dv4_j, float3,
                                                  Dv3_f)

// Short
// return 1-channel color data
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN(1, short, s, float, f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN(2, short, s, float2, Dv2_f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN(3, short, s, float3, Dv3_f)
// return 2-channel color data
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN(1, short2, Dv2_s, float, f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN(2, short2, Dv2_s, float2,
                                                  Dv2_f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN(3, short2, Dv2_s, float3,
                                                  Dv3_f)
// return 4-channel color data
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN(1, short4, Dv4_s, float, f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN(2, short4, Dv4_s, float2,
                                                  Dv2_f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN(3, short4, Dv4_s, float3,
                                                  Dv3_f)

// Unsigned Short
// return 1-channel color data
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN(1, ushort, t, float, f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN(2, ushort, t, float2, Dv2_f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN(3, ushort, t, float3, Dv3_f)
// return 2-channel color data
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN(1, ushort2, Dv2_t, float, f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN(2, ushort2, Dv2_t, float2,
                                                  Dv2_f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN(3, ushort2, Dv2_t, float3,
                                                  Dv3_f)
// return 4-channel color data
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN(1, ushort4, Dv4_t, float, f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN(2, ushort4, Dv4_t, float2,
                                                  Dv2_f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN(3, ushort4, Dv4_t, float3,
                                                  Dv3_f)

// Char
// return 1-channel color data
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_8_BUILTIN(1, char, a, float, f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_8_BUILTIN(2, char, a, float2, Dv2_f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_8_BUILTIN(3, char, a, float3, Dv3_f)
// return 2-channel color data
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_8_BUILTIN(1, char2, Dv2_a, float, f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_8_BUILTIN(2, char2, Dv2_a, float2, Dv2_f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_8_BUILTIN(3, char2, Dv2_a, float3, Dv3_f)
// return 4-channel color data
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_8_BUILTIN(1, char4, Dv4_a, float, f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_8_BUILTIN(2, char4, Dv4_a, float2, Dv2_f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_8_BUILTIN(3, char4, Dv4_a, float3, Dv3_f)

// Unsigned Char
// return 1-channel color data
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_8_BUILTIN(1, uchar, h, float, f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_8_BUILTIN(2, uchar, h, float2, Dv2_f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_8_BUILTIN(3, uchar, h, float3, Dv3_f)
// return 2-channel color data
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_8_BUILTIN(1, uchar2, Dv2_h, float, f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_8_BUILTIN(2, uchar2, Dv2_h, float2,
                                                 Dv2_f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_8_BUILTIN(3, uchar2, Dv2_h, float3,
                                                 Dv3_f)
// return 4-channel color data
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_8_BUILTIN(1, uchar4, Dv4_h, float, f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_8_BUILTIN(2, uchar4, Dv4_h, float2,
                                                 Dv2_f)
_CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_8_BUILTIN(3, uchar4, Dv4_h, float3,
                                                 Dv3_f)

#undef _CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_8_BUILTIN
#undef _CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_16_BUILTIN
#undef _CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_32_BUILTIN

#undef _CLC_DEFINE_SAMPLEDIMAGE_BINDLESS_READ_BUILTIN

#undef _CLC_DEFINE_IMAGE_BINDLESS_WRITE_BUILTIN
#undef _CLC_DEFINE_IMAGE_BINDLESS_FETCH_BUILTIN

#undef _CLC_CONST_AS
#undef _CLC_MANGLE_FUNC_IMG_HANDLE
