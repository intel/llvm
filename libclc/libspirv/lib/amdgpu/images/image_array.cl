#include "image_common.h"
#include <libspirv/spirv.h>

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define _CLC_ARRAY_COORD_PARAMS_1D(coord, layer) coord, layer
#define _CLC_ARRAY_COORD_PARAMS_2D(coord, layer) coord.x, coord.y, layer, 0

// Declare ockl functions/builtins that we link from the ROCm device libs.
float4 __ockl_image_load_1Da(_CLC_CONST_AS unsigned int *tex, int2 coord);
float4 __ockl_image_load_2Da(_CLC_CONST_AS unsigned int *tex, int4 coord);
half4 __ockl_image_loadh_1Da(_CLC_CONST_AS unsigned int *tex, int2 coord);
half4 __ockl_image_loadh_2Da(_CLC_CONST_AS unsigned int *tex, int4 coord);

void __ockl_image_store_1Da(_CLC_CONST_AS unsigned int *tex, int2 coord,
                            float4 color);
void __ockl_image_store_2Da(_CLC_CONST_AS unsigned int *tex, int4 coord,
                            float4 color);
void __ockl_image_storeh_1Da(_CLC_CONST_AS unsigned int *tex, int2 coord,
                             half4 color);
void __ockl_image_storeh_2Da(_CLC_CONST_AS unsigned int *tex, int4 coord,
                             half4 color);

float4 __ockl_image_sample_1Da(_CLC_CONST_AS unsigned int *tex,
                               _CLC_CONST_AS unsigned int *samp, float2 coord);
float4 __ockl_image_sample_2Da(_CLC_CONST_AS unsigned int *tex,
                               _CLC_CONST_AS unsigned int *samp, float4 coord);
half4 __ockl_image_sampleh_1Da(_CLC_CONST_AS unsigned int *tex,
                               _CLC_CONST_AS unsigned int *samp, float2 coord);
half4 __ockl_image_sampleh_2Da(_CLC_CONST_AS unsigned int *tex,
                               _CLC_CONST_AS unsigned int *samp, float4 coord);

//
// IMAGE ARRAYS
//

// Read Ops

#define _CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_BUILTIN(                        \
    dimension, elem_t, elem_t_mangled, coord_t, coord_mangled, vec_size,       \
    builtin_ret_t, builtin_ret_postfix)                                        \
  _CLC_DEF elem_t _CLC_MANGLE_FUNC_IMG_HANDLE(23, __spirv_ImageArrayFetch,     \
                                              I##elem_t_mangled,               \
                                              coord_mangled##ET_T0_T1_i)(      \
      ulong imageHandle, coord_t coord, int layer) {                           \
    _CLC_CONST_AS unsigned int *tex =                                          \
        (_CLC_CONST_AS unsigned int *)imageHandle;                             \
    int##vec_size arrayCoord =                                                 \
        (int##vec_size)(_CLC_ARRAY_COORD_PARAMS_##dimension##D(coord, layer)); \
    builtin_ret_t##4 color =                                                   \
        __ockl_image_load##builtin_ret_postfix##_##dimension##Da(tex,          \
                                                                 arrayCoord);  \
    return __clc_cast_from_##builtin_ret_t##4_to_##elem_t(color);              \
  }

#define _CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_32_BUILTIN(                     \
    dimension, elem_t, elem_t_mangled, coord_t, coord_mangled, vec_size)       \
  _CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_BUILTIN(                              \
      dimension, elem_t, elem_t_mangled, coord_t, coord_mangled, vec_size,     \
      float, )

#define _CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_16_BUILTIN(                     \
    dimension, elem_t, elem_t_mangled, coord_t, coord_mangled, vec_size)       \
  _CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_BUILTIN(                              \
      dimension, elem_t, elem_t_mangled, coord_t, coord_mangled, vec_size,     \
      half, h)

#define _CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_8_BUILTIN(                      \
    dimension, elem_t, elem_t_mangled, coord_t, coord_mangled, vec_size)       \
  _CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_16_BUILTIN(                           \
      dimension, elem_t, elem_t_mangled, coord_t, coord_mangled, vec_size)

// Float
// return 1-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_32_BUILTIN(1, float, f, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_32_BUILTIN(2, float, f, int2, Dv2_i, 4)
// return 2-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_32_BUILTIN(1, float2, Dv2_f, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_32_BUILTIN(2, float2, Dv2_f, int2, Dv2_i,
                                                  4)
// return 4-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_32_BUILTIN(1, float4, Dv4_f, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_32_BUILTIN(2, float4, Dv4_f, int2, Dv2_i,
                                                  4)

// Half
#ifdef cl_khr_fp16
// return 1-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_16_BUILTIN(1, half, Dh, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_16_BUILTIN(2, half, Dh, int2, Dv2_i,
                                                  4)
// return 2-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_16_BUILTIN(1, half2, Dv2_Dh, int, i,
                                                  2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_16_BUILTIN(2, half2, Dv2_Dh, int2,
                                                  Dv2_i, 4)
// return 4-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_16_BUILTIN(1, half4, Dv4_Dh, int, i,
                                                  2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_16_BUILTIN(2, half4, Dv4_Dh, int2,
                                                  Dv2_i, 4)
#endif

// Int
// return 1-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_32_BUILTIN(1, int, i, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_32_BUILTIN(2, int, i, int2, Dv2_i, 4)
// return 2-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_32_BUILTIN(1, int2, Dv2_i, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_32_BUILTIN(2, int2, Dv2_i, int2, Dv2_i,
                                                  4)
// return 4-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_32_BUILTIN(1, int4, Dv4_i, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_32_BUILTIN(2, int4, Dv4_i, int2, Dv2_i,
                                                  4)

// Unsigned Int
// return 1-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_32_BUILTIN(1, uint, j, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_32_BUILTIN(2, uint, j, int2, Dv2_i, 4)
// return 2-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_32_BUILTIN(1, uint2, Dv2_j, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_32_BUILTIN(2, uint2, Dv2_j, int2, Dv2_i,
                                                  4)
// return 4-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_32_BUILTIN(1, uint4, Dv4_j, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_32_BUILTIN(2, uint4, Dv4_j, int2, Dv2_i,
                                                  4)

// Short
// return 1-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_16_BUILTIN(1, short, s, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_16_BUILTIN(2, short, s, int2, Dv2_i, 4)
// return 2-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_16_BUILTIN(1, short2, Dv2_s, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_16_BUILTIN(2, short2, Dv2_s, int2, Dv2_i,
                                                  4)
// return 4-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_16_BUILTIN(1, short4, Dv4_s, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_16_BUILTIN(2, short4, Dv4_s, int2, Dv2_i,
                                                  4)

// Unsigned Short
// return 1-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_16_BUILTIN(1, ushort, t, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_16_BUILTIN(2, ushort, t, int2, Dv2_i, 4)
// return 2-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_16_BUILTIN(1, ushort2, Dv2_t, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_16_BUILTIN(2, ushort2, Dv2_t, int2,
                                                  Dv2_i, 4)
// return 4-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_16_BUILTIN(1, ushort4, Dv4_t, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_16_BUILTIN(2, ushort4, Dv4_t, int2,
                                                  Dv2_i, 4)

// Char
// return 1-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_8_BUILTIN(1, char, a, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_8_BUILTIN(2, char, a, int2, Dv2_i, 4)
// return 2-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_8_BUILTIN(1, char2, Dv2_a, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_8_BUILTIN(2, char2, Dv2_a, int2, Dv2_i,
                                                 4)
// return 4-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_8_BUILTIN(1, char4, Dv4_a, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_8_BUILTIN(2, char4, Dv4_a, int2, Dv2_i,
                                                 4)

// Unsigned Char
// return 1-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_8_BUILTIN(1, uchar, h, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_8_BUILTIN(2, uchar, h, int2, Dv2_i, 4)
// return 2-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_8_BUILTIN(1, uchar2, Dv2_h, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_8_BUILTIN(2, uchar2, Dv2_h, int2, Dv2_i,
                                                 4)
// return 4-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_8_BUILTIN(1, uchar4, Dv4_h, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_8_BUILTIN(2, uchar4, Dv4_h, int2, Dv2_i,
                                                 4)

#undef _CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_8_BUILTIN
#undef _CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_16_BUILTIN
#undef _CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_32_BUILTIN

// Write Ops

#define _CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_BUILTIN(                        \
    dimension, elem_t, elem_t_mangled, coord_t, coord_mangled, vec_size,       \
    builtin_ret_t, builtin_ret_postfix)                                        \
  _CLC_DEF void _CLC_MANGLE_FUNC_IMG_HANDLE(                                   \
      23, __spirv_ImageArrayWrite, I,                                          \
      coord_mangled##elem_t_mangled##EvT_T0_iT1_)(                             \
      ulong imageHandle, coord_t coord, int layer, elem_t color) {             \
    _CLC_CONST_AS unsigned int *tex =                                          \
        (_CLC_CONST_AS unsigned int *)imageHandle;                             \
    int##vec_size arrayCoord =                                                 \
        (int##vec_size)(_CLC_ARRAY_COORD_PARAMS_##dimension##D(coord, layer)); \
    builtin_ret_t##4 outColor =                                                \
        __clc_cast_from_##elem_t##_to_##builtin_ret_t##4(color);               \
    __ockl_image_store##builtin_ret_postfix##_##dimension##Da(tex, arrayCoord, \
                                                              outColor);       \
  }

#define _CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_32_BUILTIN(                     \
    dimension, elem_t, elem_t_mangled, coord_t, coord_mangled, vec_size)       \
  _CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_BUILTIN(                              \
      dimension, elem_t, elem_t_mangled, coord_t, coord_mangled, vec_size,     \
      float, )

#define _CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_16_BUILTIN(                     \
    dimension, elem_t, elem_t_mangled, coord_t, coord_mangled, vec_size)       \
  _CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_BUILTIN(                              \
      dimension, elem_t, elem_t_mangled, coord_t, coord_mangled, vec_size,     \
      half, h)

#define _CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_8_BUILTIN(                      \
    dimension, elem_t, elem_t_mangled, coord_t, coord_mangled, vec_size)       \
  _CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_16_BUILTIN(                           \
      dimension, elem_t, elem_t_mangled, coord_t, coord_mangled, vec_size)

// Float
// write 1-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_32_BUILTIN(1, float, f, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_32_BUILTIN(2, float, f, int2, Dv2_i, 4)
// write 2-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_32_BUILTIN(1, float2, Dv2_f, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_32_BUILTIN(2, float2, Dv2_f, int2, Dv2_i,
                                                  4)
// write 4-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_32_BUILTIN(1, float4, Dv4_f, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_32_BUILTIN(2, float4, Dv4_f, int2, Dv2_i,
                                                  4)

// Half
#ifdef cl_khr_fp16
// write 1-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_16_BUILTIN(1, half, Dh, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_16_BUILTIN(2, half, Dh, int2, Dv2_i,
                                                  4)
// write 2-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_16_BUILTIN(1, half2, Dv2_Dh, int, i,
                                                  2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_16_BUILTIN(2, half2, Dv2_Dh, int2,
                                                  Dv2_i, 4)
// write 4-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_16_BUILTIN(1, half4, Dv4_Dh, int, i,
                                                  2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_16_BUILTIN(2, half4, Dv4_Dh, int2,
                                                  Dv2_i, 4)
#endif

// Int
// write 1-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_32_BUILTIN(1, int, i, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_32_BUILTIN(2, int, i, int2, Dv2_i, 4)
// write 2-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_32_BUILTIN(1, int2, Dv2_i, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_32_BUILTIN(2, int2, Dv2_i, int2, Dv2_i,
                                                  4)
// write 4-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_32_BUILTIN(1, int4, Dv4_i, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_32_BUILTIN(2, int4, Dv4_i, int2, Dv2_i,
                                                  4)

// Unsigned Int
// write 1-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_32_BUILTIN(1, uint, j, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_32_BUILTIN(2, uint, j, int2, Dv2_i, 4)
// write 2-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_32_BUILTIN(1, uint2, Dv2_j, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_32_BUILTIN(2, uint2, Dv2_j, int2, Dv2_i,
                                                  4)
// write 4-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_32_BUILTIN(1, uint4, Dv4_j, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_32_BUILTIN(2, uint4, Dv4_j, int2, Dv2_i,
                                                  4)

// Short
// write 1-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_16_BUILTIN(1, short, s, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_16_BUILTIN(2, short, s, int2, Dv2_i, 4)
// write 2-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_16_BUILTIN(1, short2, Dv2_s, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_16_BUILTIN(2, short2, Dv2_s, int2, Dv2_i,
                                                  4)
// write 4-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_16_BUILTIN(1, short4, Dv4_s, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_16_BUILTIN(2, short4, Dv4_s, int2, Dv2_i,
                                                  4)

// Unsigned Short
// write 1-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_16_BUILTIN(1, ushort, t, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_16_BUILTIN(2, ushort, t, int2, Dv2_i, 4)
// write 2-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_16_BUILTIN(1, ushort2, Dv2_t, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_16_BUILTIN(2, ushort2, Dv2_t, int2,
                                                  Dv2_i, 4)
// write 4-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_16_BUILTIN(1, ushort4, Dv4_t, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_16_BUILTIN(2, ushort4, Dv4_t, int2,
                                                  Dv2_i, 4)

// Char
// write 1-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_8_BUILTIN(1, char, a, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_8_BUILTIN(2, char, a, int2, Dv2_i, 4)
// write 2-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_8_BUILTIN(1, char2, Dv2_a, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_8_BUILTIN(2, char2, Dv2_a, int2, Dv2_i,
                                                 4)
// write 4-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_8_BUILTIN(1, char4, Dv4_a, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_8_BUILTIN(2, char4, Dv4_a, int2, Dv2_i,
                                                 4)

// Unsigned Char
// write 1-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_8_BUILTIN(1, uchar, h, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_8_BUILTIN(2, uchar, h, int2, Dv2_i, 4)
// write 2-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_8_BUILTIN(1, uchar2, Dv2_h, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_8_BUILTIN(2, uchar2, Dv2_h, int2, Dv2_i,
                                                 4)
// write 4-channel color data
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_8_BUILTIN(1, uchar4, Dv4_h, int, i, 2)
_CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_8_BUILTIN(2, uchar4, Dv4_h, int2, Dv2_i,
                                                 4)

#undef _CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_8_BUILTIN
#undef _CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_16_BUILTIN
#undef _CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_32_BUILTIN

//
// SAMPLED IMAGE ARRAYS
//

// Read Ops

#define _CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_BUILTIN(                  \
    dimension, elem_t, elem_t_mangled, coord_t, coord_mangled, vec_size,       \
    builtin_ret_t, builtin_ret_postfix)                                        \
  _CLC_DEF elem_t _CLC_MANGLE_FUNC_IMG_HANDLE(22, __spirv_ImageArrayRead,      \
                                              I##elem_t_mangled,               \
                                              coord_mangled##ET_T0_T1_i)(      \
      ulong imageHandle, coord_t coord, int layer) {                           \
    _CLC_CONST_AS unsigned int *tex =                                          \
        (_CLC_CONST_AS unsigned int *)imageHandle;                             \
    _CLC_CONST_AS unsigned int *samp = tex + SAMPLER_OBJECT_OFFSET_DWORD;      \
    float##vec_size arrayCoord = (float##vec_size)(                            \
        _CLC_ARRAY_COORD_PARAMS_##dimension##D(coord, (float)layer));          \
    builtin_ret_t##4 color =                                                   \
        __ockl_image_sample##builtin_ret_postfix##_##dimension##Da(            \
            tex, samp, arrayCoord);                                            \
    return __clc_cast_from_##builtin_ret_t##4_to_##elem_t(color);              \
  }

#define _CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_32_BUILTIN(               \
    dimension, elem_t, elem_t_mangled, coord_t, coord_mangled, vec_size)       \
  _CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_BUILTIN(                        \
      dimension, elem_t, elem_t_mangled, coord_t, coord_mangled, vec_size,     \
      float, )

#define _CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_16_BUILTIN(               \
    dimension, elem_t, elem_t_mangled, coord_t, coord_mangled, vec_size)       \
  _CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_BUILTIN(                        \
      dimension, elem_t, elem_t_mangled, coord_t, coord_mangled, vec_size,     \
      half, h)

#define _CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_8_BUILTIN(                \
    dimension, elem_t, elem_t_mangled, coord_t, coord_mangled, vec_size)       \
  _CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_16_BUILTIN(                     \
      dimension, elem_t, elem_t_mangled, coord_t, coord_mangled, vec_size)

// Float
// return 1 channel color data
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_32_BUILTIN(1, float, f, float, f,
                                                        2)
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_32_BUILTIN(2, float, f, float2,
                                                        Dv2_f, 4)
// return 2 channel color data
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_32_BUILTIN(1, float2, Dv2_f, float,
                                                        f, 2)
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_32_BUILTIN(2, float2, Dv2_f,
                                                        float2, Dv2_f, 4)
// return 4 channel color data
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_32_BUILTIN(1, float4, Dv4_f, float,
                                                        f, 2)
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_32_BUILTIN(2, float4, Dv4_f,
                                                        float2, Dv2_f, 4)

// Half
#ifdef cl_khr_fp16
// return 1 channel color data
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_16_BUILTIN(1, half, Dh, float,
                                                        f, 2)
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_16_BUILTIN(2, half, Dh, float2,
                                                        Dv2_f, 4)
// return 2 channel color data
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_16_BUILTIN(1, half2, Dv2_Dh,
                                                        float, f, 2)
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_16_BUILTIN(2, half2, Dv2_Dh,
                                                        float2, Dv2_f, 4)
// return 4 channel color data
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_16_BUILTIN(1, half4, Dv4_Dh,
                                                        float, f, 2)
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_16_BUILTIN(2, half4, Dv4_Dh,
                                                        float2, Dv2_f, 4)
#endif

// Int
// return 1 channel color data
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_32_BUILTIN(1, int, i, float, f, 2)
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_32_BUILTIN(2, int, i, float2,
                                                        Dv2_f, 4)
// return 2 channel color data
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_32_BUILTIN(1, int2, Dv2_i, float,
                                                        f, 2)
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_32_BUILTIN(2, int2, Dv2_i, float2,
                                                        Dv2_f, 4)
// return 4 channel color data
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_32_BUILTIN(1, int4, Dv4_i, float,
                                                        f, 2)
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_32_BUILTIN(2, int4, Dv4_i, float2,
                                                        Dv2_f, 4)

// Unsigned Int
// return 1 channel color data
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_32_BUILTIN(1, uint, j, float, f, 2)
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_32_BUILTIN(2, uint, j, float2,
                                                        Dv2_f, 4)
// return 2 channel color data
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_32_BUILTIN(1, uint2, Dv2_j, float,
                                                        f, 2)
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_32_BUILTIN(2, uint2, Dv2_j, float2,
                                                        Dv2_f, 4)
// return 4 channel color data
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_32_BUILTIN(1, uint4, Dv4_j, float,
                                                        f, 2)
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_32_BUILTIN(2, uint4, Dv4_j, float2,
                                                        Dv2_f, 4)

// Short
// return 1 channel color data
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_16_BUILTIN(1, short, s, float, f,
                                                        2)
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_16_BUILTIN(2, short, s, float2,
                                                        Dv2_f, 4)
// return 2 channel color data
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_16_BUILTIN(1, short2, Dv2_s, float,
                                                        f, 2)
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_16_BUILTIN(2, short2, Dv2_s,
                                                        float2, Dv2_f, 4)
// return 4 channel color data
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_16_BUILTIN(1, short4, Dv4_s, float,
                                                        f, 2)
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_16_BUILTIN(2, short4, Dv4_s,
                                                        float2, Dv2_f, 4)

// Unsigned Short
// return 1 channel color data
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_16_BUILTIN(1, ushort, t, float, f,
                                                        2)
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_16_BUILTIN(2, ushort, t, float2,
                                                        Dv2_f, 4)
// return 2 channel color data
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_16_BUILTIN(1, ushort2, Dv2_t,
                                                        float, f, 2)
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_16_BUILTIN(2, ushort2, Dv2_t,
                                                        float2, Dv2_f, 4)
// return 4 channel color data
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_16_BUILTIN(1, ushort4, Dv4_t,
                                                        float, f, 2)
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_16_BUILTIN(2, ushort4, Dv4_t,
                                                        float2, Dv2_f, 4)

// Char
// return 1 channel color data
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_8_BUILTIN(1, char, a, float, f, 2)
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_8_BUILTIN(2, char, a, float2,
                                                       Dv2_f, 4)
// return 2 channel color data
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_8_BUILTIN(1, char2, Dv2_a, float,
                                                       f, 2)
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_8_BUILTIN(2, char2, Dv2_a, float2,
                                                       Dv2_f, 4)
// return 4 channel color data
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_8_BUILTIN(1, char4, Dv4_a, float,
                                                       f, 2)
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_8_BUILTIN(2, char4, Dv4_a, float2,
                                                       Dv2_f, 4)

// Unsigned Char
// return 1 channel color data
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_8_BUILTIN(1, uchar, h, float, f, 2)
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_8_BUILTIN(2, uchar, h, float2,
                                                       Dv2_f, 4)
// return 2 channel color data
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_8_BUILTIN(1, uchar2, Dv2_h, float,
                                                       f, 2)
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_8_BUILTIN(2, uchar2, Dv2_h, float2,
                                                       Dv2_f, 4)
// return 4 channel color data
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_8_BUILTIN(1, uchar4, Dv4_h, float,
                                                       f, 2)
_CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_8_BUILTIN(2, uchar4, Dv4_h, float2,
                                                       Dv2_f, 4)

#undef _CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_8_BUILTIN
#undef _CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_16_BUILTIN
#undef _CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_32_BUILTIN

#undef _CLC_DEFINE_SAMPLEDIMAGE_ARRAY_BINDLESS_READ_BUILTIN

#undef _CLC_DEFINE_IMAGE_ARRAY_BINDLESS_WRITE_BUILTIN
#undef _CLC_DEFINE_IMAGE_ARRAY_BINDLESS_FETCH_BUILTIN

#undef _CLC_ARRAY_COORD_PARAMS_1D
#undef _CLC_ARRAY_COORD_PARAMS_2D

#undef _CLC_CONST_AS
#undef _CLC_MANGLE_FUNC_IMG_HANDLE
