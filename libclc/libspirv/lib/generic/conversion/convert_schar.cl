//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

// Rschar variants

// The __spirv_*_Rchar* conversion functions are only forward-declared here.
// Their implementations are provided in other files within this folder.
// Eventually, these declarations should come from clang in the future.
// The forward-declarations here are for use within this file's _Rschar*
// implementations.
// Each Rschar function implementation is paired with its corresponding Rchar
// forward declaration directly, because char is signed in OpenCL and Rschar
// can re-use Rchar implementation.
// This approach makes the file self-contained, so _Rschar* functions don't need
// to be distributed in other files in this folder.

#define GENERATE_CONVERSIONS_FUNCTIONS_SCHAR(__func, __argtype, __sat,         \
                                             __rounding)                       \
  _CLC_OVERLOAD _CLC_DECL char __spirv_##__func##_Rchar##__sat##__rounding(    \
      __argtype x);                                                            \
  _CLC_OVERLOAD _CLC_DEF __clc_schar_t                                         \
  __spirv_##__func##_Rschar##__sat##__rounding(__argtype x) {                  \
    return (__clc_schar_t)__spirv_##__func##_Rchar##__sat##__rounding(x);      \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DECL char2 __spirv_##__func##_Rchar2##__sat##__rounding(  \
      __argtype##2 x);                                                         \
  _CLC_OVERLOAD _CLC_DEF __clc_vec2_schar_t                                    \
  __spirv_##__func##_Rschar2##__sat##__rounding(__argtype##2 x) {              \
    return __builtin_convertvector(                                            \
        __spirv_##__func##_Rchar2##__sat##__rounding(x), __clc_vec2_schar_t);  \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DECL char3 __spirv_##__func##_Rchar3##__sat##__rounding(  \
      __argtype##3 x);                                                         \
  _CLC_OVERLOAD _CLC_DEF __clc_vec3_schar_t                                    \
  __spirv_##__func##_Rschar3##__sat##__rounding(__argtype##3 x) {              \
    return __builtin_convertvector(                                            \
        __spirv_##__func##_Rchar3##__sat##__rounding(x), __clc_vec3_schar_t);  \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DECL char4 __spirv_##__func##_Rchar4##__sat##__rounding(  \
      __argtype##4 x);                                                         \
  _CLC_OVERLOAD _CLC_DEF __clc_vec4_schar_t                                    \
  __spirv_##__func##_Rschar4##__sat##__rounding(__argtype##4 x) {              \
    return __builtin_convertvector(                                            \
        __spirv_##__func##_Rchar4##__sat##__rounding(x), __clc_vec4_schar_t);  \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DECL char8 __spirv_##__func##_Rchar8##__sat##__rounding(  \
      __argtype##8 x);                                                         \
  _CLC_OVERLOAD _CLC_DEF __clc_vec8_schar_t                                    \
  __spirv_##__func##_Rschar8##__sat##__rounding(__argtype##8 x) {              \
    return __builtin_convertvector(                                            \
        __spirv_##__func##_Rchar8##__sat##__rounding(x), __clc_vec8_schar_t);  \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DECL char16                                               \
      __spirv_##__func##_Rchar16##__sat##__rounding(__argtype##16 x);          \
  _CLC_OVERLOAD _CLC_DEF __clc_vec16_schar_t                                   \
  __spirv_##__func##_Rschar16##__sat##__rounding(__argtype##16 x) {            \
    return __builtin_convertvector(                                            \
        __spirv_##__func##_Rchar16##__sat##__rounding(x),                      \
        __clc_vec16_schar_t);                                                  \
  }

#define GENERATE_CONVERSIONS_FUNCTIONS_ROUNDINGS_SCHAR(__func, __argtype)      \
  GENERATE_CONVERSIONS_FUNCTIONS_SCHAR(__func, __argtype, , )                  \
  GENERATE_CONVERSIONS_FUNCTIONS_SCHAR(__func, __argtype, , _rte)              \
  GENERATE_CONVERSIONS_FUNCTIONS_SCHAR(__func, __argtype, , _rtz)              \
  GENERATE_CONVERSIONS_FUNCTIONS_SCHAR(__func, __argtype, , _rtp)              \
  GENERATE_CONVERSIONS_FUNCTIONS_SCHAR(__func, __argtype, , _rtn)              \
  GENERATE_CONVERSIONS_FUNCTIONS_SCHAR(__func, __argtype, _sat, )              \
  GENERATE_CONVERSIONS_FUNCTIONS_SCHAR(__func, __argtype, _sat, _rte)          \
  GENERATE_CONVERSIONS_FUNCTIONS_SCHAR(__func, __argtype, _sat, _rtz)          \
  GENERATE_CONVERSIONS_FUNCTIONS_SCHAR(__func, __argtype, _sat, _rtp)          \
  GENERATE_CONVERSIONS_FUNCTIONS_SCHAR(__func, __argtype, _sat, _rtn)

GENERATE_CONVERSIONS_FUNCTIONS_ROUNDINGS_SCHAR(SConvert, char)
GENERATE_CONVERSIONS_FUNCTIONS_ROUNDINGS_SCHAR(SConvert, short)
GENERATE_CONVERSIONS_FUNCTIONS_ROUNDINGS_SCHAR(SConvert, int)
GENERATE_CONVERSIONS_FUNCTIONS_ROUNDINGS_SCHAR(SConvert, long)
GENERATE_CONVERSIONS_FUNCTIONS_SCHAR(SatConvertUToS, uchar, , )
GENERATE_CONVERSIONS_FUNCTIONS_SCHAR(SatConvertUToS, ushort, , )
GENERATE_CONVERSIONS_FUNCTIONS_SCHAR(SatConvertUToS, uint, , )
GENERATE_CONVERSIONS_FUNCTIONS_SCHAR(SatConvertUToS, ulong, , )

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
GENERATE_CONVERSIONS_FUNCTIONS_ROUNDINGS_SCHAR(ConvertFToS, half)
#endif // cl_khr_fp16
GENERATE_CONVERSIONS_FUNCTIONS_ROUNDINGS_SCHAR(ConvertFToS, float)
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
GENERATE_CONVERSIONS_FUNCTIONS_ROUNDINGS_SCHAR(ConvertFToS, double)
#endif // cl_khr_fp64
