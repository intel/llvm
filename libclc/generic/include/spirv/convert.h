//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SPIRV_CONVERSIONS
#define SPIRV_CONVERSIONS

#define _SPIRV_CONVERT_DECL(FROM_TYPE, TO_TYPE, NAME, SUFFIX) \
  _CLC_OVERLOAD _CLC_DECL TO_TYPE NAME##_R##TO_TYPE##SUFFIX(FROM_TYPE x);

#define _SPIRV_VECTOR_CONVERT_DECL(FROM_TYPE, TO_TYPE, NAME, SUFFIX) \
  _SPIRV_CONVERT_DECL(FROM_TYPE, TO_TYPE, NAME, SUFFIX) \
  _SPIRV_CONVERT_DECL(FROM_TYPE##2, TO_TYPE##2, NAME, SUFFIX) \
  _SPIRV_CONVERT_DECL(FROM_TYPE##3, TO_TYPE##3, NAME, SUFFIX) \
  _SPIRV_CONVERT_DECL(FROM_TYPE##4, TO_TYPE##4, NAME, SUFFIX) \
  _SPIRV_CONVERT_DECL(FROM_TYPE##8, TO_TYPE##8, NAME, SUFFIX) \
  _SPIRV_CONVERT_DECL(FROM_TYPE##16, TO_TYPE##16, NAME, SUFFIX)

#define _SPIRV_VECTOR_CONVERT_TO_S(FROM_TYPE, NAME, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_DECL(FROM_TYPE, char, NAME, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_DECL(FROM_TYPE, int, NAME, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_DECL(FROM_TYPE, short, NAME, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_DECL(FROM_TYPE, long, NAME, SUFFIX)

#define _SPIRV_VECTOR_CONVERT_TO_U(FROM_TYPE, NAME, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_DECL(FROM_TYPE, uchar, NAME, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_DECL(FROM_TYPE, uint, NAME, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_DECL(FROM_TYPE, ushort, NAME, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_DECL(FROM_TYPE, ulong, NAME, SUFFIX)

#ifdef cl_khr_fp64
#define _SPIRV_VECTOR_CONVERT_TO_F(FROM_TYPE, NAME, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_DECL(FROM_TYPE, float, NAME, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_DECL(FROM_TYPE, double, NAME, SUFFIX)
#else
#define _SPIRV_VECTOR_CONVERT_TO_F(FROM_TYPE, NAME, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_DECL(FROM_TYPE, float, NAME, SUFFIX)
#endif

#define _SPIRV_VECTOR_CONVERT_TO_INNER(SUFFIX) \
  /* Conversions between signed. */ \
  _SPIRV_VECTOR_CONVERT_TO_S(char, __spirv_SConvert, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_TO_S(int, __spirv_SConvert, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_TO_S(short, __spirv_SConvert, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_TO_S(long, __spirv_SConvert, SUFFIX) \
  /* Conversions between unsigned. */ \
  _SPIRV_VECTOR_CONVERT_TO_U(uchar, __spirv_UConvert, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_TO_U(uint, __spirv_UConvert, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_TO_U(ushort, __spirv_UConvert, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_TO_U(ulong, __spirv_UConvert, SUFFIX) \
  /* Conversions between floats. */ \
  _SPIRV_VECTOR_CONVERT_TO_F(float, __spirv_FConvert, SUFFIX) \
  /* Conversions to float. */ \
  _SPIRV_VECTOR_CONVERT_TO_F(char, __spirv_ConvertSToF, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_TO_F(int, __spirv_ConvertSToF, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_TO_F(short, __spirv_ConvertSToF, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_TO_F(long, __spirv_ConvertSToF, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_TO_F(uchar, __spirv_ConvertUToF, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_TO_F(uint, __spirv_ConvertUToF, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_TO_F(ushort, __spirv_ConvertUToF, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_TO_F(ulong, __spirv_ConvertUToF, SUFFIX) \
  /* Conversions from float. */ \
  _SPIRV_VECTOR_CONVERT_TO_S(float, __spirv_ConvertFToS, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_TO_U(float, __spirv_ConvertFToU, SUFFIX) \
  /* Saturated conversions from signed to unsigned. */ \
  _SPIRV_VECTOR_CONVERT_TO_U(char, __spirv_SatConvertSToU, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_TO_U(int, __spirv_SatConvertSToU, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_TO_U(short, __spirv_SatConvertSToU, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_TO_U(long, __spirv_SatConvertSToU, SUFFIX) \
  /* Saturated conversions from unsigned to signed. */ \
  _SPIRV_VECTOR_CONVERT_TO_S(uchar, __spirv_SatConvertUToS, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_TO_S(uint, __spirv_SatConvertUToS, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_TO_S(ushort, __spirv_SatConvertUToS, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_TO_S(ulong, __spirv_SatConvertUToS, SUFFIX)

#ifdef cl_khr_fp64
#define _SPIRV_VECTOR_CONVERT_TO(SUFFIX) \
  _SPIRV_VECTOR_CONVERT_TO_INNER(SUFFIX) \
  _SPIRV_VECTOR_CONVERT_TO_F(double, __spirv_FConvert, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_TO_S(double, __spirv_ConvertFToS, SUFFIX) \
  _SPIRV_VECTOR_CONVERT_TO_U(double, __spirv_ConvertFToU, SUFFIX)
#else
#define _SPIRV_VECTOR_CONVERT_TO(SUFFIX) \
  _SPIRV_VECTOR_CONVERT_TO_INNER(SUFFIX)
#endif

_SPIRV_VECTOR_CONVERT_TO(_rtn)
_SPIRV_VECTOR_CONVERT_TO(_rte)
_SPIRV_VECTOR_CONVERT_TO(_rtz)
_SPIRV_VECTOR_CONVERT_TO(_rtp)
_SPIRV_VECTOR_CONVERT_TO()

#endif // SPIRV_CONVERSIONS
