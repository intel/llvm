//==-- half_convert.cpp - fallback implementation of half to other type
// convert--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../device_imf.hpp"

#ifdef __LIBDEVICE_IMF_ENABLED__

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_half2float(_iml_half_internal x) {
  return __half2float(x);
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_float2half_rd(float x) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_FConvert_Rhalf_rtn(x);
#else
  return __iml_fp2half(x, __IML_RTN);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_float2half_rn(float x) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_FConvert_Rhalf_rte(x);
#else
  return __iml_fp2half(x, __IML_RTE);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_float2half_ru(float x) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_FConvert_Rhalf_rtp(x);
#else
  return __iml_fp2half(x, __IML_RTP);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_float2half_rz(float x) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_FConvert_Rhalf_rtz(x);
#else
  return __iml_fp2half(x, __IML_RTZ);
#endif
}

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_half2int_rd(_iml_half_internal h) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToS_Rint_sat_rtn(h);
#else
  return __iml_half2integral_s<int>(h, __IML_RTN);
#endif
}

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_half2int_rn(_iml_half_internal h) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToS_Rint_sat_rte(h);
#else
  return __iml_half2integral_s<int>(h, __IML_RTE);
#endif
}

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_half2int_ru(_iml_half_internal h) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToS_Rint_sat_rtp(h);
#else
  return __iml_half2integral_s<int>(h, __IML_RTP);
#endif
}

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_half2int_rz(_iml_half_internal h) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToS_Rint_sat_rtz(h);
#else
  return __iml_half2integral_s<int>(h, __IML_RTZ);
#endif
}

DEVICE_EXTERN_C_INLINE
long long __devicelib_imf_half2ll_rd(_iml_half_internal h) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToS_Rlong_sat_rtn(h);
#else
  return __iml_half2integral_s<long long>(h, __IML_RTN);
#endif
}

DEVICE_EXTERN_C_INLINE
long long __devicelib_imf_half2ll_rn(_iml_half_internal h) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToS_Rlong_sat_rte(h);
#else
  return __iml_half2integral_s<long long>(h, __IML_RTE);
#endif
}

DEVICE_EXTERN_C_INLINE
long long __devicelib_imf_half2ll_ru(_iml_half_internal h) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToS_Rlong_sat_rtp(h);
#else
  return __iml_half2integral_s<long long>(h, __IML_RTP);
#endif
}

DEVICE_EXTERN_C_INLINE
long long __devicelib_imf_half2ll_rz(_iml_half_internal h) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToS_Rlong_sat_rtz(h);
#else
  return __iml_half2integral_s<long long>(h, __IML_RTZ);
#endif
}

DEVICE_EXTERN_C_INLINE
short __devicelib_imf_half2short_rd(_iml_half_internal h) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToS_Rshort_sat_rtn(h);
#else
  return __iml_half2integral_s<short>(h, __IML_RTN);
#endif
}

DEVICE_EXTERN_C_INLINE
short __devicelib_imf_half2short_rn(_iml_half_internal h) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToS_Rshort_sat_rte(h);
#else
  return __iml_half2integral_s<short>(h, __IML_RTE);
#endif
}

DEVICE_EXTERN_C_INLINE
short __devicelib_imf_half2short_ru(_iml_half_internal h) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToS_Rshort_sat_rtp(h);
#else
  return __iml_half2integral_s<short>(h, __IML_RTP);
#endif
}

DEVICE_EXTERN_C_INLINE
short __devicelib_imf_half2short_rz(_iml_half_internal h) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToS_Rshort_sat_rtz(h);
#else
  return __iml_half2integral_s<short>(h, __IML_RTZ);
#endif
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_half2uint_rd(_iml_half_internal h) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToU_Ruint_sat_rtn(h);
#else
  return __iml_half2integral_u<unsigned int>(h, __IML_RTN);
#endif
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_half2uint_rn(_iml_half_internal h) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToU_Ruint_sat_rte(h);
#else
  return __iml_half2integral_u<unsigned int>(h, __IML_RTE);
#endif
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_half2uint_ru(_iml_half_internal h) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToU_Ruint_sat_rtp(h);
#else
  return __iml_half2integral_u<unsigned int>(h, __IML_RTP);
#endif
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_half2uint_rz(_iml_half_internal h) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToU_Ruint_sat_rtz(h);
#else
  return __iml_half2integral_u<unsigned int>(h, __IML_RTZ);
#endif
}

DEVICE_EXTERN_C_INLINE
unsigned long long __devicelib_imf_half2ull_rd(_iml_half_internal h) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToU_Rulong_sat_rtn(h);
#else
  return __iml_half2integral_u<unsigned long long>(h, __IML_RTN);
#endif
}

DEVICE_EXTERN_C_INLINE
unsigned long long __devicelib_imf_half2ull_rn(_iml_half_internal h) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToU_Rulong_sat_rte(h);
#else
  return __iml_half2integral_u<unsigned long long>(h, __IML_RTE);
#endif
}

DEVICE_EXTERN_C_INLINE
unsigned long long __devicelib_imf_half2ull_ru(_iml_half_internal h) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToU_Rulong_sat_rtp(h);
#else
  return __iml_half2integral_u<unsigned long long>(h, __IML_RTP);
#endif
}

DEVICE_EXTERN_C_INLINE
unsigned long long __devicelib_imf_half2ull_rz(_iml_half_internal h) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToU_Rulong_sat_rtz(h);
#else
  return __iml_half2integral_u<unsigned long long>(h, __IML_RTZ);
#endif
}

DEVICE_EXTERN_C_INLINE
unsigned short __devicelib_imf_half2ushort_rd(_iml_half_internal h) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToU_Rushort_sat_rtn(h);
#else
  return __iml_half2integral_u<unsigned short>(h, __IML_RTN);
#endif
}

DEVICE_EXTERN_C_INLINE
unsigned short __devicelib_imf_half2ushort_rn(_iml_half_internal h) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToU_Rushort_sat_rte(h);
#else
  return __iml_half2integral_u<unsigned short>(h, __IML_RTE);
#endif
}

DEVICE_EXTERN_C_INLINE
unsigned short __devicelib_imf_half2ushort_ru(_iml_half_internal h) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToU_Rushort_sat_rtp(h);
#else
  return __iml_half2integral_u<unsigned short>(h, __IML_RTP);
#endif
}

DEVICE_EXTERN_C_INLINE
unsigned short __devicelib_imf_half2ushort_rz(_iml_half_internal h) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToU_Rushort_sat_rtz(h);
#else
  return __iml_half2integral_u<unsigned short>(h, __IML_RTZ);
#endif
}

DEVICE_EXTERN_C_INLINE
short __devicelib_imf_half_as_short(_iml_half_internal h) {
  return __builtin_bit_cast(short, h);
}

DEVICE_EXTERN_C_INLINE
unsigned short __devicelib_imf_half_as_ushort(_iml_half_internal h) {
  return __builtin_bit_cast(unsigned short, h);
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_int2half_rd(int x) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertSToF_Rhalf_rtn(x);
#else
  return __iml_integral2half_s<int>(x, __IML_RTN);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_int2half_rn(int x) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertSToF_Rhalf_rte(x);
#else
  return __iml_integral2half_s<int>(x, __IML_RTE);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_int2half_ru(int x) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertSToF_Rhalf_rtp(x);
#else
  return __iml_integral2half_s<int>(x, __IML_RTP);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_int2half_rz(int x) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertSToF_Rhalf_rtz(x);
#else
  return __iml_integral2half_s<int>(x, __IML_RTZ);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_ll2half_rd(long long x) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertSToF_Rhalf_rtn((int64_t)x);
#else
  return __iml_integral2half_s<long long>(x, __IML_RTN);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_ll2half_rn(long long x) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertSToF_Rhalf_rte((int64_t)x);
#else
  return __iml_integral2half_s<long long>(x, __IML_RTE);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_ll2half_ru(long long x) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertSToF_Rhalf_rtp((int64_t)x);
#else
  return __iml_integral2half_s<long long>(x, __IML_RTP);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_ll2half_rz(long long x) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertSToF_Rhalf_rtz((int64_t)x);
#else
  return __iml_integral2half_s<long long>(x, __IML_RTZ);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_short2half_rd(short x) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertSToF_Rhalf_rtn(x);
#else
  return __iml_integral2half_s<short>(x, __IML_RTN);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_short2half_rn(short x) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertSToF_Rhalf_rte(x);
#else
  return __iml_integral2half_s<short>(x, __IML_RTE);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_short2half_ru(short x) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertSToF_Rhalf_rtp(x);
#else
  return __iml_integral2half_s<short>(x, __IML_RTP);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_short2half_rz(short x) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertSToF_Rhalf_rtz(x);
#else
  return __iml_integral2half_s<short>(x, __IML_RTZ);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_short_as_half(short x) {
  return __builtin_bit_cast(_iml_half_internal, x);
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_uint2half_rd(unsigned int x) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertUToF_Rhalf_rtn(x);
#else
  return __iml_integral2half_u<unsigned int>(x, __IML_RTN);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_uint2half_rn(unsigned int x) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertUToF_Rhalf_rte(x);
#else
  return __iml_integral2half_u<unsigned int>(x, __IML_RTE);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_uint2half_ru(unsigned int x) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertUToF_Rhalf_rtp(x);
#else
  return __iml_integral2half_u<unsigned int>(x, __IML_RTP);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_uint2half_rz(unsigned int x) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertUToF_Rhalf_rtz(x);
#else
  return __iml_integral2half_u<unsigned int>(x, __IML_RTZ);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_ull2half_rd(unsigned long long x) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertUToF_Rhalf_rtn((uint64_t)x);
#else
  return __iml_integral2half_u<unsigned long long>(x, __IML_RTN);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_ull2half_rn(unsigned long long x) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertUToF_Rhalf_rte((uint64_t)x);
#else
  return __iml_integral2half_u<unsigned long long>(x, __IML_RTE);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_ull2half_ru(unsigned long long x) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertUToF_Rhalf_rtp((uint64_t)x);
#else
  return __iml_integral2half_u<unsigned long long>(x, __IML_RTP);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_ull2half_rz(unsigned long long x) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertUToF_Rhalf_rtz((uint64_t)x);
#else
  return __iml_integral2half_u<unsigned long long>(x, __IML_RTZ);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_ushort2half_rd(unsigned short x) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertUToF_Rhalf_rtn(x);
#else
  return __iml_integral2half_u<unsigned short>(x, __IML_RTN);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_ushort2half_rn(unsigned short x) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertUToF_Rhalf_rte(x);
#else
  return __iml_integral2half_u<unsigned short>(x, __IML_RTE);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_ushort2half_ru(unsigned short x) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertUToF_Rhalf_rtp(x);
#else
  return __iml_integral2half_u<unsigned short>(x, __IML_RTP);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_ushort2half_rz(unsigned short x) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertUToF_Rhalf_rtz(x);
#else
  return __iml_integral2half_u<unsigned short>(x, __IML_RTZ);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_ushort_as_half(unsigned short x) {
  return __builtin_bit_cast(_iml_half_internal, x);
}
#endif // __LIBDEVICE_IMF_ENABLED__
