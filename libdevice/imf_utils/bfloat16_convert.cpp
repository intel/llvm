//==-- bfloat16_convert.cpp - fallback implementation of bfloat16 to other type
// convert--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../device.h"

#ifdef __LIBDEVICE_IMF_ENABLED__

#include "../device_imf.hpp"

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_bfloat162float(_iml_bf16_internal b) {
  return __bfloat162float(b);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_float2bfloat16(float f) {
  return __float2bfloat16(f, __IML_RTE);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_float2bfloat16_rd(float f) {
  return __float2bfloat16(f, __IML_RTN);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_float2bfloat16_rn(float f) {
  return __float2bfloat16(f, __IML_RTE);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_float2bfloat16_ru(float f) {
  return __float2bfloat16(f, __IML_RTP);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_float2bfloat16_rz(float f) {
  return __float2bfloat16(f, __IML_RTZ);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_bfloat162uint_rd(_iml_bf16_internal b) {
  return __iml_bfloat162integral_u<unsigned int>(b, __IML_RTN);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_bfloat162uint_rn(_iml_bf16_internal b) {
  return __iml_bfloat162integral_u<unsigned int>(b, __IML_RTE);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_bfloat162uint_ru(_iml_bf16_internal b) {
  return __iml_bfloat162integral_u<unsigned int>(b, __IML_RTP);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_bfloat162uint_rz(_iml_bf16_internal b) {
  return __iml_bfloat162integral_u<unsigned int>(b, __IML_RTZ);
}

DEVICE_EXTERN_C_INLINE
unsigned short __devicelib_imf_bfloat162ushort_rd(_iml_bf16_internal b) {
  return __iml_bfloat162integral_u<unsigned short>(b, __IML_RTN);
}

DEVICE_EXTERN_C_INLINE
unsigned short __devicelib_imf_bfloat162ushort_rn(_iml_bf16_internal b) {
  return __iml_bfloat162integral_u<unsigned short>(b, __IML_RTE);
}

DEVICE_EXTERN_C_INLINE
unsigned short __devicelib_imf_bfloat162ushort_ru(_iml_bf16_internal b) {
  return __iml_bfloat162integral_u<unsigned short>(b, __IML_RTP);
}

DEVICE_EXTERN_C_INLINE
unsigned short __devicelib_imf_bfloat162ushort_rz(_iml_bf16_internal b) {
  return __iml_bfloat162integral_u<unsigned short>(b, __IML_RTZ);
}

DEVICE_EXTERN_C_INLINE
unsigned long long __devicelib_imf_bfloat162ull_rd(_iml_bf16_internal b) {
  return __iml_bfloat162integral_u<unsigned long long>(b, __IML_RTN);
}

DEVICE_EXTERN_C_INLINE
unsigned long long __devicelib_imf_bfloat162ull_rn(_iml_bf16_internal b) {
  return __iml_bfloat162integral_u<unsigned long long>(b, __IML_RTE);
}

DEVICE_EXTERN_C_INLINE
unsigned long long __devicelib_imf_bfloat162ull_ru(_iml_bf16_internal b) {
  return __iml_bfloat162integral_u<unsigned long long>(b, __IML_RTP);
}

DEVICE_EXTERN_C_INLINE
unsigned long long __devicelib_imf_bfloat162ull_rz(_iml_bf16_internal b) {
  return __iml_bfloat162integral_u<unsigned long long>(b, __IML_RTZ);
}

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_bfloat162int_rd(_iml_bf16_internal b) {
  return __iml_bfloat162integral_s<int>(b, __IML_RTN);
}

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_bfloat162int_rn(_iml_bf16_internal b) {
  return __iml_bfloat162integral_s<int>(b, __IML_RTE);
}

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_bfloat162int_ru(_iml_bf16_internal b) {
  return __iml_bfloat162integral_s<int>(b, __IML_RTP);
}

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_bfloat162int_rz(_iml_bf16_internal b) {
  return __iml_bfloat162integral_s<int>(b, __IML_RTZ);
}

DEVICE_EXTERN_C_INLINE
short __devicelib_imf_bfloat162short_rd(_iml_bf16_internal b) {
  return __iml_bfloat162integral_s<short>(b, __IML_RTN);
}

DEVICE_EXTERN_C_INLINE
short __devicelib_imf_bfloat162short_rn(_iml_bf16_internal b) {
  return __iml_bfloat162integral_s<short>(b, __IML_RTE);
}

DEVICE_EXTERN_C_INLINE
short __devicelib_imf_bfloat162short_ru(_iml_bf16_internal b) {
  return __iml_bfloat162integral_s<short>(b, __IML_RTP);
}

DEVICE_EXTERN_C_INLINE
short __devicelib_imf_bfloat162short_rz(_iml_bf16_internal b) {
  return __iml_bfloat162integral_s<short>(b, __IML_RTZ);
}

DEVICE_EXTERN_C_INLINE
long long __devicelib_imf_bfloat162ll_rd(_iml_bf16_internal b) {
  return __iml_bfloat162integral_s<long long>(b, __IML_RTN);
}

DEVICE_EXTERN_C_INLINE
long long __devicelib_imf_bfloat162ll_rn(_iml_bf16_internal b) {
  return __iml_bfloat162integral_s<long long>(b, __IML_RTE);
}

DEVICE_EXTERN_C_INLINE
long long __devicelib_imf_bfloat162ll_ru(_iml_bf16_internal b) {
  return __iml_bfloat162integral_s<long long>(b, __IML_RTP);
}

DEVICE_EXTERN_C_INLINE
long long __devicelib_imf_bfloat162ll_rz(_iml_bf16_internal b) {
  return __iml_bfloat162integral_s<long long>(b, __IML_RTZ);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_ushort2bfloat16_rd(unsigned short x) {
  return __iml_integral2bfloat16_u<unsigned short>(x, __IML_RTN);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_ushort2bfloat16_rn(unsigned short x) {
  return __iml_integral2bfloat16_u<unsigned short>(x, __IML_RTE);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_ushort2bfloat16_ru(unsigned short x) {
  return __iml_integral2bfloat16_u<unsigned short>(x, __IML_RTP);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_ushort2bfloat16_rz(unsigned short x) {
  return __iml_integral2bfloat16_u<unsigned short>(x, __IML_RTZ);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_uint2bfloat16_rd(unsigned int x) {
  return __iml_integral2bfloat16_u<unsigned int>(x, __IML_RTN);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_uint2bfloat16_rn(unsigned int x) {
  return __iml_integral2bfloat16_u<unsigned int>(x, __IML_RTE);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_uint2bfloat16_ru(unsigned int x) {
  return __iml_integral2bfloat16_u<unsigned int>(x, __IML_RTP);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_uint2bfloat16_rz(unsigned int x) {
  return __iml_integral2bfloat16_u<unsigned int>(x, __IML_RTZ);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_ull2bfloat16_rd(unsigned long long x) {
  return __iml_integral2bfloat16_u<unsigned long long>(x, __IML_RTN);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_ull2bfloat16_rn(unsigned long long x) {
  return __iml_integral2bfloat16_u<unsigned long long>(x, __IML_RTE);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_ull2bfloat16_ru(unsigned long long x) {
  return __iml_integral2bfloat16_u<unsigned long long>(x, __IML_RTP);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_ull2bfloat16_rz(unsigned long long x) {
  return __iml_integral2bfloat16_u<unsigned long long>(x, __IML_RTZ);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_short2bfloat16_rd(short x) {
  return __iml_integral2bfloat16_s<short>(x, __IML_RTN);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_short2bfloat16_rn(short x) {
  return __iml_integral2bfloat16_s<short>(x, __IML_RTE);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_short2bfloat16_ru(short x) {
  return __iml_integral2bfloat16_s<short>(x, __IML_RTP);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_short2bfloat16_rz(short x) {
  return __iml_integral2bfloat16_s<short>(x, __IML_RTZ);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_int2bfloat16_rd(int x) {
  return __iml_integral2bfloat16_s<int>(x, __IML_RTN);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_int2bfloat16_rn(int x) {
  return __iml_integral2bfloat16_s<int>(x, __IML_RTE);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_int2bfloat16_ru(int x) {
  return __iml_integral2bfloat16_s<int>(x, __IML_RTP);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_int2bfloat16_rz(int x) {
  return __iml_integral2bfloat16_s<int>(x, __IML_RTZ);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_ll2bfloat16_rd(long long x) {
  return __iml_integral2bfloat16_s<long long>(x, __IML_RTN);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_ll2bfloat16_rn(long long x) {
  return __iml_integral2bfloat16_s<long long>(x, __IML_RTE);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_ll2bfloat16_ru(long long x) {
  return __iml_integral2bfloat16_s<long long>(x, __IML_RTP);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_ll2bfloat16_rz(long long x) {
  return __iml_integral2bfloat16_s<long long>(x, __IML_RTZ);
}

DEVICE_EXTERN_C_INLINE
short __devicelib_imf_bfloat16_as_short(_iml_bf16_internal b) {
  return __builtin_bit_cast(short, b);
}

DEVICE_EXTERN_C_INLINE
unsigned short __devicelib_imf_bfloat16_as_ushort(_iml_bf16_internal b) {
  return __builtin_bit_cast(unsigned short, b);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_short_as_bfloat16(short x) {
  return __builtin_bit_cast(_iml_bf16_internal, x);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_ushort_as_bfloat16(unsigned short x) {
  return __builtin_bit_cast(_iml_bf16_internal, x);
}
#endif
