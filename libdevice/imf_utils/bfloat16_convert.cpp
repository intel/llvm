//==-- bfloat16_convert.cpp - fallback implementation of bfloat16 to other type
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
#endif
