//==----- imf_wrapper_bf16.cpp - wrappers for BFloat16 intel math library
// functions ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "device.h"

#ifdef __LIBDEVICE_IMF_ENABLED__

#include "imf_bf16.hpp"

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_bfloat162float(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
float __imf_bfloat162float(_iml_bf16_internal b) {
  return __devicelib_imf_bfloat162float(b);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_float2bfloat16(float);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_float2bfloat16(float f) {
  return __devicelib_imf_float2bfloat16(f);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_float2bfloat16_rd(float);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_float2bfloat16_rd(float f) {
  return __devicelib_imf_float2bfloat16_rd(f);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_float2bfloat16_rn(float);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_float2bfloat16_rn(float f) {
  return __devicelib_imf_float2bfloat16_rn(f);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_float2bfloat16_ru(float);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_float2bfloat16_ru(float f) {
  return __devicelib_imf_float2bfloat16_ru(f);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_float2bfloat16_rz(float);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_float2bfloat16_rz(float f) {
  return __devicelib_imf_float2bfloat16_rz(f);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_bfloat162uint_rd(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
unsigned int __imf_bfloat162uint_rd(_iml_bf16_internal b) {
  return __devicelib_imf_bfloat162uint_rd(b);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_bfloat162uint_rn(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
unsigned int __imf_bfloat162uint_rn(_iml_bf16_internal b) {
  return __devicelib_imf_bfloat162uint_rn(b);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_bfloat162uint_ru(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
unsigned int __imf_bfloat162uint_ru(_iml_bf16_internal b) {
  return __devicelib_imf_bfloat162uint_ru(b);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_bfloat162uint_rz(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
unsigned int __imf_bfloat162uint_rz(_iml_bf16_internal b) {
  return __devicelib_imf_bfloat162uint_rz(b);
}

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_bfloat162int_rd(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
int __imf_bfloat162int_rd(_iml_bf16_internal b) {
  return __devicelib_imf_bfloat162int_rd(b);
}

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_bfloat162int_rn(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
int __imf_bfloat162int_rn(_iml_bf16_internal b) {
  return __devicelib_imf_bfloat162int_rn(b);
}

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_bfloat162int_ru(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
int __imf_bfloat162int_ru(_iml_bf16_internal b) {
  return __devicelib_imf_bfloat162int_ru(b);
}

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_bfloat162int_rz(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
int __imf_bfloat162int_rz(_iml_bf16_internal b) {
  return __devicelib_imf_bfloat162int_rz(b);
}

DEVICE_EXTERN_C_INLINE
unsigned short __devicelib_imf_bfloat162ushort_rd(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
unsigned short __imf_bfloat162ushort_rd(_iml_bf16_internal b) {
  return __devicelib_imf_bfloat162ushort_rd(b);
}

DEVICE_EXTERN_C_INLINE
unsigned short __devicelib_imf_bfloat162ushort_rn(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
unsigned short __imf_bfloat162ushort_rn(_iml_bf16_internal b) {
  return __devicelib_imf_bfloat162ushort_rn(b);
}

DEVICE_EXTERN_C_INLINE
unsigned short __devicelib_imf_bfloat162ushort_ru(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
unsigned short __imf_bfloat162ushort_ru(_iml_bf16_internal b) {
  return __devicelib_imf_bfloat162ushort_ru(b);
}

DEVICE_EXTERN_C_INLINE
unsigned short __devicelib_imf_bfloat162ushort_rz(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
unsigned short __imf_bfloat162ushort_rz(_iml_bf16_internal b) {
  return __devicelib_imf_bfloat162ushort_rz(b);
}

DEVICE_EXTERN_C_INLINE
short __devicelib_imf_bfloat162short_rd(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
short __imf_bfloat162short_rd(_iml_bf16_internal b) {
  return __devicelib_imf_bfloat162short_rd(b);
}

DEVICE_EXTERN_C_INLINE
short __devicelib_imf_bfloat162short_rn(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
short __imf_bfloat162short_rn(_iml_bf16_internal b) {
  return __devicelib_imf_bfloat162short_rn(b);
}

DEVICE_EXTERN_C_INLINE
short __devicelib_imf_bfloat162short_ru(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
short __imf_bfloat162short_ru(_iml_bf16_internal b) {
  return __devicelib_imf_bfloat162short_ru(b);
}

DEVICE_EXTERN_C_INLINE
short __devicelib_imf_bfloat162short_rz(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
short __imf_bfloat162short_rz(_iml_bf16_internal b) {
  return __devicelib_imf_bfloat162short_rz(b);
}

DEVICE_EXTERN_C_INLINE
unsigned long long __devicelib_imf_bfloat162ull_rd(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
unsigned long long __imf_bfloat162ull_rd(_iml_bf16_internal b) {
  return __devicelib_imf_bfloat162ull_rd(b);
}

DEVICE_EXTERN_C_INLINE
unsigned long long __devicelib_imf_bfloat162ull_rn(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
unsigned long long __imf_bfloat162ull_rn(_iml_bf16_internal b) {
  return __devicelib_imf_bfloat162ull_rn(b);
}

DEVICE_EXTERN_C_INLINE
unsigned long long __devicelib_imf_bfloat162ull_ru(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
unsigned long long __imf_bfloat162ull_ru(_iml_bf16_internal b) {
  return __devicelib_imf_bfloat162ull_ru(b);
}

DEVICE_EXTERN_C_INLINE
unsigned long long __devicelib_imf_bfloat162ull_rz(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
unsigned long long __imf_bfloat162ull_rz(_iml_bf16_internal b) {
  return __devicelib_imf_bfloat162ull_rz(b);
}

DEVICE_EXTERN_C_INLINE
long long __devicelib_imf_bfloat162ll_rd(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
long long __imf_bfloat162ll_rd(_iml_bf16_internal b) {
  return __devicelib_imf_bfloat162ll_rd(b);
}

DEVICE_EXTERN_C_INLINE
long long __devicelib_imf_bfloat162ll_rn(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
long long __imf_bfloat162ll_rn(_iml_bf16_internal b) {
  return __devicelib_imf_bfloat162ll_rn(b);
}

DEVICE_EXTERN_C_INLINE
long long __devicelib_imf_bfloat162ll_ru(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
long long __imf_bfloat162ll_ru(_iml_bf16_internal b) {
  return __devicelib_imf_bfloat162ll_ru(b);
}

DEVICE_EXTERN_C_INLINE
long long __devicelib_imf_bfloat162ll_rz(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
long long __imf_bfloat162ll_rz(_iml_bf16_internal b) {
  return __devicelib_imf_bfloat162ll_rz(b);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_ushort2bfloat16_rd(unsigned short);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_ushort2bfloat16_rd(unsigned short x) {
  return __devicelib_imf_ushort2bfloat16_rd(x);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_ushort2bfloat16_rn(unsigned short);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_ushort2bfloat16_rn(unsigned short x) {
  return __devicelib_imf_ushort2bfloat16_rn(x);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_ushort2bfloat16_ru(unsigned short);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_ushort2bfloat16_ru(unsigned short x) {
  return __devicelib_imf_ushort2bfloat16_ru(x);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_ushort2bfloat16_rz(unsigned short);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_ushort2bfloat16_rz(unsigned short x) {
  return __devicelib_imf_ushort2bfloat16_rz(x);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_uint2bfloat16_rd(unsigned int);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_uint2bfloat16_rd(unsigned int x) {
  return __devicelib_imf_uint2bfloat16_rd(x);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_uint2bfloat16_rn(unsigned int);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_uint2bfloat16_rn(unsigned int x) {
  return __devicelib_imf_uint2bfloat16_rn(x);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_uint2bfloat16_ru(unsigned int);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_uint2bfloat16_ru(unsigned int x) {
  return __devicelib_imf_uint2bfloat16_ru(x);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_uint2bfloat16_rz(unsigned int);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_uint2bfloat16_rz(unsigned int x) {
  return __devicelib_imf_uint2bfloat16_rz(x);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_ull2bfloat16_rd(unsigned long long);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_ull2bfloat16_rd(unsigned long long x) {
  return __devicelib_imf_ull2bfloat16_rd(x);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_ull2bfloat16_rn(unsigned long long);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_ull2bfloat16_rn(unsigned long long x) {
  return __devicelib_imf_ull2bfloat16_rn(x);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_ull2bfloat16_ru(unsigned long long);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_ull2bfloat16_ru(unsigned long long x) {
  return __devicelib_imf_ull2bfloat16_ru(x);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_ull2bfloat16_rz(unsigned long long);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_ull2bfloat16_rz(unsigned long long x) {
  return __devicelib_imf_ull2bfloat16_rz(x);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_short2bfloat16_rd(short);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_short2bfloat16_rd(short x) {
  return __devicelib_imf_short2bfloat16_rd(x);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_short2bfloat16_rn(short);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_short2bfloat16_rn(short x) {
  return __devicelib_imf_short2bfloat16_rn(x);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_short2bfloat16_ru(short);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_short2bfloat16_ru(short x) {
  return __devicelib_imf_short2bfloat16_ru(x);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_short2bfloat16_rz(short);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_short2bfloat16_rz(short x) {
  return __devicelib_imf_short2bfloat16_rz(x);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_int2bfloat16_rd(int);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_int2bfloat16_rd(int x) {
  return __devicelib_imf_int2bfloat16_rd(x);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_int2bfloat16_rn(int);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_int2bfloat16_rn(int x) {
  return __devicelib_imf_int2bfloat16_rn(x);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_int2bfloat16_ru(int);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_int2bfloat16_ru(int x) {
  return __devicelib_imf_int2bfloat16_ru(x);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_int2bfloat16_rz(int);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_int2bfloat16_rz(int x) {
  return __devicelib_imf_int2bfloat16_rz(x);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_ll2bfloat16_rd(long long);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_ll2bfloat16_rd(long long x) {
  return __devicelib_imf_ll2bfloat16_rd(x);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_ll2bfloat16_rn(long long);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_ll2bfloat16_rn(long long x) {
  return __devicelib_imf_ll2bfloat16_rn(x);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_ll2bfloat16_ru(long long);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_ll2bfloat16_ru(long long x) {
  return __devicelib_imf_ll2bfloat16_ru(x);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_ll2bfloat16_rz(long long);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_ll2bfloat16_rz(long long x) {
  return __devicelib_imf_ll2bfloat16_rz(x);
}

DEVICE_EXTERN_C_INLINE
short __devicelib_imf_bfloat16_as_short(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
short __imf_bfloat16_as_short(_iml_bf16_internal b) {
  return __devicelib_imf_bfloat16_as_short(b);
}

DEVICE_EXTERN_C_INLINE
unsigned short __devicelib_imf_bfloat16_as_ushort(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
unsigned short __imf_bfloat16_as_ushort(_iml_bf16_internal b) {
  return __devicelib_imf_bfloat16_as_ushort(b);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_short_as_bfloat16(short);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_short_as_bfloat16(short x) {
  return __devicelib_imf_short_as_bfloat16(x);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_ushort_as_bfloat16(unsigned short);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_ushort_as_bfloat16(unsigned short x) {
  return __devicelib_imf_ushort_as_bfloat16(x);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_fmabf16(_iml_bf16_internal,
                                           _iml_bf16_internal,
                                           _iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_fmabf16(_iml_bf16_internal a, _iml_bf16_internal b,
                                 _iml_bf16_internal c) {
  return __devicelib_imf_fmabf16(a, b, c);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_sqrtbf16(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_sqrtbf16(_iml_bf16_internal a) {
  return __devicelib_imf_sqrtbf16(a);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_rsqrtbf16(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_rsqrtbf16(_iml_bf16_internal a) {
  return __devicelib_imf_rsqrtbf16(a);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_fminbf16(_iml_bf16_internal,
                                            _iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_fminbf16(_iml_bf16_internal a, _iml_bf16_internal b) {
  return __devicelib_imf_fminbf16(a, b);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_fmaxbf16(_iml_bf16_internal,
                                            _iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_fmaxbf16(_iml_bf16_internal a, _iml_bf16_internal b) {
  return __devicelib_imf_fmaxbf16(a, b);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_fabsbf16(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_fabsbf16(_iml_bf16_internal a) {
  return __devicelib_imf_fabsbf16(a);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_copysignbf16(_iml_bf16_internal,
                                                _iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_copysignbf16(_iml_bf16_internal a,
                                      _iml_bf16_internal b) {
  return __devicelib_imf_copysignbf16(a, b);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_rintbf16(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_rintbf16(_iml_bf16_internal a) {
  return __devicelib_imf_rintbf16(a);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_floorbf16(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_floorbf16(_iml_bf16_internal a) {
  return __devicelib_imf_floorbf16(a);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_ceilbf16(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_ceilbf16(_iml_bf16_internal a) {
  return __devicelib_imf_ceilbf16(a);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_truncbf16(_iml_bf16_internal);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_truncbf16(_iml_bf16_internal a) {
  return __devicelib_imf_truncbf16(a);
}
#endif // __LIBDEVICE_IMF_ENABLED__
