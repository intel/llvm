//==----- imf_wrapper.cpp - wrappers for intel math library functions ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "device_imf.hpp"

#ifdef __LIBDEVICE_IMF_ENABLED__

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_saturatef(float);

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_half2float(_iml_half_internal);

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_float2int_rd(float);

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_float2int_rn(float);

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_float2int_ru(float);

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_float2int_rz(float);

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_float2uint_rd(float);

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_float2uint_rn(float);

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_float2uint_ru(float);

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_float2uint_rz(float);

DEVICE_EXTERN_C_INLINE
long long int __devicelib_imf_float2ll_rd(float);

DEVICE_EXTERN_C_INLINE
long long int __devicelib_imf_float2ll_rn(float);

DEVICE_EXTERN_C_INLINE
long long int __devicelib_imf_float2ll_ru(float);

DEVICE_EXTERN_C_INLINE
long long int __devicelib_imf_float2ll_rz(float);

DEVICE_EXTERN_C_INLINE
unsigned long long int __devicelib_imf_float2ull_rd(float);

DEVICE_EXTERN_C_INLINE
unsigned long long int __devicelib_imf_float2ull_rn(float);

DEVICE_EXTERN_C_INLINE
unsigned long long int __devicelib_imf_float2ull_ru(float);

DEVICE_EXTERN_C_INLINE
unsigned long long int __devicelib_imf_float2ull_rz(float);

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_float_as_int(float);

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_float_as_uint(float);

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_int2float_rd(int);

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_int2float_rn(int);

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_int2float_ru(int);

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_int2float_rz(int);

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_int_as_float(int);

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_ll2float_rd(long long int);

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_ll2float_rn(long long int);

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_ll2float_ru(long long int);

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_ll2float_rz(long long int);

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_uint2float_rd(unsigned int);

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_uint2float_rn(unsigned int);

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_uint2float_ru(unsigned int);

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_uint2float_rz(unsigned int);

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_uint_as_float(unsigned int);

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_ull2float_rd(unsigned long long int);

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_ull2float_rn(unsigned long long int);

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_ull2float_ru(unsigned long long int);

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_ull2float_rz(unsigned long long int);

DEVICE_EXTERN_C_INLINE
float __imf_saturatef(float x) { return __devicelib_imf_saturatef(x); }

DEVICE_EXTERN_C_INLINE
float __imf_expf(float x) { return __devicelib_imf_expf(x); }

DEVICE_EXTERN_C_INLINE
_iml_half_internal __imf_expf16(_iml_half_internal x) {
  return __devicelib_imf_expf16(x);
}

DEVICE_EXTERN_C_INLINE
int __imf_float2int_rd(float x) { return __devicelib_imf_float2int_rd(x); }

DEVICE_EXTERN_C_INLINE
int __imf_float2int_rn(float x) { return __devicelib_imf_float2int_rn(x); }

DEVICE_EXTERN_C_INLINE
int __imf_float2int_ru(float x) { return __devicelib_imf_float2int_ru(x); }

DEVICE_EXTERN_C_INLINE
int __imf_float2int_rz(float x) { return __devicelib_imf_float2int_rz(x); }

DEVICE_EXTERN_C_INLINE
unsigned int __imf_float2uint_rd(float x) {
  return __devicelib_imf_float2uint_rd(x);
}

DEVICE_EXTERN_C_INLINE
unsigned int __imf_float2uint_rn(float x) {
  return __devicelib_imf_float2uint_rn(x);
}

DEVICE_EXTERN_C_INLINE
unsigned int __imf_float2uint_ru(float x) {
  return __devicelib_imf_float2uint_ru(x);
}

DEVICE_EXTERN_C_INLINE
unsigned int __imf_float2uint_rz(float x) {
  return __devicelib_imf_float2uint_rz(x);
}

DEVICE_EXTERN_C_INLINE
long long int __imf_float2ll_rd(float x) {
  return __devicelib_imf_float2ll_rd(x);
}

DEVICE_EXTERN_C_INLINE
long long int __imf_float2ll_rn(float x) {
  return __devicelib_imf_float2ll_rn(x);
}

DEVICE_EXTERN_C_INLINE
long long int __imf_float2ll_ru(float x) {
  return __devicelib_imf_float2ll_ru(x);
}

DEVICE_EXTERN_C_INLINE
long long int __imf_float2ll_rz(float x) {
  return __devicelib_imf_float2ll_rz(x);
}

DEVICE_EXTERN_C_INLINE
unsigned long long int __imf_float2ull_rd(float x) {
  return __devicelib_imf_float2ull_rd(x);
}

DEVICE_EXTERN_C_INLINE
unsigned long long int __imf_float2ull_rn(float x) {
  return __devicelib_imf_float2ull_rn(x);
}

DEVICE_EXTERN_C_INLINE
unsigned long long int __imf_float2ull_ru(float x) {
  return __devicelib_imf_float2ull_ru(x);
}

DEVICE_EXTERN_C_INLINE
unsigned long long int __imf_float2ull_rz(float x) {
  return __devicelib_imf_float2ull_rz(x);
}

DEVICE_EXTERN_C_INLINE
int __imf_float_as_int(float x) { return __devicelib_imf_float_as_int(x); }

DEVICE_EXTERN_C_INLINE
unsigned int __imf_float_as_uint(float x) {
  return __devicelib_imf_float_as_uint(x);
}

DEVICE_EXTERN_C_INLINE
float __imf_int2float_rd(int x) { return __devicelib_imf_int2float_rd(x); }

DEVICE_EXTERN_C_INLINE
float __imf_int2float_rn(int x) { return __devicelib_imf_int2float_rn(x); }

DEVICE_EXTERN_C_INLINE
float __imf_int2float_ru(int x) { return __devicelib_imf_int2float_ru(x); }

DEVICE_EXTERN_C_INLINE
float __imf_int2float_rz(int x) { return __devicelib_imf_int2float_rz(x); }

DEVICE_EXTERN_C_INLINE
float __imf_int_as_float(int x) { return __devicelib_imf_int_as_float(x); }

DEVICE_EXTERN_C_INLINE
float __imf_ll2float_rd(long long int x) {
  return __devicelib_imf_ll2float_rd(x);
}

DEVICE_EXTERN_C_INLINE
float __imf_ll2float_rn(long long int x) {
  return __devicelib_imf_ll2float_rn(x);
}

DEVICE_EXTERN_C_INLINE
float __imf_ll2float_ru(long long int x) {
  return __devicelib_imf_ll2float_ru(x);
}

DEVICE_EXTERN_C_INLINE
float __imf_ll2float_rz(long long int x) {
  return __devicelib_imf_ll2float_rz(x);
}

DEVICE_EXTERN_C_INLINE
float __imf_uint2float_rd(unsigned int x) {
  return __devicelib_imf_uint2float_rd(x);
}

DEVICE_EXTERN_C_INLINE
float __imf_uint2float_rn(unsigned int x) {
  return __devicelib_imf_uint2float_rn(x);
}

DEVICE_EXTERN_C_INLINE
float __imf_uint2float_ru(unsigned int x) {
  return __devicelib_imf_uint2float_ru(x);
}

DEVICE_EXTERN_C_INLINE
float __imf_uint2float_rz(unsigned int x) {
  return __devicelib_imf_uint2float_rz(x);
}

DEVICE_EXTERN_C_INLINE
float __imf_uint_as_float(unsigned int x) {
  return __devicelib_imf_uint_as_float(x);
}

DEVICE_EXTERN_C_INLINE
float __imf_ull2float_rd(unsigned long long int x) {
  return __devicelib_imf_ull2float_rd(x);
}

DEVICE_EXTERN_C_INLINE
float __imf_ull2float_rn(unsigned long long int x) {
  return __devicelib_imf_ull2float_rn(x);
}

DEVICE_EXTERN_C_INLINE
float __imf_ull2float_ru(unsigned long long int x) {
  return __devicelib_imf_ull2float_ru(x);
}

DEVICE_EXTERN_C_INLINE
float __imf_ull2float_rz(unsigned long long int x) {
  return __devicelib_imf_ull2float_rz(x);
}

DEVICE_EXTERN_C_INLINE
float __imf_half2float(_iml_half_internal x) {
  return __devicelib_imf_half2float(x);
}

#endif // __LIBDEVICE_IMF_ENABLED__
