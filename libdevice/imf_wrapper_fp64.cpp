//==----- imf_wrapper_fp64.cpp - wrappers for double precision intel math
// library functions ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "device_imf.hpp"

#ifdef __LIBDEVICE_IMF_ENABLED__

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_double2float_rd(double);

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_double2float_rn(double);

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_double2float_ru(double);

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_double2float_rz(double);

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_double2hiint(double);

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_double2int_rd(double);

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_double2int_rn(double);

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_double2int_ru(double);

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_double2int_rz(double);

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_double2uint_rd(double);

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_double2uint_rn(double);

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_double2uint_ru(double);

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_double2uint_rz(double);

DEVICE_EXTERN_C_INLINE
long long int __devicelib_imf_double2ll_rd(double);

DEVICE_EXTERN_C_INLINE
long long int __devicelib_imf_double2ll_rn(double);

DEVICE_EXTERN_C_INLINE
long long int __devicelib_imf_double2ll_ru(double);

DEVICE_EXTERN_C_INLINE
long long int __devicelib_imf_double2ll_rz(double);

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_double2loint(double);

DEVICE_EXTERN_C_INLINE
unsigned long long int __devicelib_imf_double2ull_rd(double);

DEVICE_EXTERN_C_INLINE
unsigned long long int __devicelib_imf_double2ull_rn(double);

DEVICE_EXTERN_C_INLINE
unsigned long long int __devicelib_imf_double2ull_ru(double);

DEVICE_EXTERN_C_INLINE
unsigned long long int __devicelib_imf_double2ull_rz(double);

DEVICE_EXTERN_C_INLINE
long long int __devicelib_imf_double_as_longlong(double);

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_hiloint2double(int, int);

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_int2double_rn(int);

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_ll2double_rd(long long int);

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_ll2double_rn(long long int);

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_ll2double_ru(long long int);

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_ll2double_rz(long long int);

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_longlong_as_double(long long int);

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_uint2double_rn(unsigned int);

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_ull2double_rd(unsigned long long int);

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_ull2double_rn(unsigned long long int);

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_ull2double_ru(unsigned long long int);

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_ull2double_rz(unsigned long long int);

DEVICE_EXTERN_C_INLINE
float __imf_double2float_rd(double x) {
  return __devicelib_imf_double2float_rd(x);
}

DEVICE_EXTERN_C_INLINE
float __imf_double2float_rn(double x) {
  return __devicelib_imf_double2float_rn(x);
}

DEVICE_EXTERN_C_INLINE
float __imf_double2float_ru(double x) {
  return __devicelib_imf_double2float_ru(x);
}

DEVICE_EXTERN_C_INLINE
float __imf_double2float_rz(double x) {
  return __devicelib_imf_double2float_rz(x);
}

DEVICE_EXTERN_C_INLINE
int __imf_double2int_rd(double x) { return __devicelib_imf_double2int_rd(x); }

DEVICE_EXTERN_C_INLINE
int __imf_double2int_rn(double x) { return __devicelib_imf_double2int_rn(x); }

DEVICE_EXTERN_C_INLINE
int __imf_double2int_ru(double x) { return __devicelib_imf_double2int_ru(x); }

DEVICE_EXTERN_C_INLINE
int __imf_double2int_rz(double x) { return __devicelib_imf_double2int_rz(x); }

// TODO: For __imf_double2hiint and __imf_double2loint, we assume underlying
// device is little-endian. We need to check if it is necessary to provide an
// endian independent implementation.
DEVICE_EXTERN_C_INLINE
int __imf_double2hiint(double x) { return __devicelib_imf_double2hiint(x); }

DEVICE_EXTERN_C_INLINE
int __imf_double2loint(double x) { return __devicelib_imf_double2loint(x); }

DEVICE_EXTERN_C_INLINE
unsigned int __imf_double2uint_rd(double x) {
  return __devicelib_imf_double2uint_rd(x);
}

DEVICE_EXTERN_C_INLINE
unsigned int __imf_double2uint_rn(double x) {
  return __devicelib_imf_double2uint_rn(x);
}

DEVICE_EXTERN_C_INLINE
unsigned int __imf_double2uint_ru(double x) {
  return __devicelib_imf_double2uint_ru(x);
}

DEVICE_EXTERN_C_INLINE
unsigned int __imf_double2uint_rz(double x) {
  return __devicelib_imf_double2uint_rz(x);
}

DEVICE_EXTERN_C_INLINE
long long int __imf_double2ll_rd(double x) {
  return __devicelib_imf_double2ll_rd(x);
}

DEVICE_EXTERN_C_INLINE
long long int __imf_double2ll_rn(double x) {
  return __devicelib_imf_double2ll_rn(x);
}

DEVICE_EXTERN_C_INLINE
long long int __imf_double2ll_ru(double x) {
  return __devicelib_imf_double2ll_ru(x);
}

DEVICE_EXTERN_C_INLINE
long long int __imf_double2ll_rz(double x) {
  return __devicelib_imf_double2ll_rz(x);
}

DEVICE_EXTERN_C_INLINE
unsigned long long int __imf_double2ull_rd(double x) {
  return __devicelib_imf_double2ull_rd(x);
}

DEVICE_EXTERN_C_INLINE
unsigned long long int __imf_double2ull_rn(double x) {
  return __devicelib_imf_double2ull_rn(x);
}

DEVICE_EXTERN_C_INLINE
unsigned long long int __imf_double2ull_ru(double x) {
  return __devicelib_imf_double2ull_ru(x);
}

DEVICE_EXTERN_C_INLINE
unsigned long long int __imf_double2ull_rz(double x) {
  return __devicelib_imf_double2ull_rz(x);
}

DEVICE_EXTERN_C_INLINE
long long int __imf_double_as_longlong(double x) {
  return __devicelib_imf_double_as_longlong(x);
}

DEVICE_EXTERN_C_INLINE
double __imf_hiloint2double(int hi, int lo) {
  return __devicelib_imf_hiloint2double(hi, lo);
}

DEVICE_EXTERN_C_INLINE
double __imf_int2double_rn(int x) { return __devicelib_imf_int2double_rn(x); }

DEVICE_EXTERN_C_INLINE
double __imf_ll2double_rd(long long int x) {
  return __devicelib_imf_ll2double_rd(x);
}

DEVICE_EXTERN_C_INLINE
double __imf_ll2double_rn(long long int x) {
  return __devicelib_imf_ll2double_rn(x);
}

DEVICE_EXTERN_C_INLINE
double __imf_ll2double_ru(long long int x) {
  return __devicelib_imf_ll2double_ru(x);
}

DEVICE_EXTERN_C_INLINE
double __imf_ll2double_rz(long long int x) {
  return __devicelib_imf_ll2double_rz(x);
}

DEVICE_EXTERN_C_INLINE
double __imf_longlong_as_double(long long int x) {
  return __devicelib_imf_longlong_as_double(x);
}

DEVICE_EXTERN_C_INLINE
double __imf_uint2double_rn(unsigned int x) {
  return __devicelib_imf_uint2double_rn(x);
}

DEVICE_EXTERN_C_INLINE
double __imf_ull2double_rd(unsigned long long int x) {
  return __devicelib_imf_ull2double_rd(x);
}

DEVICE_EXTERN_C_INLINE
double __imf_ull2double_rn(unsigned long long int x) {
  return __devicelib_imf_ull2double_rn(x);
}

DEVICE_EXTERN_C_INLINE
double __imf_ull2double_ru(unsigned long long int x) {
  return __devicelib_imf_ull2double_ru(x);
}

DEVICE_EXTERN_C_INLINE
double __imf_ull2double_rz(unsigned long long int x) {
  return __devicelib_imf_ull2double_rz(x);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_fma(double, double, double);

DEVICE_EXTERN_C_INLINE
double __imf_fma(double x, double y, double z) {
  return __devicelib_imf_fma(x, y, z);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_floor(double);

DEVICE_EXTERN_C_INLINE
double __imf_floor(double x) { return __devicelib_imf_floor(x); }

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_ceil(double);

DEVICE_EXTERN_C_INLINE
double __imf_ceil(double x) { return __devicelib_imf_ceil(x); }

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_trunc(double);

DEVICE_EXTERN_C_INLINE
double __imf_trunc(double x) { return __devicelib_imf_trunc(x); }

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_rint(double);

DEVICE_EXTERN_C_INLINE
double __imf_rint(double x) { return __devicelib_imf_rint(x); }

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_nearbyint(double);

DEVICE_EXTERN_C_INLINE
double __imf_nearbyint(double x) { return __devicelib_imf_nearbyint(x); }

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_sqrt(double);

DEVICE_EXTERN_C_INLINE
double __imf_sqrt(double x) { return __devicelib_imf_sqrt(x); }

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_rsqrt(double);

DEVICE_EXTERN_C_INLINE
double __imf_rsqrt(double x) { return __devicelib_imf_rsqrt(x); }

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_inv(double);

DEVICE_EXTERN_C_INLINE
double __imf_inv(double x) { return __devicelib_imf_inv(x); }

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_fabs(double);

DEVICE_EXTERN_C_INLINE
double __imf_fabs(double x) { return __devicelib_imf_fabs(x); }

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_fmax(double, double);

DEVICE_EXTERN_C_INLINE
double __imf_fmax(double x, double y) { return __devicelib_imf_fmax(x, y); }

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_fmin(double, double);

DEVICE_EXTERN_C_INLINE
double __imf_fmin(double x, double y) { return __devicelib_imf_fmin(x, y); }

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_copysign(double, double);

DEVICE_EXTERN_C_INLINE
double __imf_copysign(double x, double y) {
  return __devicelib_imf_copysign(x, y);
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_double2half(double);

DEVICE_EXTERN_C_INLINE
_iml_half_internal __imf_double2half(double x) {
  return __devicelib_imf_double2half(x);
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_double2bfloat16(double);

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __imf_double2bfloat16(double x) {
  return __devicelib_imf_double2bfloat16(x);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_dadd_rd(double, double);

DEVICE_EXTERN_C_INLINE
double __imf_dadd_rd(double x, double y) {
  return __devicelib_imf_dadd_rd(x, y);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_dadd_rn(double, double);

DEVICE_EXTERN_C_INLINE
double __imf_dadd_rn(double x, double y) {
  return __devicelib_imf_dadd_rn(x, y);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_dadd_ru(double, double);

DEVICE_EXTERN_C_INLINE
double __imf_dadd_ru(double x, double y) {
  return __devicelib_imf_dadd_ru(x, y);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_dadd_rz(double, double);

DEVICE_EXTERN_C_INLINE
double __imf_dadd_rz(double x, double y) {
  return __devicelib_imf_dadd_rz(x, y);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_dsub_rd(double, double);

DEVICE_EXTERN_C_INLINE
double __imf_dsub_rd(double x, double y) {
  return __devicelib_imf_dsub_rd(x, y);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_dsub_rn(double, double);

DEVICE_EXTERN_C_INLINE
double __imf_dsub_rn(double x, double y) {
  return __devicelib_imf_dsub_rn(x, y);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_dsub_ru(double, double);

DEVICE_EXTERN_C_INLINE
double __imf_dsub_ru(double x, double y) {
  return __devicelib_imf_dsub_ru(x, y);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_dsub_rz(double, double);

DEVICE_EXTERN_C_INLINE
double __imf_dsub_rz(double x, double y) {
  return __devicelib_imf_dsub_rz(x, y);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_dmul_rd(double, double);

DEVICE_EXTERN_C_INLINE
double __imf_dmul_rd(double x, double y) {
  return __devicelib_imf_dmul_rd(x, y);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_dmul_rn(double, double);

DEVICE_EXTERN_C_INLINE
double __imf_dmul_rn(double x, double y) {
  return __devicelib_imf_dmul_rn(x, y);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_dmul_ru(double, double);

DEVICE_EXTERN_C_INLINE
double __imf_dmul_ru(double x, double y) {
  return __devicelib_imf_dmul_ru(x, y);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_dmul_rz(double, double);

DEVICE_EXTERN_C_INLINE
double __imf_dmul_rz(double x, double y) {
  return __devicelib_imf_dmul_rz(x, y);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_ddiv_rd(double, double);

DEVICE_EXTERN_C_INLINE
double __imf_ddiv_rd(double x, double y) {
  return __devicelib_imf_ddiv_rd(x, y);
}

DEVICE_EXTERN_C_INLINE
double __imf_drcp_rd(double x) { return __devicelib_imf_ddiv_rd(1.0, x); }

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_ddiv_rn(double, double);

DEVICE_EXTERN_C_INLINE
double __imf_ddiv_rn(double x, double y) {
  return __devicelib_imf_ddiv_rn(x, y);
}

DEVICE_EXTERN_C_INLINE
double __imf_drcp_rn(double x) { return __devicelib_imf_ddiv_rn(1.0, x); }

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_ddiv_ru(double, double);

DEVICE_EXTERN_C_INLINE
double __imf_ddiv_ru(double x, double y) {
  return __devicelib_imf_ddiv_ru(x, y);
}

DEVICE_EXTERN_C_INLINE
double __imf_drcp_ru(double x) { return __devicelib_imf_ddiv_ru(1.0, x); }

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_ddiv_rz(double, double);

DEVICE_EXTERN_C_INLINE
double __imf_ddiv_rz(double x, double y) {
  return __devicelib_imf_ddiv_rz(x, y);
}

DEVICE_EXTERN_C_INLINE
double __imf_drcp_rz(double x) { return __devicelib_imf_ddiv_rz(1.0, x); }

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_fma_rd(double, double, double);

DEVICE_EXTERN_C_INLINE
double __imf_fma_rd(double x, double y, double z) {
  return __devicelib_imf_fma_rd(x, y, z);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_fma_rn(double, double, double);

DEVICE_EXTERN_C_INLINE
double __imf_fma_rn(double x, double y, double z) {
  return __devicelib_imf_fma_rn(x, y, z);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_fma_ru(double, double, double);

DEVICE_EXTERN_C_INLINE
double __imf_fma_ru(double x, double y, double z) {
  return __devicelib_imf_fma_ru(x, y, z);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_fma_rz(double, double, double);

DEVICE_EXTERN_C_INLINE
double __imf_fma_rz(double x, double y, double z) {
  return __devicelib_imf_fma_rz(x, y, z);
}
#endif // __LIBDEVICE_IMF_ENABLED__
