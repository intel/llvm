//==------- fp64_round.cpp - simple fp64 op with rounding mode support------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../device_imf.hpp"

#ifdef __LIBDEVICE_IMF_ENABLED__
DEVICE_EXTERN_C_INLINE
double __devicelib_imf_dadd_rd(double x, double y) {
  return __fp_add_sub_entry(x, y, __IML_RTN);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_dadd_rn(double x, double y) {
  return __fp_add_sub_entry(x, y, __IML_RTE);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_dadd_ru(double x, double y) {
  return __fp_add_sub_entry(x, y, __IML_RTP);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_dadd_rz(double x, double y) {
  return __fp_add_sub_entry(x, y, __IML_RTZ);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_dsub_rd(double x, double y) {
  return __fp_add_sub_entry(x, -y, __IML_RTN);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_dsub_rn(double x, double y) {
  return __fp_add_sub_entry(x, -y, __IML_RTE);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_dsub_ru(double x, double y) {
  return __fp_add_sub_entry(x, -y, __IML_RTP);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_dsub_rz(double x, double y) {
  return __fp_add_sub_entry(x, -y, __IML_RTZ);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_dmul_rd(double x, double y) {
  return __fp_mul(x, y, __IML_RTN);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_dmul_rn(double x, double y) {
  return __fp_mul(x, y, __IML_RTE);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_dmul_ru(double x, double y) {
  return __fp_mul(x, y, __IML_RTP);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_dmul_rz(double x, double y) {
  return __fp_mul(x, y, __IML_RTZ);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_ddiv_rd(double x, double y) {
  return __fp_div(x, y, __IML_RTN);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_ddiv_rn(double x, double y) {
  return __fp_div(x, y, __IML_RTE);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_ddiv_ru(double x, double y) {
  return __fp_div(x, y, __IML_RTP);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_ddiv_rz(double x, double y) {
  return __fp_div(x, y, __IML_RTZ);
}
#endif
