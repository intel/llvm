//==------- fp32_round.cpp - simple fp32 op with rounding mode support------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../device_imf.hpp"
#include "../imf_rounding_op.hpp"

#ifdef __LIBDEVICE_IMF_ENABLED__
DEVICE_EXTERN_C_INLINE
float __devicelib_imf_fadd_rd(float x, float y) {
  return __fp_add_sub_entry(x, y, __IML_RTN);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_fadd_rn(float x, float y) {
  return __fp_add_sub_entry(x, y, __IML_RTE);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_fadd_ru(float x, float y) {
  return __fp_add_sub_entry(x, y, __IML_RTP);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_fadd_rz(float x, float y) {
  return __fp_add_sub_entry(x, y, __IML_RTZ);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_fsub_rd(float x, float y) {
  return __fp_add_sub_entry(x, -y, __IML_RTN);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_fsub_rn(float x, float y) {
  return __fp_add_sub_entry(x, -y, __IML_RTE);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_fsub_ru(float x, float y) {
  return __fp_add_sub_entry(x, -y, __IML_RTP);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_fsub_rz(float x, float y) {
  return __fp_add_sub_entry(x, -y, __IML_RTZ);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_fmul_rd(float x, float y) {
  return __fp_mul(x, y, __IML_RTN);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_fmul_rn(float x, float y) {
  return __fp_mul(x, y, __IML_RTE);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_fmul_ru(float x, float y) {
  return __fp_mul(x, y, __IML_RTP);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_fmul_rz(float x, float y) {
  return __fp_mul(x, y, __IML_RTZ);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_fdiv_rd(float x, float y) {
  return __fp_div(x, y, __IML_RTN);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_fdiv_rn(float x, float y) {
  return __fp_div(x, y, __IML_RTE);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_fdiv_ru(float x, float y) {
  return __fp_div(x, y, __IML_RTP);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_fdiv_rz(float x, float y) {
  return __fp_div(x, y, __IML_RTZ);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_fmaf_rd(float x, float y, float z) {
  return __fp_fma(x, y, z, __IML_RTN);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_fmaf_rn(float x, float y, float z) {
  return __fp_fma(x, y, z, __IML_RTE);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_fmaf_ru(float x, float y, float z) {
  return __fp_fma(x, y, z, __IML_RTP);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_fmaf_rz(float x, float y, float z) {
  return __fp_fma(x, y, z, __IML_RTZ);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_sqrtf_rd(float x) {
  return __fp_sqrt(x);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_sqrtf_rn(float x) {
  return __fp_sqrt(x);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_sqrtf_ru(float x) {
  return __fp_sqrt(x);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_sqrtf_rz(float x) {
  return __fp_sqrt(x);
}
#endif
