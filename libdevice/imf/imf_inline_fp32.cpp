//==----- imf_inline_fp32.cpp - some fp32 trivial intel math functions -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../device.h"

#ifdef __LIBDEVICE_IMF_ENABLED__

#include "../device_imf.hpp"

DEVICE_EXTERN_C_INLINE _iml_half_internal __devicelib_imf_fmaf16(
    _iml_half_internal a, _iml_half_internal b, _iml_half_internal c) {
  _iml_half ha(a), hb(b), hc(c);
  return __fma(ha, hb, hc).get_internal();
}

DEVICE_EXTERN_C_INLINE _iml_half_internal
__devicelib_imf_floorf16(_iml_half_internal x) {
  _iml_half hx(x);
  return __floor(hx).get_internal();
}

DEVICE_EXTERN_C_INLINE _iml_half_internal
__devicelib_imf_ceilf16(_iml_half_internal x) {
  _iml_half hx(x);
  return __ceil(hx).get_internal();
}

DEVICE_EXTERN_C_INLINE _iml_half_internal
__devicelib_imf_truncf16(_iml_half_internal x) {
  _iml_half hx(x);
  return __trunc(hx).get_internal();
}

DEVICE_EXTERN_C_INLINE _iml_half_internal
__devicelib_imf_rintf16(_iml_half_internal x) {
  _iml_half hx(x);
  return __rint(hx).get_internal();
}

DEVICE_EXTERN_C_INLINE _iml_half_internal
__devicelib_imf_nearbyintf16(_iml_half_internal x) {
  _iml_half hx(x);
  return __rint(hx).get_internal();
}

DEVICE_EXTERN_C_INLINE _iml_half_internal
__devicelib_imf_sqrtf16(_iml_half_internal a) {
  _iml_half ha(a);
  return __sqrt(ha).get_internal();
}

DEVICE_EXTERN_C_INLINE _iml_half_internal
__devicelib_imf_rsqrtf16(_iml_half_internal a) {
  _iml_half ha(a);
  return __rsqrt(ha).get_internal();
}

DEVICE_EXTERN_C_INLINE _iml_half_internal
__devicelib_imf_invf16(_iml_half_internal a) {
  _iml_half ha(a), h1(1.0f);
  return (h1 / ha).get_internal();
}

DEVICE_EXTERN_C_INLINE _iml_half_internal
__devicelib_imf_fmaxf16(_iml_half_internal a, _iml_half_internal b) {
  _iml_half ha(a), hb(b);
  return __fmax(ha, hb).get_internal();
}

DEVICE_EXTERN_C_INLINE _iml_half_internal
__devicelib_imf_fminf16(_iml_half_internal a, _iml_half_internal b) {
  _iml_half ha(a), hb(b);
  return __fmin(ha, hb).get_internal();
}

DEVICE_EXTERN_C_INLINE _iml_half_internal
__devicelib_imf_fabsf16(_iml_half_internal x) {
  _iml_half hx(x);
  return __fabs(hx).get_internal();
}

DEVICE_EXTERN_C_INLINE _iml_half_internal
__devicelib_imf_copysignf16(_iml_half_internal a, _iml_half_internal b) {
  _iml_half ha(a), hb(b);
  return __copysign(ha, hb).get_internal();
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_saturatef(float x) { return __fclamp(x, .0f, 1.f); }

DEVICE_EXTERN_C_INLINE float __devicelib_imf_fmaf(float a, float b, float c) {
  return __fma(a, b, c);
}

DEVICE_EXTERN_C_INLINE float __devicelib_imf_floorf(float x) {
  return __floor(x);
}

DEVICE_EXTERN_C_INLINE float __devicelib_imf_ceilf(float x) {
  return __ceil(x);
}

DEVICE_EXTERN_C_INLINE float __devicelib_imf_truncf(float x) {
  return __trunc(x);
}

DEVICE_EXTERN_C_INLINE float __devicelib_imf_rintf(float x) {
  return __rint(x);
}

DEVICE_EXTERN_C_INLINE float __devicelib_imf_nearbyintf(float x) {
  return __rint(x);
}

DEVICE_EXTERN_C_INLINE float __devicelib_imf_sqrtf(float a) {
  return __sqrt(a);
}

DEVICE_EXTERN_C_INLINE float __devicelib_imf_rsqrtf(float a) {
  return __rsqrt(a);
}

DEVICE_EXTERN_C_INLINE float __devicelib_imf_invf(float a) { return 1.0f / a; }

DEVICE_EXTERN_C_INLINE float __devicelib_imf_copysignf(float a, float b) {
  return __copysign(a, b);
}
#endif /*__LIBDEVICE_IMF_ENABLED__*/
