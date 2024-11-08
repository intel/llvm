//==----- imf_inline_bf16.cpp - some bf16 trivial intel math functions -----==//
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
_iml_bf16_internal __devicelib_imf_fmabf16(_iml_bf16_internal a,
                                           _iml_bf16_internal b,
                                           _iml_bf16_internal c) {
  return __fma(_iml_bf16(a), _iml_bf16(b), _iml_bf16(c)).get_internal();
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_sqrtbf16(_iml_bf16_internal a) {
  return __sqrt(_iml_bf16(a)).get_internal();
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_rsqrtbf16(_iml_bf16_internal a) {
  return __rsqrt(_iml_bf16(a)).get_internal();
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_fminbf16(_iml_bf16_internal a,
                                            _iml_bf16_internal b) {
  return __fmin(_iml_bf16(a), _iml_bf16(b)).get_internal();
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_fmaxbf16(_iml_bf16_internal a,
                                            _iml_bf16_internal b) {
  return __fmax(_iml_bf16(a), _iml_bf16(b)).get_internal();
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_copysignbf16(_iml_bf16_internal a,
                                                _iml_bf16_internal b) {
  return __copysign(_iml_bf16(a), _iml_bf16(b)).get_internal();
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_fabsbf16(_iml_bf16_internal a) {
  return __fabs(_iml_bf16(a)).get_internal();
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_rintbf16(_iml_bf16_internal a) {
  return __rint(_iml_bf16(a)).get_internal();
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_floorbf16(_iml_bf16_internal a) {
  return __floor(_iml_bf16(a)).get_internal();
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_ceilbf16(_iml_bf16_internal a) {
  return __ceil(_iml_bf16(a)).get_internal();
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_truncbf16(_iml_bf16_internal a) {
  return __trunc(_iml_bf16(a)).get_internal();
}
#endif
