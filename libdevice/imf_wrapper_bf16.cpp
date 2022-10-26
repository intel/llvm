//==----- imf_wrapper_bf16.cpp - wrappers for BFloat16 intel math library
// functions ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "imf_bf16.hpp"

#ifdef __LIBDEVICE_IMF_ENABLED__

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
