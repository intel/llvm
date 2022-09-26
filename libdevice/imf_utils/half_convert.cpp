//==-- half_convert.cpp - fallback implementation of half to other type
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
float __devicelib_imf_half2float(_iml_half_internal x) {
  return __half2float(x);
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_float2half_rd(float x) {
#if defined(__SPIR__)
  return __spirv_FConvert_Rhalf_rtn(x);
#else
  return __iml_half2fp(x, __IML_RTN);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_float2half_rn(float x) {
#if defined(__SPIR__)
  return __spirv_FConvert_Rhalf_rte(x);
#else
  return __iml_half2fp(x, __IML_RTE);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_float2half_ru(float x) {
#if defined(__SPIR__)
  return __spirv_FConvert_Rhalf_rtp(x);
#else
  return __iml_half2fp(x, __IML_RTP);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_float2half_rz(float x) {
#if defined(__SPIR__)
  return __spirv_FConvert_Rhalf_rtz(x);
#else
  return __iml_half2fp(x, __IML_RTZ);
#endif
}
#endif // __LIBDEVICE_IMF_ENABLED__
