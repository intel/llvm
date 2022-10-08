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
#endif
