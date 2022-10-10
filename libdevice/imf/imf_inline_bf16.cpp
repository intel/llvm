//==----- imf_inline_bf16.cpp - some bf16 trivial intel math functions -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "../device_imf.hpp"

#ifdef __LIBDEVICE_IMF_ENABLED__
DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_fmabf16(
    _iml_bf16_internal a, _iml_bf16_internal b, _iml_bf16_internal c) {
  return __fma(_iml_bf16(a), _iml_bf16(b), _iml_bf16(c)).get_internal();
}
#endif
