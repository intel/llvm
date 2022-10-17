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
#endif // __LIBDEVICE_IMF_ENABLED__
