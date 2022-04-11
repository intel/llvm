//==-- fallback-imf.cpp - fallback implementation of intel math functions --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "device_math.h"

#ifdef __LIBDEVICE_IMF_ENABLED__

static inline float __fclamp(float x, float y, float z) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_fmin(__builtin_fmax(x, y), z);
#elif defined(__SPIR__)
  return __spirv_ocl_fclamp(x, y, z);
#endif
}

DEVICE_EXTERN_C
float __devicelib_imf_saturatef(float x) { return __fclamp(x, .0f, 1.f); }

#endif // __LIBDEVICE_IMF_ENABLED__
