//==- math.h -------------------------------------------------------------0-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#if defined(__SYCL_DEVICE_ONLY__) && defined(_WIN32)
// Define _*dsign before including math.h because _*dsign is used by signbit
// defined in UCRT headers.
extern "C" __attribute__((sycl_device_only, always_inline)) int
_fdsign(float x) {
  return __builtin_signbit(x);
}
extern "C" __attribute__((sycl_device_only, always_inline)) int
_dsign(double x) {
  return __builtin_signbit(x);
}
#endif // __SYCL_DEVICE_ONLY__ && _WIN32

#if defined(__has_include_next)
// GCC/clang support go through this path.
#include_next <math.h>
#else
// MSVC doesn't support "#include_next", so we have to be creative.
// Our header is located in "stl_wrappers/" directory so it won't be picked by
// the following include. MSVC's installation, on the other hand, has the layout
// where the following would result in the header we want. This is obviously
// hacky, but the best we can do...
#include <../include/math.h>
#endif
