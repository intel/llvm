//===----------- imf_fp64_dl.cpp - fp64 functions required by DL ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// Subset of intel math functions is required by deep learning frameworks and
/// we decide to keep all functions required in a separate file and build a new
/// spirv module using this file. By this way, we can reduce unnecessary jit
/// overhead in these deep learning frameworks.
//===----------------------------------------------------------------------===//

#include "../device_imf.hpp"
#ifdef __LIBDEVICE_IMF_ENABLED__

DEVICE_EXTERN_C_INLINE double __devicelib_imf_fabs(double x) {
  return __fabs(x);
}

DEVICE_EXTERN_C_INLINE double __devicelib_imf_fmax(double a, double b) {
  return __fmax(a, b);
}

DEVICE_EXTERN_C_INLINE double __devicelib_imf_fmin(double a, double b) {
  return __fmin(a, b);
}

#endif /*__LIBDEVICE_IMF_ENABLED__*/
