//==----- imf_wrapper.cpp - wrappers for intel math library functions ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "device.h"

#ifdef __LIBDEVICE_IMF_ENABLED__
DEVICE_EXTERN_C
float __devicelib_imf_saturatef(float x);

DEVICE_EXTERN_C
float __imf_saturatef(float x) { return __devicelib_imf_saturatef(x); }

#endif // __LIBDEVICE_IMF_ENABLED__
