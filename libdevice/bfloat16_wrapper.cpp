//==--- bfloat16_wrapper.cpp - wrappers for bfloat16 library functions ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "device.h"

#if defined(__SPIR__) || defined(__SPIRV__)

#include <CL/__spirv/spirv_ops.hpp>
#include <cstdint>

DEVICE_EXTERN_C_INLINE
uint16_t __devicelib_ConvertFToBF16INTEL(const float &x) {
  return __spirv_ConvertFToBF16INTEL(x);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_ConvertBF16ToFINTEL(const uint16_t &x) {
  return __spirv_ConvertBF16ToFINTEL(x);
}

#endif // __SPIR__ || __SPIRV__
