//==------- fallback-bfloat16.cpp - bfloat16 conversions in software -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "device.h"

#if defined(__SPIR__) || defined(__SPIRV__)

#include <cstdint>

// To support fallback device libraries on-demand loading, please update the
// DeviceLibFuncMap in llvm/tools/sycl-post-link/sycl-post-link.cpp if you add
// or remove any item in this file.
// TODO: generate the DeviceLibFuncMap in sycl-post-link.cpp automatically
// during the build based on libdevice to avoid manually sync.

DEVICE_EXTERN_C_INLINE uint16_t
__devicelib_ConvertFToBF16INTEL(const float &a) {
  // In case float value is nan - propagate bfloat16's qnan
  if (__spirv_IsNan(a))
    return 0xffc1;
  union {
    uint32_t intStorage;
    float floatValue;
  };
  floatValue = a;
  // Do RNE and truncate
  uint32_t roundingBias = ((intStorage >> 16) & 0x1) + 0x00007FFF;
  return static_cast<uint16_t>((intStorage + roundingBias) >> 16);
}

DEVICE_EXTERN_C_INLINE float
__devicelib_ConvertBF16ToFINTEL(const uint16_t &a) {
  union {
    uint32_t intStorage;
    float floatValue;
  };
  intStorage = a << 16;
  return floatValue;
}

DEVICE_EXTERN_C_INLINE
void __devicelib_ConvertFToBF16INTELVec(const float *src, uint16_t *dst, int size) { 
  if (size == 1 || size == 2 || size == 3 || size == 4 || size == 8 || size == 16) {
    for (int i = 0; i < size; ++i) {
      dst[i] = __devicelib_ConvertFToBF16INTEL(src[i]);
    }
  }
  // Invalid size. Should we throw an exception/assert when size is invalid?
  else {
  }
}

DEVICE_EXTERN_C_INLINE
void __devicelib_ConvertBF16ToFINTELVec(const uint16_t *src, float *dst, int size) {
  if (size == 1 || size == 2 || size == 3 || size == 4 || size == 8 || size == 16) {
    for (int i = 0; i < size; ++i) {
      dst[i] = __devicelib_ConvertBF16ToFINTEL(src[i]);
    }
  }
  // Invalid size. Should we throw an exception/assert when size is invalid?
  else {
  }
}

#endif // __SPIR__ || __SPIRV__
