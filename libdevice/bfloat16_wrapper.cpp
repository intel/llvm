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
#include <CL/__spirv/spirv_types.hpp>
#include <cassert>
#include <cstdint>

DEVICE_EXTERN_C_INLINE
uint16_t __devicelib_ConvertFToBF16INTEL(const float &x) {
  return __spirv_ConvertFToBF16INTEL(x);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_ConvertBF16ToFINTEL(const uint16_t &x) {
  return __spirv_ConvertBF16ToFINTEL(x);
}

// For vector of size 1.
DEVICE_EXTERN_C_INLINE
void __devicelib_ConvertFToBF16INTELVec1(const float *src, uint16_t *dst) {
  dst[0] = __spirv_ConvertFToBF16INTEL(src[0]);
}
DEVICE_EXTERN_C_INLINE
void __devicelib_ConvertBF16ToFINTELVec1(const uint16_t *src, float *dst) {
  dst[0] = __spirv_ConvertBF16ToFINTEL(src[0]);
}

// Generate the conversion functions for vector of size 2, 3, 4, 8, 16.
#define GenerateConvertFunctionForVec(size)                                    \
  DEVICE_EXTERN_C_INLINE                                                       \
  void __devicelib_ConvertFToBF16INTELVec##size(const float *src,              \
                                                uint16_t *dst) {               \
    __ocl_vec_t<float, size> x =                                               \
        *__builtin_bit_cast(const __ocl_vec_t<float, size> *, src);            \
    __ocl_vec_t<uint16_t, size> y = __spirv_ConvertFToBF16INTEL(x);            \
    *__builtin_bit_cast(__ocl_vec_t<uint16_t, size> *, dst) = y;               \
  }                                                                            \
  DEVICE_EXTERN_C_INLINE                                                       \
  void __devicelib_ConvertBF16ToFINTELVec##size(const uint16_t *src,           \
                                                float *dst) {                  \
    __ocl_vec_t<uint16_t, size> x =                                            \
        *__builtin_bit_cast(const __ocl_vec_t<uint16_t, size> *, src);         \
    __ocl_vec_t<float, size> y = __spirv_ConvertBF16ToFINTEL(x);               \
    *__builtin_bit_cast(__ocl_vec_t<float, size> *, dst) = y;                  \
  }

// clang-format off
GenerateConvertFunctionForVec(2)
GenerateConvertFunctionForVec(3)
GenerateConvertFunctionForVec(4)
GenerateConvertFunctionForVec(8)
GenerateConvertFunctionForVec(16)
// clang-format on
#undef GenerateConvertFunctionForVec

#endif // __SPIR__ || __SPIRV__
