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
#include <cstdint>
#include <cassert>

DEVICE_EXTERN_C_INLINE
uint16_t __devicelib_ConvertFToBF16INTEL(const float &x) {
  return __spirv_ConvertFToBF16INTEL(x);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_ConvertBF16ToFINTEL(const uint16_t &x) {
  return __spirv_ConvertBF16ToFINTEL(x);
}

DEVICE_EXTERN_C_INLINE
void __devicelib_ConvertFToBF16INTELVec(const float *src, uint16_t *dst, int size) { 
  if (size == 1) {
    dst[0] = __spirv_ConvertFToBF16INTEL(src[0]);
  } else if (size == 2) {
    __ocl_vec_t<float, 2> x = *reinterpret_cast<const __ocl_vec_t<float, 2> *>(src);
    __ocl_vec_t<uint16_t, 2> y = __spirv_ConvertFToBF16INTEL(x);
    *reinterpret_cast<__ocl_vec_t<uint16_t, 2> *>(dst) = y;
  } else if (size == 3) {
    __ocl_vec_t<float, 3> x = *reinterpret_cast<const __ocl_vec_t<float, 3> *>(src);
    __ocl_vec_t<uint16_t, 3> y = __spirv_ConvertFToBF16INTEL(x);
    *reinterpret_cast<__ocl_vec_t<uint16_t, 3> *>(dst) = y;
  } else if (size == 4) {
    __ocl_vec_t<float, 4> x = *reinterpret_cast<const __ocl_vec_t<float, 4> *>(src);
    __ocl_vec_t<uint16_t, 4> y = __spirv_ConvertFToBF16INTEL(x);
    *reinterpret_cast<__ocl_vec_t<uint16_t, 4> *>(dst) = y;
  } else if (size == 8) {
    __ocl_vec_t<float, 8> x = *reinterpret_cast<const __ocl_vec_t<float, 8> *>(src);
    __ocl_vec_t<uint16_t, 8> y = __spirv_ConvertFToBF16INTEL(x);
    *reinterpret_cast<__ocl_vec_t<uint16_t, 8> *>(dst) = y;
  } else if (size == 16) {
    __ocl_vec_t<float, 16> x = *reinterpret_cast<const __ocl_vec_t<float, 16> *>(src);
    __ocl_vec_t<uint16_t, 16> y = __spirv_ConvertFToBF16INTEL(x);
    *reinterpret_cast<__ocl_vec_t<uint16_t, 16> *>(dst) = y;
  }
  // Invalid size. Should we throw an exception/assert when size is invalid?
  else {
  }
}

DEVICE_EXTERN_C_INLINE
void __devicelib_ConvertBF16ToFINTELVec(const uint16_t *src, float *dst, int size) {
  if (size == 1) {
    dst[0] = __spirv_ConvertBF16ToFINTEL(src[0]);
  } else if (size == 2) {
    __ocl_vec_t<uint16_t, 2> x = *reinterpret_cast<const __ocl_vec_t<uint16_t, 2> *>(src);
    __ocl_vec_t<float, 2> y = __spirv_ConvertBF16ToFINTEL(x);
    *reinterpret_cast<__ocl_vec_t<float, 2> *>(dst) = y;
  } else if (size == 3) {
    __ocl_vec_t<uint16_t, 3> x = *reinterpret_cast<const __ocl_vec_t<uint16_t, 3> *>(src);
    __ocl_vec_t<float, 3> y = __spirv_ConvertBF16ToFINTEL(x);
    *reinterpret_cast<__ocl_vec_t<float, 3> *>(dst) = y;
  } else if (size == 4) {
    __ocl_vec_t<uint16_t, 4> x = *reinterpret_cast<const __ocl_vec_t<uint16_t, 4> *>(src);
    __ocl_vec_t<float, 4> y = __spirv_ConvertBF16ToFINTEL(x);
    *reinterpret_cast<__ocl_vec_t<float, 4> *>(dst) = y;
  } else if (size == 8) {
    __ocl_vec_t<uint16_t, 8> x = *reinterpret_cast<const __ocl_vec_t<uint16_t, 8> *>(src);
    __ocl_vec_t<float, 8> y = __spirv_ConvertBF16ToFINTEL(x);
    *reinterpret_cast<__ocl_vec_t<float, 8> *>(dst) = y;
  } else if (size == 16) {
    __ocl_vec_t<uint16_t, 16> x = *reinterpret_cast<const __ocl_vec_t<uint16_t, 16> *>(src);
    __ocl_vec_t<float, 16> y = __spirv_ConvertBF16ToFINTEL(x);
    *reinterpret_cast<__ocl_vec_t<float, 16> *>(dst) = y;
  }
  // Invalid size. Should we throw an exception/assert when size is invalid?
  else {
  }
}

#endif // __SPIR__ || __SPIRV__
