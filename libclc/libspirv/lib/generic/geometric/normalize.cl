//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#define _CLC_SPIRV_NORMALIZE_IMPL(FP_TYPE, FLOAT_MARK, INT_TYPE, VLEN,         \
                                  MAX_SQRT, MIN_SQRT)                          \
  _CLC_OVERLOAD _CLC_DEF FP_TYPE##VLEN __spirv_ocl_normalize(                  \
      FP_TYPE##VLEN p) {                                                       \
    if (__spirv_All(__spirv_SConvert_Rchar##VLEN(                              \
            p == (FP_TYPE##VLEN)0.0##FLOAT_MARK)))                             \
      return p;                                                                \
    FP_TYPE l2 = __spirv_Dot(p, p);                                            \
    if (l2 < FLT_MIN) {                                                        \
      p *= MAX_SQRT;                                                           \
      l2 = __spirv_Dot(p, p);                                                  \
    } else if (l2 == INFINITY) {                                               \
      p *= MIN_SQRT;                                                           \
      l2 = __spirv_Dot(p, p);                                                  \
      if (l2 == INFINITY) {                                                    \
        p = __spirv_ocl_copysign(                                              \
            __spirv_ocl_select(                                                \
                (FP_TYPE##VLEN)0.0##FLOAT_MARK,                                \
                (FP_TYPE##VLEN)1.0##FLOAT_MARK,                                \
                __spirv_SConvert_R##INT_TYPE##VLEN(__spirv_IsInf(p))),         \
            p);                                                                \
        l2 = __spirv_Dot(p, p);                                                \
      }                                                                        \
    }                                                                          \
    return p * __spirv_ocl_rsqrt(l2);                                          \
  }

#define _CLC_SPIRV_NORMALIZE(VLEN)                                             \
  _CLC_SPIRV_NORMALIZE_IMPL(float, f, int, VLEN, 0x1.0p+86F, 0x1.0p-65f)

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_normalize(float p) {
  return __spirv_ocl_sign(p);
}

_CLC_SPIRV_NORMALIZE(2)
_CLC_SPIRV_NORMALIZE(3)
_CLC_SPIRV_NORMALIZE(4)

#undef _CLC_SPIRV_NORMALIZE

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define _CLC_SPIRV_NORMALIZE(VLEN)                                             \
  _CLC_SPIRV_NORMALIZE_IMPL(double, , long, VLEN, 0x1.0p+563, 0x1.0p-513)

_CLC_OVERLOAD _CLC_DEF double __spirv_ocl_normalize(double p) {
  return __spirv_ocl_sign(p);
}

_CLC_SPIRV_NORMALIZE(2)
_CLC_SPIRV_NORMALIZE(3)
_CLC_SPIRV_NORMALIZE(4)

#undef _CLC_SPIRV_NORMALIZE

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define _CLC_SPIRV_NORMALIZE(VLEN)                                             \
  _CLC_SPIRV_NORMALIZE_IMPL(half, h, short, VLEN, HALF_MAX_SQRT, HALF_MIN_SQRT)

_CLC_OVERLOAD _CLC_DEF half __spirv_ocl_normalize(half p) {
  return __spirv_ocl_sign(p);
}

_CLC_SPIRV_NORMALIZE(2)
_CLC_SPIRV_NORMALIZE(3)
_CLC_SPIRV_NORMALIZE(4)

#undef _CLC_SPIRV_NORMALIZE

#endif
