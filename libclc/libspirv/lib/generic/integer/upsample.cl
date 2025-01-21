//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <core/clc_core.h>
#include <libspirv/spirv.h>

#define __CLC_UPSAMPLE_IMPL(PREFIX, BGENTYPE, GENTYPE, UGENTYPE)               \
  _CLC_OVERLOAD _CLC_DEF BGENTYPE __spirv_ocl_##PREFIX##_upsample(             \
      GENTYPE hi, UGENTYPE lo) {                                               \
    return __clc_upsample(hi, lo);                                             \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DEF BGENTYPE##2 __spirv_ocl_##PREFIX##_upsample(          \
      GENTYPE##2 hi, UGENTYPE##2 lo) {                                         \
    return __clc_upsample(hi, lo);                                             \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DEF BGENTYPE##3 __spirv_ocl_##PREFIX##_upsample(          \
      GENTYPE##3 hi, UGENTYPE##3 lo) {                                         \
    return __clc_upsample(hi, lo);                                             \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DEF BGENTYPE##4 __spirv_ocl_##PREFIX##_upsample(          \
      GENTYPE##4 hi, UGENTYPE##4 lo) {                                         \
    return __clc_upsample(hi, lo);                                             \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DEF BGENTYPE##8 __spirv_ocl_##PREFIX##_upsample(          \
      GENTYPE##8 hi, UGENTYPE##8 lo) {                                         \
    return __clc_upsample(hi, lo);                                             \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DEF BGENTYPE##16 __spirv_ocl_##PREFIX##_upsample(         \
      GENTYPE##16 hi, UGENTYPE##16 lo) {                                       \
    return __clc_upsample(hi, lo);                                             \
  }

#define __CLC_UPSAMPLE_TYPES()                                                 \
  __CLC_UPSAMPLE_IMPL(s, short, char, uchar)                                   \
  __CLC_UPSAMPLE_IMPL(s, short, schar, uchar)                                  \
  __CLC_UPSAMPLE_IMPL(u, ushort, uchar, uchar)                                 \
  __CLC_UPSAMPLE_IMPL(s, int, short, ushort)                                   \
  __CLC_UPSAMPLE_IMPL(u, uint, ushort, ushort)                                 \
  __CLC_UPSAMPLE_IMPL(s, long, int, uint)                                      \
  __CLC_UPSAMPLE_IMPL(u, ulong, uint, uint)

__CLC_UPSAMPLE_TYPES()

#undef __CLC_UPSAMPLE_TYPES
#undef __CLC_UPSAMPLE_IMPL
