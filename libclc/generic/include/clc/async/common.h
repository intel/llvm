//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLC_ASYNC_COMMON
#define CLC_ASYNC_COMMON

// Macro used by all data types, for generic and nvidia, for async copy when
// arch < sm80
#define STRIDED_COPY(DST_AS, SRC_AS, DST_STRIDE, SRC_STRIDE)                   \
  size_t size, id;                                                             \
  if (scope == Workgroup) {                                                    \
    size = __spirv_WorkgroupSize_x() * __spirv_WorkgroupSize_y() *             \
           __spirv_WorkgroupSize_z();                                          \
    id = (__spirv_WorkgroupSize_y() * __spirv_WorkgroupSize_x() *              \
          __spirv_LocalInvocationId_z()) +                                     \
         (__spirv_WorkgroupSize_x() * __spirv_LocalInvocationId_y()) +         \
         __spirv_LocalInvocationId_x();                                        \
  } else {                                                                     \
    size = __spirv_SubgroupSize();                                             \
    id = __spirv_SubgroupLocalInvocationId();                                  \
  }                                                                            \
  size_t i;                                                                    \
                                                                               \
  for (i = id; i < num_gentypes; i += size) {                                  \
    dst[i * DST_STRIDE] = src[i * SRC_STRIDE];                                 \
  }

#endif // CLC_ASYNC_COMMON
