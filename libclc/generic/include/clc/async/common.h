//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLC_ASYNC_COMMON
#define CLC_ASYNC_COMMON

#define SET_GROUP_SIZE_AND_ID(SIZE, ID)                                        \
  if (scope == Workgroup) {                                                    \
    SIZE = __spirv_WorkgroupSize_x() * __spirv_WorkgroupSize_y() *             \
           __spirv_WorkgroupSize_z();                                          \
    ID = (__spirv_WorkgroupSize_y() * __spirv_WorkgroupSize_x() *              \
          __spirv_LocalInvocationId_z()) +                                     \
         (__spirv_WorkgroupSize_x() * __spirv_LocalInvocationId_y()) +         \
         __spirv_LocalInvocationId_x();                                        \
  } else {                                                                     \
    SIZE = __spirv_SubgroupSize();                                             \
    ID = __spirv_SubgroupLocalInvocationId();                                  \
  }
// Macro used by all data types, for generic and nvidia, for async copy when
// arch < sm80
#define STRIDED_COPY(DST_AS, SRC_AS, DST_STRIDE, SRC_STRIDE)                   \
  size_t size, id;                                                             \
  SET_GROUP_SIZE_AND_ID(size, id);                                             \
  for (size_t i = id; i < num_gentypes; i += size) {                           \
    dst[i * DST_STRIDE] = src[i * SRC_STRIDE];                                 \
  }

#endif // CLC_ASYNC_COMMON