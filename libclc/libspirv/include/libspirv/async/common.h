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
  SIZE = __spirv_BuiltInWorkgroupSize(0) * __spirv_BuiltInWorkgroupSize(1) *   \
         __spirv_BuiltInWorkgroupSize(2);                                      \
  ID = (__spirv_BuiltInWorkgroupSize(1) * __spirv_BuiltInWorkgroupSize(0) *    \
        __spirv_BuiltInLocalInvocationId(2)) +                                 \
       (__spirv_BuiltInWorkgroupSize(0) *                                      \
        __spirv_BuiltInLocalInvocationId(1)) +                                 \
       __spirv_BuiltInLocalInvocationId(0);

// Macro used by all data types, for generic and nvidia, for async copy when
// arch < sm80
#define STRIDED_COPY(DST_AS, SRC_AS, DST_STRIDE, SRC_STRIDE)                   \
  size_t size, id;                                                             \
  SET_GROUP_SIZE_AND_ID(size, id);                                             \
  for (size_t i = id; i < num_gentypes; i += size) {                           \
    dst[i * DST_STRIDE] = src[i * SRC_STRIDE];                                 \
  }

#endif // CLC_ASYNC_COMMON
