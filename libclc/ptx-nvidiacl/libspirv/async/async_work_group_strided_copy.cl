//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/async/common.h>
#include <spirv/spirv.h>

#define __CLC_BODY                                                             \
<../../../generic/libspirv/async/async_work_group_strided_copy.inc>
#define __CLC_GEN_VEC3
#include "../../include/clc/async/gentype.inc"
#undef __CLC_GEN_VEC3
#undef __CLC_BODY

int __clc_nvvm_reflect_arch();

#define __CLC_GROUP_CP_ASYNC_DST_GLOBAL(TYPE)                                  \
  _CLC_OVERLOAD _CLC_DEF event_t __spirv_GroupAsyncCopy(                       \
      unsigned int scope, __attribute__((address_space(1))) TYPE *dst,         \
      const __attribute__((address_space(3))) TYPE *src, size_t num_gentypes,  \
      size_t stride, event_t event) {                                          \
    STRIDED_COPY(__attribute__((address_space(1))),                            \
                 __attribute__((address_space(3))), stride, 1);                \
    return event;                                                              \
  }

__CLC_GROUP_CP_ASYNC_DST_GLOBAL(int);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(int2);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(int4);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(uint);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(uint2);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(uint4);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(float);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(float2);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(float4);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(long);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(long2);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(ulong);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(ulong2);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(double);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(double2);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(half2);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(half4);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(half8);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(short2);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(short4);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(short8);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(ushort2);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(ushort4);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(ushort8);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(char4);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(char8);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(char16);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(schar4);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(schar8);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(schar16);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(uchar4);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(uchar8);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(uchar16);

#undef __CLC_GROUP_CP_ASYNC_DST_GLOBAL

#define __CLC_GROUP_CP_ASYNC_4(TYPE)                                           \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT event_t __spirv_GroupAsyncCopy(       \
      unsigned int scope, __attribute__((address_space(3))) TYPE *dst,         \
      const __attribute__((address_space(1))) TYPE *src, size_t num_gentypes,  \
      size_t stride, event_t event) {                                          \
    if (__clc_nvvm_reflect_arch() >= 800) {                                    \
      size_t id, size;                                                         \
      SET_GROUP_SIZE_AND_ID(size, id);                                         \
      for (size_t i = id; i < num_gentypes; i += size) {                       \
        __nvvm_cp_async_ca_shared_global_4(dst + i, src + i * stride);         \
      }                                                                        \
      __nvvm_cp_async_commit_group();                                          \
    } else {                                                                   \
      STRIDED_COPY(__attribute__((address_space(3))),                          \
                   __attribute__((address_space(1))), 1, stride);              \
    }                                                                          \
    return event;                                                              \
  }

__CLC_GROUP_CP_ASYNC_4(int);
__CLC_GROUP_CP_ASYNC_4(uint);
__CLC_GROUP_CP_ASYNC_4(float);
__CLC_GROUP_CP_ASYNC_4(short2);
__CLC_GROUP_CP_ASYNC_4(ushort2);
__CLC_GROUP_CP_ASYNC_4(half2);
__CLC_GROUP_CP_ASYNC_4(char4);
__CLC_GROUP_CP_ASYNC_4(schar4);
__CLC_GROUP_CP_ASYNC_4(uchar4);

#undef __CLC_GROUP_CP_ASYNC_4

#define __CLC_GROUP_CP_ASYNC_8(TYPE)                                           \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT event_t __spirv_GroupAsyncCopy(       \
      unsigned int scope, __attribute__((address_space(3))) TYPE *dst,         \
      const __attribute__((address_space(1))) TYPE *src, size_t num_gentypes,  \
      size_t stride, event_t event) {                                          \
    if (__clc_nvvm_reflect_arch() >= 800) {                                    \
      size_t id, size;                                                         \
      SET_GROUP_SIZE_AND_ID(size, id);                                         \
      for (size_t i = id; i < num_gentypes; i += size) {                       \
        __nvvm_cp_async_ca_shared_global_8(dst + i, src + i * stride);         \
      }                                                                        \
      __nvvm_cp_async_commit_group();                                          \
    } else {                                                                   \
      STRIDED_COPY(__attribute__((address_space(3))),                          \
                   __attribute__((address_space(1))), 1, stride);              \
    }                                                                          \
    return event;                                                              \
  }

__CLC_GROUP_CP_ASYNC_8(long);
__CLC_GROUP_CP_ASYNC_8(ulong);
__CLC_GROUP_CP_ASYNC_8(double);
__CLC_GROUP_CP_ASYNC_8(short4);
__CLC_GROUP_CP_ASYNC_8(ushort4);
__CLC_GROUP_CP_ASYNC_8(half4);
__CLC_GROUP_CP_ASYNC_8(int2);
__CLC_GROUP_CP_ASYNC_8(uint2);
__CLC_GROUP_CP_ASYNC_8(float2);
__CLC_GROUP_CP_ASYNC_8(char8);
__CLC_GROUP_CP_ASYNC_8(schar8);
__CLC_GROUP_CP_ASYNC_8(uchar8);

#undef __CLC_GROUP_CP_ASYNC_8

#define __CLC_GROUP_CP_ASYNC_16(TYPE)                                          \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT event_t __spirv_GroupAsyncCopy(       \
      unsigned int scope, __attribute__((address_space(3))) TYPE *dst,         \
      const __attribute__((address_space(1))) TYPE *src, size_t num_gentypes,  \
      size_t stride, event_t event) {                                          \
    if (__clc_nvvm_reflect_arch() >= 800) {                                    \
      size_t id, size;                                                         \
      SET_GROUP_SIZE_AND_ID(size, id);                                         \
      for (size_t i = id; i < num_gentypes; i += size) {                       \
        __nvvm_cp_async_ca_shared_global_16(dst + i, src + i * stride);        \
      }                                                                        \
      __nvvm_cp_async_commit_group();                                          \
    } else {                                                                   \
      STRIDED_COPY(__attribute__((address_space(3))),                          \
                   __attribute__((address_space(1))), 1, stride);              \
    }                                                                          \
    return event;                                                              \
  }

  __CLC_GROUP_CP_ASYNC_16(int4);
  __CLC_GROUP_CP_ASYNC_16(uint4);
  __CLC_GROUP_CP_ASYNC_16(float4);
  __CLC_GROUP_CP_ASYNC_16(long2);
  __CLC_GROUP_CP_ASYNC_16(ulong2);
  __CLC_GROUP_CP_ASYNC_16(double2);
  __CLC_GROUP_CP_ASYNC_16(short8);
  __CLC_GROUP_CP_ASYNC_16(ushort8);
  __CLC_GROUP_CP_ASYNC_16(half8);
  __CLC_GROUP_CP_ASYNC_16(char16);
  __CLC_GROUP_CP_ASYNC_16(uchar16);
  __CLC_GROUP_CP_ASYNC_16(schar16);

#undef __CLC_GROUP_CP_ASYNC_16
