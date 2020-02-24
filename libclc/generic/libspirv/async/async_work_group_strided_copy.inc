//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define STRIDED_COPY(DST_AS, SRC_AS, DST_STRIDE, SRC_STRIDE)               \
  size_t size = __spirv_LocalInvocationId_x() *                            \
                __spirv_LocalInvocationId_y() *                            \
                __spirv_LocalInvocationId_z();                             \
  size_t id = (__spirv_WorkgroupSize_y() * __spirv_WorkgroupSize_z() *     \
               __spirv_LocalInvocationId_x()) +                            \
              (__spirv_WorkgroupSize_z() *                                 \
               __spirv_LocalInvocationId_y()) +                            \
              __spirv_LocalInvocationId_z();                               \
  size_t i;                                                                \
                                                                           \
  for (i = id; i < num_gentypes; i += size) {                              \
    dst[i * DST_STRIDE] = src[i * SRC_STRIDE];                             \
  }

#define __CLC_CONCAT(a, b, c) a ## b ## c
#define __CLC_XCONCAT(a, b, c) __CLC_CONCAT(a, b, c)

_CLC_DEF event_t __CLC_XCONCAT(_Z22__spirv_GroupAsyncCopyI, __CLC_GENTYPE_MANGLED, E9ocl_eventN5__spv5ScopeEPU3AS1T_PU3AS3S3_mmS0_) (
    enum Scope scope,
    global __CLC_GENTYPE *dst,
    const local __CLC_GENTYPE *src,
    size_t num_gentypes,
    size_t stride,
    event_t event) {
  STRIDED_COPY(global, local, stride, 1);
  return event;
}

_CLC_DEF event_t __CLC_XCONCAT(_Z22__spirv_GroupAsyncCopyI, __CLC_GENTYPE_MANGLED, E9ocl_eventN5__spv5ScopeEPU3AS3T_PU3AS1S3_mmS0_) (
    enum Scope scope,
    local __CLC_GENTYPE *dst,
    const global __CLC_GENTYPE *src,
    size_t num_gentypes,
    size_t stride,
    event_t event) {
  STRIDED_COPY(local, global, 1, stride);
  return event;
}

#undef __CLC_XCONCAT
#undef __CLC_CONCAT
