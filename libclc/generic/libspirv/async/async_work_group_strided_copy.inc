//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DEF event_t
__spirv_GroupAsyncCopy(unsigned int scope, global __CLC_GENTYPE *dst,
                       const local __CLC_GENTYPE *src, size_t num_gentypes,
                       size_t stride, event_t event) {
  STRIDED_COPY(global, local, stride, 1);
  return event;
}

_CLC_OVERLOAD _CLC_DEF event_t
__spirv_GroupAsyncCopy(unsigned int scope, local __CLC_GENTYPE *dst,
                       const global __CLC_GENTYPE *src, size_t num_gentypes,
                       size_t stride, event_t event) {
  STRIDED_COPY(local, global, 1, stride);
  return event;
}
