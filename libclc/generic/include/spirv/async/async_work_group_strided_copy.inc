//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DECL event_t __spirv_GroupAsyncCopy(
    unsigned int scope, __SPIRV_DST_ADDR_SPACE __SPIRV_GENTYPE *dst,
    const __SPIRV_SRC_ADDR_SPACE __SPIRV_GENTYPE *src, size_t num_elements,
    size_t stride, event_t event);
