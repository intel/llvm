//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define FN_NAME __CLC_XCONCAT(_Z22__spirv_GroupAsyncCopyI, __SPIRV_GENTYPE_MANGLED, E9ocl_eventN5__spv5ScopeEPU3, __SPIRV_DST_ADDR_SPACE_MANGLED, T_PU3, __SPIRV_SRC_ADDR_SPACE_MANGLED, S3_mmS0_)
_CLC_DECL event_t FN_NAME (
  enum Scope scope,
  __SPIRV_DST_ADDR_SPACE __SPIRV_GENTYPE *dst,
  const __SPIRV_SRC_ADDR_SPACE __SPIRV_GENTYPE *src,
  size_t num_elements,
  size_t stride,
  event_t event);
