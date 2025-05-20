//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/clc.h>

// We don't have separate mechanism for read and write fences
_CLC_DEF _CLC_OVERLOAD void read_mem_fence(cl_mem_fence_flags flags) {
  __spirv_MemoryBarrier(flags, 1);
}

_CLC_DEF _CLC_OVERLOAD void write_mem_fence(cl_mem_fence_flags flags) {
  __spirv_MemoryBarrier(flags, 1);
}
