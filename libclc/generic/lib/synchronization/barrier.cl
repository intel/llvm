//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>
#include <clc/clc.h>

_CLC_DEF void barrier(cl_mem_fence_flags flags) {
  unsigned int mem_semantic =
      SequentiallyConsistent |
      (flag & CLK_GLOBAL_MEM_FENCE ? CrossWorkgroupMemory : 0) |
      (flag & CLK_LOCAL_MEM_FENCE ? WorkgroupMemory : 0)
          __spirv_ControlBarrier(Workgroup, Workgroup, SequentiallyConsistent);
}
