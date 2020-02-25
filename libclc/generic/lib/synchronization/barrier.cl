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
  unsigned int mem_semantic = (flag & CLK_GLOBAL_MEM_FENCE ? 0x200 : 0) |
                              (flag & CLK_LOCAL_MEM_FENCE ? 0x100 : 0)
  // TODO: Stop manually mangling this name. Need C++ namespaces to get the exact mangling.
  _Z22__spirv_ControlBarrierN5__spv5ScopeES0_j(Workgroup, Workgroup, mem_semantic);
}
