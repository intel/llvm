//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>
#include <spirv/spirv.h>

#define BUILTIN_FENCE(semantics, scope_memory)                                 \
  if (semantics & 0x2)                                                         \
    return __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, scope_memory);             \
  else if (semantics & 0x4)                                                    \
    return __builtin_amdgcn_fence(__ATOMIC_RELEASE, scope_memory);             \
  else if (semantics & 0x8)                                                    \
    return __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, scope_memory);             \
  else if (semantics & 0x10)                                                   \
    return __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, scope_memory);             \
  else                                                                         \
    return __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, scope_memory);

_CLC_DEF _CLC_OVERLOAD void __mem_fence(unsigned int scope_memory,
                                        unsigned int semantics) {
  switch (scope_memory) {
  default:
    BUILTIN_FENCE(semantics, "")
  case 1: // Device
    BUILTIN_FENCE(semantics, "agent")
  case 2: // Workgroup
    BUILTIN_FENCE(semantics, "workgroup")
  case 3: // Subgroup
    BUILTIN_FENCE(semantics, "wavefront")
  case 4: // Invocation
    BUILTIN_FENCE(semantics, "singlethread")
  }
}
#undef BUILTIN_FENCE

_CLC_OVERLOAD _CLC_DEF void __spirv_MemoryBarrier(unsigned int scope_memory,
                                                  unsigned int semantics) {
  __mem_fence(scope_memory, semantics);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT void
__spirv_ControlBarrier(unsigned int scope_execution, unsigned scope_memory,
                       unsigned int semantics) {
  if (semantics) {
    __mem_fence(scope_memory, semantics);
  }
  __builtin_amdgcn_s_barrier();
}
