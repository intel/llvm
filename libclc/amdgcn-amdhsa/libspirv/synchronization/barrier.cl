//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>
#include <spirv/spirv.h>
#include <spirv/spirv_types.h>

#define BUILTIN_FENCE(semantics, scope_memory)                                 \
  if (semantics & Acquire)                                                     \
    return __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, scope_memory);             \
  else if (semantics & Release)                                                \
    return __builtin_amdgcn_fence(__ATOMIC_RELEASE, scope_memory);             \
  else if (semantics & AcquireRelease)                                         \
    return __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, scope_memory);             \
  else if (semantics & SequentiallyConsistent)                                 \
    return __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, scope_memory);             \
  else                                                                         \
    return __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, scope_memory);

_CLC_DEF _CLC_OVERLOAD void __mem_fence(unsigned int scope_memory,
                                        unsigned int semantics) {
  switch ((enum Scope)scope_memory) {
  case CrossDevice:
    BUILTIN_FENCE(semantics, "")
  case Device:
    BUILTIN_FENCE(semantics, "agent")
  case Workgroup:
    BUILTIN_FENCE(semantics, "workgroup")
  case Subgroup:
    BUILTIN_FENCE(semantics, "wavefront")
  case Invocation:
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
