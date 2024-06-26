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


#define BUILTIN_FENCE(order, scope_memory)                                     \
  /* None implies Monotonic (for llvm/AMDGPU), or relaxed in C++.              \
   * This does not make sense as ordering argument for a fence instruction     \
   * and is not part of the supported orderings for a fence in AMDGPU. */      \
  if (order != None) {                                                         \
    switch (order) {                                                           \
    case Acquire:                                                              \
      return __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, scope_memory);           \
    case Release:                                                              \
      return __builtin_amdgcn_fence(__ATOMIC_RELEASE, scope_memory);           \
    case AcquireRelease:                                                       \
      return __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, scope_memory);           \
    case SequentiallyConsistent:                                               \
      return __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, scope_memory);           \
    default:                                                                   \
      __builtin_trap();                                                        \
      __builtin_unreachable();                                                 \
    }                                                                          \
  }

_CLC_INLINE void builtin_fence_order(unsigned int scope_memory,
                                     unsigned int order) {
  switch ((enum Scope)scope_memory) {
  case CrossDevice:
    BUILTIN_FENCE(order, "")
  case Device:
    BUILTIN_FENCE(order, "agent")
  case Workgroup:
    BUILTIN_FENCE(order, "workgroup")
  case Subgroup:
    BUILTIN_FENCE(order, "wavefront")
  case Invocation:
    BUILTIN_FENCE(order, "singlethread")
  }
}
#undef BUILTIN_FENCE

_CLC_DEF _CLC_OVERLOAD void __mem_fence(unsigned int scope_memory,
                                        unsigned int semantics) {
  builtin_fence_order(scope_memory, semantics & 0x1F);
}

_CLC_OVERLOAD _CLC_DEF void __spirv_MemoryBarrier(unsigned int scope_memory,
                                                  unsigned int semantics) {
  __mem_fence(scope_memory, semantics);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT void
__spirv_ControlBarrier(unsigned int scope_execution, unsigned int scope_memory,
                       unsigned int semantics) {
  if (semantics) {
    __mem_fence(scope_memory, semantics);
  }
  __builtin_amdgcn_s_barrier();
}
