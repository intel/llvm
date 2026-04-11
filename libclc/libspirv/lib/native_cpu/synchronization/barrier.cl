//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>
#include <libspirv/spirv_types.h>

// Native CPU barrier implementations using standard C atomics

_CLC_INLINE void builtin_fence_order(int scope_memory, int order) {
  // Map SPIR-V memory order to C atomic order
  int atomic_order = __ATOMIC_SEQ_CST; // Default to strongest ordering

  switch (order & 0x1F) {
  case None:
    atomic_order = __ATOMIC_RELAXED;
    break;
  case Acquire:
    atomic_order = __ATOMIC_ACQUIRE;
    break;
  case Release:
    atomic_order = __ATOMIC_RELEASE;
    break;
  case AcquireRelease:
    atomic_order = __ATOMIC_ACQ_REL;
    break;
  case SequentiallyConsistent:
    atomic_order = __ATOMIC_SEQ_CST;
    break;
  default:
    atomic_order = __ATOMIC_SEQ_CST;
    break;
  }

  __atomic_thread_fence(atomic_order);
}

_CLC_DEF _CLC_OVERLOAD void __mem_fence(int scope_memory, int semantics) {
  builtin_fence_order(scope_memory, semantics & 0x1F);
}

_CLC_OVERLOAD _CLC_DEF void __spirv_MemoryBarrier(int scope_memory,
                                                   int semantics) {
  __mem_fence(scope_memory, semantics);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT void
__spirv_ControlBarrier(int scope_execution, int scope_memory, int semantics) {
  // For NativeCPU, barriers are handled by the runtime
  // We just need to provide a memory fence
  if (semantics) {
    __mem_fence(scope_memory, semantics);
  }

  // The actual thread synchronization is handled by the NativeCPU runtime
  // through the __mux_work_group_barrier builtin, which gets lowered later
  // For now, we just ensure memory ordering with a full fence
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
}
