//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "atomic_helpers.h"
#include <spirv/spirv.h>
#include <spirv/spirv_types.h>

#define AMDGPU_ATOMIC_CMPXCHG_IMPL(TYPE, TYPE_MANGLED, AS, AS_MANGLED, SUB1,                                                                                         \
                                   SUB2)                                                                                                                             \
  _CLC_DEF TYPE                                                                                                                                                      \
      _Z29__spirv_AtomicCompareExchangeP##AS_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagENS##SUB1##_19MemorySemanticsMask4FlagES##SUB2##_##TYPE_MANGLED##TYPE_MANGLED( \
          volatile AS TYPE *p, enum Scope scope,                                                                                                                     \
          enum MemorySemanticsMask success_semantics,                                                                                                                \
          enum MemorySemanticsMask failure_semantics, TYPE desired,                                                                                                  \
          TYPE expected) {                                                                                                                                           \
    int atomic_scope = 0, memory_order_success = 0, memory_order_failure = 0;                                                                                        \
    GET_ATOMIC_SCOPE_AND_ORDER(scope, atomic_scope, success_semantics,                                                                                               \
                               memory_order_success)                                                                                                                 \
    GET_ATOMIC_SCOPE_AND_ORDER(scope, atomic_scope, failure_semantics,                                                                                               \
                               memory_order_failure)                                                                                                                 \
    __hip_atomic_compare_exchange_strong(p, &expected, desired,                                                                                                      \
                                         memory_order_success,                                                                                                       \
                                         memory_order_failure, atomic_scope);                                                                                        \
    /* If cmpxchg                                                                                                                                                    \
     *  succeeds:                                                                                                                                                    \
         - `expected` is unchanged, holding the old val that was at `p`                                                                                              \
         - `p` is changed to hold `desired`                                                                                                                          \
     *  fails:                                                                                                                                                       \
         - `expected` is changed to hold the current val at `p`                                                                                                      \
         - `p` is unchanged*/                                                                                                                                        \
    return expected;                                                                                                                                                 \
  }

#define AMDGPU_ATOMIC_CMPXCHG(TYPE, TYPE_MANGLED)                              \
  AMDGPU_ATOMIC_CMPXCHG_IMPL(TYPE, TYPE_MANGLED, global, U3AS1, 1, 5)          \
  AMDGPU_ATOMIC_CMPXCHG_IMPL(TYPE, TYPE_MANGLED, local, U3AS3, 1, 5)           \
  AMDGPU_ATOMIC_CMPXCHG_IMPL(TYPE, TYPE_MANGLED, , , 0, 4)

AMDGPU_ATOMIC_CMPXCHG(int, i)
AMDGPU_ATOMIC_CMPXCHG(unsigned, j)
AMDGPU_ATOMIC_CMPXCHG(long, l)
AMDGPU_ATOMIC_CMPXCHG(unsigned long, m)
AMDGPU_ATOMIC_CMPXCHG(float, f)

// TODO implement for fp64

#undef AMDGPU_ATOMIC
#undef AMDGPU_ATOMIC_IMPL
#undef AMDGPU_ATOMIC_CPMXCHG
#undef AMDGPU_ATOMIC_CPMXCHG_IMPL
#undef AMDGPU_ARCH_GEQ
#undef AMDGPU_ARCH_BETWEEN
#undef GET_ATOMIC_SCOPE_AND_ORDER
