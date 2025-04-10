//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "atomic_helpers.h"
#include <libspirv/spirv.h>
#include <libspirv/spirv_types.h>

#define AMDGPU_ATOMIC_CMPXCHG_IMPL(TYPE, AS)                                   \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_AtomicCompareExchange(                   \
      AS TYPE *p, int scope, int success_semantics, int failure_semantics,     \
      TYPE desired, TYPE expected) {                                           \
    int atomic_scope = 0, memory_order_success = 0, memory_order_failure = 0;  \
    GET_ATOMIC_SCOPE_AND_ORDER(scope, atomic_scope, success_semantics,         \
                               memory_order_success)                           \
    GET_ATOMIC_SCOPE_AND_ORDER(scope, atomic_scope, failure_semantics,         \
                               memory_order_failure)                           \
    __hip_atomic_compare_exchange_strong(p, &expected, desired,                \
                                         memory_order_success,                 \
                                         memory_order_failure, atomic_scope);  \
    /* If cmpxchg                                                              \
     *  succeeds:                                                              \
         - `expected` is unchanged, holding the old val that was at `p`        \
         - `p` is changed to hold `desired`                                    \
     *  fails:                                                                 \
         - `expected` is changed to hold the current val at `p`                \
         - `p` is unchanged*/                                                  \
    return expected;                                                           \
  }

#define AMDGPU_ATOMIC_CMPXCHG(TYPE)                                            \
  AMDGPU_ATOMIC_CMPXCHG_IMPL(TYPE, global)                                     \
  AMDGPU_ATOMIC_CMPXCHG_IMPL(TYPE, local)                                      \
  AMDGPU_ATOMIC_CMPXCHG_IMPL(TYPE, )

AMDGPU_ATOMIC_CMPXCHG(int)
AMDGPU_ATOMIC_CMPXCHG(unsigned)
AMDGPU_ATOMIC_CMPXCHG(long)
AMDGPU_ATOMIC_CMPXCHG(unsigned long)
AMDGPU_ATOMIC_CMPXCHG(float)

// TODO implement for fp64

#undef AMDGPU_ATOMIC
#undef AMDGPU_ATOMIC_IMPL
#undef AMDGPU_ATOMIC_CPMXCHG
#undef AMDGPU_ATOMIC_CPMXCHG_IMPL
#undef AMDGPU_ARCH_GEQ
#undef AMDGPU_ARCH_BETWEEN
#undef GET_ATOMIC_SCOPE_AND_ORDER
