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

#define AMDGPU_ATOMIC_STORE_IMPL(TYPE, TYPE_MANGLED, AS, AS_MANGLED, SUB1)                                                           \
  _CLC_DEF void                                                                                                                      \
      _Z19__spirv_AtomicStore##P##AS_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagENS##SUB1##_19MemorySemanticsMask4FlagE##TYPE_MANGLED( \
          volatile AS TYPE *p, enum Scope scope,                                                                                     \
          enum MemorySemanticsMask semantics, TYPE val) {                                                                            \
    int atomic_scope = 0, memory_order = 0;                                                                                          \
    GET_ATOMIC_SCOPE_AND_ORDER(scope, atomic_scope, semantics, memory_order)                                                         \
    __hip_atomic_store(p, val, memory_order, atomic_scope);                                                                          \
    return;                                                                                                                          \
  }

#define AMDGPU_ATOMIC_STORE(TYPE, TYPE_MANGLED)                                \
  AMDGPU_ATOMIC_STORE_IMPL(TYPE, TYPE_MANGLED, global, U3AS1, 1)               \
  AMDGPU_ATOMIC_STORE_IMPL(TYPE, TYPE_MANGLED, local, U3AS3, 1)                \
  AMDGPU_ATOMIC_STORE_IMPL(TYPE, TYPE_MANGLED, , , 0)

AMDGPU_ATOMIC_STORE(int, i)
AMDGPU_ATOMIC_STORE(unsigned int, j)
AMDGPU_ATOMIC_STORE(long, l)
AMDGPU_ATOMIC_STORE(unsigned long, m)
AMDGPU_ATOMIC_STORE(float, f)

// TODO implement for fp64

#undef AMDGPU_ATOMIC
#undef AMDGPU_ATOMIC_IMPL
#undef AMDGPU_ATOMIC_STORE
#undef AMDGPU_ATOMIC_STORE_IMPL
#undef AMDGPU_ARCH_GEQ
#undef AMDGPU_ARCH_BETWEEN
#undef GET_ATOMIC_SCOPE_AND_ORDER
