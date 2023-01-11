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

#define AMDGPU_ATOMIC_STORE(FUNC_NAME, TYPE, TYPE_MANGLED, AS, AS_MANGLED)                                        \
  _CLC_DEF void                                                                                                   \
      FUNC_NAME##PU3##AS_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE##TYPE_MANGLED( \
          volatile AS TYPE *p, enum Scope scope,                                                                  \
          enum MemorySemanticsMask semantics, TYPE val) {                                                         \
    int atomic_scope = 0, memory_order = 0;                                                                       \
    GET_ATOMIC_SCOPE_AND_ORDER(scope, atomic_scope, semantics, memory_order)                                      \
    __hip_atomic_store(p, val, memory_order, atomic_scope);                                                       \
    return;                                                                                                       \
  }

AMDGPU_ATOMIC_STORE(_Z19__spirv_AtomicStore, int, i, global, AS1)
AMDGPU_ATOMIC_STORE(_Z19__spirv_AtomicStore, unsigned int, j, global, AS1)
AMDGPU_ATOMIC_STORE(_Z19__spirv_AtomicStore, int, i, local, AS3)
AMDGPU_ATOMIC_STORE(_Z19__spirv_AtomicStore, unsigned int, j, local, AS3)

#ifdef cl_khr_int64_base_atomics
AMDGPU_ATOMIC_STORE(_Z19__spirv_AtomicStore, long, l, global, AS1)
AMDGPU_ATOMIC_STORE(_Z19__spirv_AtomicStore, unsigned long, m, global, AS1)
AMDGPU_ATOMIC_STORE(_Z19__spirv_AtomicStore, long, l, local, AS3)
AMDGPU_ATOMIC_STORE(_Z19__spirv_AtomicStore, unsigned long, m, local, AS3)
#endif

AMDGPU_ATOMIC_STORE(_Z19__spirv_AtomicStore, float, f, global, AS1)
AMDGPU_ATOMIC_STORE(_Z19__spirv_AtomicStore, float, f, local, AS3)

// TODO implement for fp64

#undef AMDGPU_ATOMIC_STORE
