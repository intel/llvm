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

#define AMDGPU_ATOMIC_LOAD_IMPL(TYPE, TYPE_MANGLED, AS, AS_MANGLED)                                          \
  _CLC_DEF TYPE                                                                                              \
      _Z18__spirv_AtomicLoadP##AS_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE( \
          const volatile AS TYPE *p, enum Scope scope,                                                       \
          enum MemorySemanticsMask semantics) {                                                              \
    int atomic_scope = 0, memory_order = 0;                                                                  \
    GET_ATOMIC_SCOPE_AND_ORDER(scope, atomic_scope, semantics, memory_order)                                 \
    return __hip_atomic_load(p, memory_order, atomic_scope);                                                 \
  }

#define AMDGPU_ATOMIC_LOAD(TYPE, TYPE_MANGLED)                                 \
  AMDGPU_ATOMIC_LOAD_IMPL(TYPE, TYPE_MANGLED, global, U3AS1)                   \
  AMDGPU_ATOMIC_LOAD_IMPL(TYPE, TYPE_MANGLED, local, U3AS3)                    \
  AMDGPU_ATOMIC_LOAD_IMPL(TYPE, TYPE_MANGLED, , )

AMDGPU_ATOMIC_LOAD(int, Ki)
AMDGPU_ATOMIC_LOAD(unsigned int, Kj)
AMDGPU_ATOMIC_LOAD(long, Kl)
AMDGPU_ATOMIC_LOAD(unsigned long, Km)
AMDGPU_ATOMIC_LOAD(float, Kf)

// TODO implement for fp64

#undef AMDGPU_ATOMIC
#undef AMDGPU_ATOMIC_IMPL
#undef AMDGPU_ATOMIC_LOAD
#undef AMDGPU_ATOMIC_LOAD_IMPL
#undef AMDGPU_ARCH_GEQ
#undef AMDGPU_ARCH_BETWEEN
#undef GET_ATOMIC_SCOPE_AND_ORDER
