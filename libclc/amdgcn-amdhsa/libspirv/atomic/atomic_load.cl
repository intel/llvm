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

#define AMDGPU_ATOMIC_LOAD(FUNC_NAME, TYPE, TYPE_MANGLED, AS, AS_MANGLED)                           \
  _CLC_DEF TYPE                                                                                     \
      FUNC_NAME##PU3##AS_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE( \
          const volatile AS TYPE *p, enum Scope scope,                                              \
          enum MemorySemanticsMask semantics) {                                                     \
    int atomic_scope = 0, memory_order = 0;                                                         \
    GET_ATOMIC_SCOPE_AND_ORDER(scope, atomic_scope, semantics, memory_order)                        \
    TYPE res = __hip_atomic_load(p, memory_order, atomic_scope);                                    \
    return *(TYPE *)&res;                                                                           \
  }

AMDGPU_ATOMIC_LOAD(_Z18__spirv_AtomicLoad, int, Ki, global, AS1)
AMDGPU_ATOMIC_LOAD(_Z18__spirv_AtomicLoad, unsigned int, Kj, global, AS1)
AMDGPU_ATOMIC_LOAD(_Z18__spirv_AtomicLoad, int, Ki, local, AS3)
AMDGPU_ATOMIC_LOAD(_Z18__spirv_AtomicLoad, unsigned int, Kj, local, AS3)

#ifdef cl_khr_int64_base_atomics
AMDGPU_ATOMIC_LOAD(_Z18__spirv_AtomicLoad, long, Kl, global, AS1)
AMDGPU_ATOMIC_LOAD(_Z18__spirv_AtomicLoad, unsigned long, Km, global, AS1)
AMDGPU_ATOMIC_LOAD(_Z18__spirv_AtomicLoad, long, Kl, local, AS3)
AMDGPU_ATOMIC_LOAD(_Z18__spirv_AtomicLoad, unsigned long, Km, local, AS3)
#endif

AMDGPU_ATOMIC_LOAD(_Z18__spirv_AtomicLoad, float, Kf, global, AS1)
AMDGPU_ATOMIC_LOAD(_Z18__spirv_AtomicLoad, float, Kf, local, AS3)

// TODO implement for fp64

#undef AMDGPU_ATOMIC
