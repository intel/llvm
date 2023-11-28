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

#define AMDGPU_ATOMIC_SUB_IMPL(FUNC_NAME, TYPE, TYPE_MANGLED, AS, AS_MANGLED,                                                 \
                               NOT_GENERIC, BUILTIN)                                                                          \
  _CLC_DEF TYPE                                                                                                               \
      FUNC_NAME##P##AS_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagENS##NOT_GENERIC##_19MemorySemanticsMask4FlagE##TYPE_MANGLED( \
          volatile AS TYPE *p, enum Scope scope,                                                                              \
          enum MemorySemanticsMask semantics, TYPE val) {                                                                     \
    int atomic_scope = 0, memory_order = 0;                                                                                   \
    GET_ATOMIC_SCOPE_AND_ORDER(scope, atomic_scope, semantics, memory_order)                                                  \
    return BUILTIN(p, -val, memory_order);                                                                                    \
  }

#define AMDGPU_ATOMIC_SUB(FUNC_NAME, TYPE, TYPE_MANGLED, BUILTIN)              \
  AMDGPU_ATOMIC_SUB_IMPL(FUNC_NAME, TYPE, TYPE_MANGLED, global, U3AS1, 1,      \
                         BUILTIN)                                              \
  AMDGPU_ATOMIC_SUB_IMPL(FUNC_NAME, TYPE, TYPE_MANGLED, local, U3AS3, 1,       \
                         BUILTIN)                                              \
  AMDGPU_ATOMIC_SUB_IMPL(FUNC_NAME, TYPE, TYPE_MANGLED, , , 0, BUILTIN)

AMDGPU_ATOMIC_SUB(_Z18__spirv_AtomicISub, int, i, __atomic_fetch_add)
AMDGPU_ATOMIC_SUB(_Z18__spirv_AtomicISub, unsigned int, j, __atomic_fetch_add)
AMDGPU_ATOMIC_SUB(_Z18__spirv_AtomicISub, long, l, __atomic_fetch_add)
AMDGPU_ATOMIC_SUB(_Z18__spirv_AtomicISub, unsigned long, m, __atomic_fetch_add)

#undef AMDGPU_ATOMIC
#undef AMDGPU_ATOMIC_IMPL
#undef AMDGPU_ATOMIC_SUB
#undef AMDGPU_ATOMIC_SUB_IMPL
#undef AMDGPU_ARCH_GEQ
#undef AMDGPU_ARCH_BETWEEN
#undef GET_ATOMIC_SCOPE_AND_ORDER
