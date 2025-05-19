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

#define AMDGPU_ATOMIC_STORE_IMPL(TYPE, AS)                                     \
  _CLC_OVERLOAD _CLC_DEF void __spirv_AtomicStore(AS TYPE *p, int scope,       \
                                                  int semantics, TYPE val) {   \
    int atomic_scope = 0, memory_order = 0;                                    \
    GET_ATOMIC_SCOPE_AND_ORDER(scope, atomic_scope, semantics, memory_order)   \
    __hip_atomic_store(p, val, memory_order, atomic_scope);                    \
    return;                                                                    \
  }

#define AMDGPU_ATOMIC_STORE(TYPE)                                              \
  AMDGPU_ATOMIC_STORE_IMPL(TYPE, global)                                       \
  AMDGPU_ATOMIC_STORE_IMPL(TYPE, local)                                        \
  AMDGPU_ATOMIC_STORE_IMPL(TYPE, )

AMDGPU_ATOMIC_STORE(int)
AMDGPU_ATOMIC_STORE(long)
AMDGPU_ATOMIC_STORE(float)

// TODO implement for fp64

#undef AMDGPU_ATOMIC
#undef AMDGPU_ATOMIC_IMPL
#undef AMDGPU_ATOMIC_STORE
#undef AMDGPU_ATOMIC_STORE_IMPL
#undef AMDGPU_ARCH_GEQ
#undef AMDGPU_ARCH_BETWEEN
#undef GET_ATOMIC_SCOPE_AND_ORDER
