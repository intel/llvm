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

#define AMDGPU_ATOMIC_SUB_IMPL(FUNC_NAME, TYPE, AS, BUILTIN)                   \
  _CLC_OVERLOAD _CLC_DEF TYPE FUNC_NAME(AS TYPE *p, int scope, int semantics,  \
                                        TYPE val) {                            \
    int atomic_scope = 0, memory_order = 0;                                    \
    GET_ATOMIC_SCOPE_AND_ORDER(scope, atomic_scope, semantics, memory_order)   \
    return BUILTIN(p, -val, memory_order);                                     \
  }

#define AMDGPU_ATOMIC_SUB(FUNC_NAME, TYPE, BUILTIN)                            \
  AMDGPU_ATOMIC_SUB_IMPL(FUNC_NAME, TYPE, global, BUILTIN)                     \
  AMDGPU_ATOMIC_SUB_IMPL(FUNC_NAME, TYPE, local, BUILTIN)                      \
  AMDGPU_ATOMIC_SUB_IMPL(FUNC_NAME, TYPE, , BUILTIN)

AMDGPU_ATOMIC_SUB(__spirv_AtomicISub, int, __atomic_fetch_add)
AMDGPU_ATOMIC_SUB(__spirv_AtomicISub, unsigned int, __atomic_fetch_add)
AMDGPU_ATOMIC_SUB(__spirv_AtomicISub, long, __atomic_fetch_add)
AMDGPU_ATOMIC_SUB(__spirv_AtomicISub, unsigned long, __atomic_fetch_add)

#undef AMDGPU_ATOMIC
#undef AMDGPU_ATOMIC_IMPL
#undef AMDGPU_ATOMIC_SUB
#undef AMDGPU_ATOMIC_SUB_IMPL
#undef AMDGPU_ARCH_GEQ
#undef AMDGPU_ARCH_BETWEEN
#undef GET_ATOMIC_SCOPE_AND_ORDER
