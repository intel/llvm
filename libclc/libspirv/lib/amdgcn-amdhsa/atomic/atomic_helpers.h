//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>
#include <libspirv/spirv_types.h>

extern int __oclc_amdgpu_reflect(__constant char *);

#define AMDGPU_ARCH_GEQ(LOWER) __oclc_ISA_version >= LOWER
#define AMDGPU_ARCH_BETWEEN(LOWER, UPPER)                                      \
  __oclc_ISA_version >= LOWER &&__oclc_ISA_version < UPPER

#define GET_ATOMIC_SCOPE_AND_ORDER(IN_SCOPE, OUT_SCOPE, IN_SEMANTICS,          \
                                   OUT_ORDER)                                  \
  {                                                                            \
    switch (IN_SCOPE) {                                                        \
    case Invocation:                                                           \
    case Subgroup:                                                             \
      OUT_SCOPE = __HIP_MEMORY_SCOPE_WAVEFRONT;                                \
      break;                                                                   \
    case Workgroup:                                                            \
      OUT_SCOPE = __HIP_MEMORY_SCOPE_WORKGROUP;                                \
      break;                                                                   \
    case Device:                                                               \
      OUT_SCOPE = __HIP_MEMORY_SCOPE_AGENT;                                    \
      break;                                                                   \
    case CrossDevice:                                                          \
      OUT_SCOPE = __HIP_MEMORY_SCOPE_SYSTEM;                                   \
      break;                                                                   \
    default:                                                                   \
      __builtin_trap();                                                        \
      __builtin_unreachable();                                                 \
    }                                                                          \
    unsigned order = IN_SEMANTICS & 0x1F;                                      \
    switch (order) {                                                           \
    case None:                                                                 \
      OUT_ORDER = __ATOMIC_RELAXED;                                            \
      break;                                                                   \
    case Acquire:                                                              \
      OUT_ORDER = __ATOMIC_ACQUIRE;                                            \
      break;                                                                   \
    case Release:                                                              \
      OUT_ORDER = __ATOMIC_RELEASE;                                            \
      break;                                                                   \
    case AcquireRelease:                                                       \
      OUT_ORDER = __ATOMIC_ACQ_REL;                                            \
      break;                                                                   \
    case SequentiallyConsistent:                                               \
      OUT_ORDER = __ATOMIC_SEQ_CST;                                            \
      break;                                                                   \
    default:                                                                   \
      __builtin_trap();                                                        \
      __builtin_unreachable();                                                 \
    }                                                                          \
  }

#define AMDGPU_ATOMIC_IMPL(FUNC_NAME, TYPE, AS, BUILTIN)                       \
  _CLC_OVERLOAD _CLC_DECL TYPE FUNC_NAME(AS TYPE *p, int scope, int semantics, \
                                         TYPE val) {                           \
    int atomic_scope = 0, memory_order = 0;                                    \
    GET_ATOMIC_SCOPE_AND_ORDER(scope, atomic_scope, semantics, memory_order)   \
    return BUILTIN(p, val, memory_order, atomic_scope);                        \
  }

#define AMDGPU_ATOMIC(FUNC_NAME, TYPE, BUILTIN)                                \
  AMDGPU_ATOMIC_IMPL(FUNC_NAME, TYPE, global, BUILTIN)                         \
  AMDGPU_ATOMIC_IMPL(FUNC_NAME, TYPE, local, BUILTIN)                          \
  AMDGPU_ATOMIC_IMPL(FUNC_NAME, TYPE, , BUILTIN)

// Safe atomics will either choose a slow CAS atomic impl (default) or a fast
// native atomic if --amdgpu-unsafe-int-atomics is passed to LLVM.
//
// Safe atomics using CAS may be necessary if PCIe does not support atomic
// operations such as and, or, xor
#define AMDGPU_SAFE_ATOMIC_IMPL(FUNC_NAME, TYPE, AS, OP, USE_BUILTIN_COND,     \
                                BUILTIN)                                       \
  _CLC_OVERLOAD _CLC_DEF TYPE FUNC_NAME(AS TYPE *p, int scope, int semantics,  \
                                        TYPE val) {                            \
    int atomic_scope = 0, memory_order = 0;                                    \
    GET_ATOMIC_SCOPE_AND_ORDER(scope, atomic_scope, semantics, memory_order)   \
    if (USE_BUILTIN_COND)                                                      \
      return BUILTIN(p, val, memory_order, atomic_scope);                      \
    /* CAS atomics*/                                                           \
    TYPE oldval = __hip_atomic_load(p, memory_order, atomic_scope);            \
    TYPE newval = 0;                                                           \
    do {                                                                       \
      newval = oldval OP val;                                                  \
    } while (!__hip_atomic_compare_exchange_strong(                            \
        p, &oldval, newval, atomic_scope, atomic_scope, memory_order));        \
    return oldval;                                                             \
  }

#define AMDGPU_SAFE_ATOMIC(FUNC_NAME, TYPE, OP, BUILTIN)                       \
  AMDGPU_SAFE_ATOMIC_IMPL(                                                     \
      FUNC_NAME, TYPE, global, OP,                                             \
      __oclc_amdgpu_reflect("AMDGPU_OCLC_UNSAFE_INT_ATOMICS"), BUILTIN)        \
  AMDGPU_SAFE_ATOMIC_IMPL(FUNC_NAME, TYPE, local, OP,                          \
                          true /* local AS should always use builtin*/,        \
                          BUILTIN)                                             \
  AMDGPU_SAFE_ATOMIC_IMPL(                                                     \
      FUNC_NAME, TYPE, , OP,                                                   \
      __oclc_amdgpu_reflect("AMDGPU_OCLC_UNSAFE_INT_ATOMICS"), BUILTIN)
