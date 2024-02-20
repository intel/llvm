//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>
#include <spirv/spirv_types.h>

#define AMDGPU_ARCH_GEQ(LOWER) __oclc_ISA_version >= LOWER
#define AMDGPU_ARCH_BETWEEN(LOWER, UPPER)                                      \
  __oclc_ISA_version >= LOWER &&__oclc_ISA_version < UPPER

#define GET_ATOMIC_SCOPE_AND_ORDER(IN_SCOPE, OUT_SCOPE, IN_SEMANTICS,          \
                                   OUT_ORDER)                                  \
  {                                                                            \
    switch (IN_SCOPE) {                                                        \
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

#define AMDGPU_ATOMIC_IMPL(FUNC_NAME, TYPE, TYPE_MANGLED, AS, AS_MANGLED,                                              \
                           SUB1, BUILTIN)                                                                              \
  _CLC_DEF TYPE                                                                                                        \
      FUNC_NAME##P##AS_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagENS##SUB1##_19MemorySemanticsMask4FlagE##TYPE_MANGLED( \
          volatile AS TYPE *p, enum Scope scope,                                                                       \
          enum MemorySemanticsMask semantics, TYPE val) {                                                              \
    int atomic_scope = 0, memory_order = 0;                                                                            \
    GET_ATOMIC_SCOPE_AND_ORDER(scope, atomic_scope, semantics, memory_order)                                           \
    return BUILTIN(p, val, memory_order, atomic_scope);                                                                \
  }

#define AMDGPU_ATOMIC(FUNC_NAME, TYPE, TYPE_MANGLED, BUILTIN)                  \
  AMDGPU_ATOMIC_IMPL(FUNC_NAME, TYPE, TYPE_MANGLED, global, U3AS1, 1, BUILTIN) \
  AMDGPU_ATOMIC_IMPL(FUNC_NAME, TYPE, TYPE_MANGLED, local, U3AS3, 1, BUILTIN)  \
  AMDGPU_ATOMIC_IMPL(FUNC_NAME, TYPE, TYPE_MANGLED, , , 0, BUILTIN)

#define AMDGPU_CAS_ATOMIC_IMPL(FUNC_NAME, TYPE, TYPE_MANGLED, AS, AS_MANGLED,                                          \
                               SUB1, OP)                                                                               \
  _CLC_DEF TYPE                                                                                                        \
      FUNC_NAME##P##AS_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagENS##SUB1##_19MemorySemanticsMask4FlagE##TYPE_MANGLED( \
          volatile AS TYPE *p, enum Scope scope,                                                                       \
          enum MemorySemanticsMask semantics, TYPE val) {                                                              \
    int atomic_scope = 0, memory_order = 0;                                                                            \
    GET_ATOMIC_SCOPE_AND_ORDER(scope, atomic_scope, semantics, memory_order)                                           \
    TYPE oldval = __hip_atomic_load(p, memory_order, atomic_scope);                                                    \
    TYPE newval = 0;                                                                                                   \
    do {                                                                                                               \
      newval = oldval OP val;                                                                                          \
    } while (!__hip_atomic_compare_exchange_strong(                                                                    \
        p, &oldval, newval, atomic_scope, atomic_scope, memory_order));                                                \
    return oldval;                                                                                                     \
  }

#define AMDGPU_CAS_ATOMIC(FUNC_NAME, TYPE, TYPE_MANGLED, OP)                   \
  AMDGPU_CAS_ATOMIC_IMPL(FUNC_NAME, TYPE, TYPE_MANGLED, global, U3AS1, 1, OP)  \
  AMDGPU_CAS_ATOMIC_IMPL(FUNC_NAME, TYPE, TYPE_MANGLED, local, U3AS3, 1, OP)   \
  AMDGPU_CAS_ATOMIC_IMPL(FUNC_NAME, TYPE, TYPE_MANGLED, , , 0, OP)
