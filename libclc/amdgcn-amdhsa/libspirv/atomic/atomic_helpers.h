//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>
#include <spirv/spirv_types.h>

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

#define AMDGPU_ATOMIC_IMPL(FUNC_NAME, TYPE, TYPE_MANGLED, AS, AS_MANGLED,                                                     \
                           NOT_GENERIC, BUILTIN)                                                                              \
  _CLC_DEF TYPE                                                                                                               \
      FUNC_NAME##P##AS_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagENS##NOT_GENERIC##_19MemorySemanticsMask4FlagE##TYPE_MANGLED( \
          volatile AS TYPE *p, enum Scope scope,                                                                              \
          enum MemorySemanticsMask semantics, TYPE val) {                                                                     \
    int atomic_scope = 0, memory_order = 0;                                                                                   \
    GET_ATOMIC_SCOPE_AND_ORDER(scope, atomic_scope, semantics, memory_order)                                                  \
    TYPE ret = BUILTIN(p, val, memory_order, atomic_scope);                                                                   \
    return *(TYPE *)&ret;                                                                                                     \
  }

#define AMDGPU_ATOMIC(FUNC_NAME, TYPE, TYPE_MANGLED, BUILTIN)                  \
  AMDGPU_ATOMIC_IMPL(FUNC_NAME, TYPE, TYPE_MANGLED, global, U3AS1, 1, BUILTIN) \
  AMDGPU_ATOMIC_IMPL(FUNC_NAME, TYPE, TYPE_MANGLED, local, U3AS3, 1, BUILTIN)  \
  AMDGPU_ATOMIC_IMPL(FUNC_NAME, TYPE, TYPE_MANGLED, , , 0, BUILTIN)

