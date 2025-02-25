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

#define AMDGPU_ATOMIC_FP_MINMAX_IMPL(                                                                                                           \
    OPNAME, OP, TYPE, TYPE_MANGLED, STORAGE_TYPE, STORAGE_TYPE_MANGLED, AS,                                                                     \
    AS_MANGLED, SUB1, SUB2, CHECK, NEW_BUILTIN)                                                                                                 \
  _CLC_DEF STORAGE_TYPE                                                                                                                         \
      _Z29__spirv_AtomicCompareExchangeP##AS_MANGLED##STORAGE_TYPE_MANGLED##N5__spv5Scope4FlagENS##SUB1##_19MemorySemanticsMask4FlagES##SUB2(   \
          volatile AS STORAGE_TYPE *, enum Scope, enum MemorySemanticsMask,                                                                     \
          enum MemorySemanticsMask, STORAGE_TYPE desired,                                                                                       \
          STORAGE_TYPE expected);                                                                                                               \
  _CLC_DEF STORAGE_TYPE                                                                                                                         \
      _Z18__spirv_AtomicLoadP##AS_MANGLED##K##STORAGE_TYPE_MANGLED##N5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE(                         \
          const volatile AS STORAGE_TYPE *, enum Scope,                                                                                         \
          enum MemorySemanticsMask);                                                                                                            \
  _CLC_DEF TYPE                                                                                                                                 \
      _Z21__spirv_AtomicF##OPNAME##EXTP##AS_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagENS##SUB1##_19MemorySemanticsMask4FlagE##TYPE_MANGLED(     \
          volatile AS TYPE *p, enum Scope scope,                                                                                                \
          enum MemorySemanticsMask semantics, TYPE val) {                                                                                       \
    if (CHECK)                                                                                                                                  \
      return NEW_BUILTIN(p, val);                                                                                                               \
    int atomic_scope = 0, memory_order = 0;                                                                                                     \
    volatile AS STORAGE_TYPE *int_pointer = (volatile AS STORAGE_TYPE *)p;                                                                      \
    STORAGE_TYPE old_int_val = 0, new_int_val = 0;                                                                                              \
    TYPE old_val = 0;                                                                                                                           \
    do {                                                                                                                                        \
      old_int_val =                                                                                                                             \
          _Z18__spirv_AtomicLoadP##AS_MANGLED##K##STORAGE_TYPE_MANGLED##N5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE(                     \
              int_pointer, scope, semantics);                                                                                                   \
      old_val = *(TYPE *)&old_int_val;                                                                                                          \
      if (old_val OP val)                                                                                                                       \
        return old_val;                                                                                                                         \
      new_int_val = *(STORAGE_TYPE *)&val;                                                                                                      \
    } while (                                                                                                                                   \
        _Z29__spirv_AtomicCompareExchangeP##AS_MANGLED##STORAGE_TYPE_MANGLED##N5__spv5Scope4FlagENS##SUB1##_19MemorySemanticsMask4FlagES##SUB2( \
            int_pointer, scope, semantics, semantics, new_int_val,                                                                              \
            old_int_val) != old_int_val);                                                                                                       \
                                                                                                                                                \
    return old_val;                                                                                                                             \
  }
