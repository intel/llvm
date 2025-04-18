//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <atomic_helpers.h>
#include <libspirv/spirv.h>
#include <libspirv/spirv_types.h>

__CLC_NVVM_ATOMIC(int, i, int, i, max, _Z18__spirv_AtomicSMax)
__CLC_NVVM_ATOMIC(long, l, long, l, max, _Z18__spirv_AtomicSMax)
__CLC_NVVM_ATOMIC(unsigned int, j, unsigned int, ui, max,
                  _Z18__spirv_AtomicUMax)
__CLC_NVVM_ATOMIC(unsigned long, m, unsigned long, ul, max,
                  _Z18__spirv_AtomicUMax)

#undef __CLC_NVVM_ATOMIC_TYPES
#undef __CLC_NVVM_ATOMIC
#undef __CLC_NVVM_ATOMIC_IMPL

#define __CLC_NVVM_ATOMIC_MAX_IMPL(                                            \
    TYPE, TYPE_MANGLED, TYPE_INT, TYPE_INT_MANGLED, OP_MANGLED, ADDR_SPACE,    \
    POINTER_AND_ADDR_SPACE_MANGLED, SUBSTITUTION1, SUBSTITUTION2)              \
  TYPE_INT                                                                     \
  _Z18__spirv_\
AtomicLoad##POINTER_AND_ADDR_SPACE_MANGLED##K##TYPE_INT_MANGLED##N5__spv5Scope4\
FlagENS1_19MemorySemanticsMask4FlagE(volatile ADDR_SPACE const TYPE_INT *,     \
                                     enum Scope, enum MemorySemanticsMask);    \
  TYPE_INT                                                                     \
  _Z29__spirv_\
AtomicCompareExchange##POINTER_AND_ADDR_SPACE_MANGLED##TYPE_INT_MANGLED##N5__sp\
v5Scope4FlagENS##SUBSTITUTION1##_19MemorySemanticsMask\
4FlagES##SUBSTITUTION2##_##TYPE_INT_MANGLED##TYPE_INT_MANGLED(                 \
      volatile ADDR_SPACE TYPE_INT *, enum Scope, enum MemorySemanticsMask,    \
      enum MemorySemanticsMask, TYPE_INT, TYPE_INT);                           \
  __attribute__((always_inline)) _CLC_DECL TYPE _Z21__spirv_\
Atomic##OP_MANGLED##POINTER_AND_ADDR_SPACE_MANGLED##TYPE_MANGLED##N5__spv5Scope\
4FlagENS##SUBSTITUTION1##_19MemorySemanticsMask4FlagE##TYPE_MANGLED(           \
      volatile ADDR_SPACE TYPE *pointer, enum Scope scope,                     \
      enum MemorySemanticsMask semantics, TYPE val) {                          \
    enum MemorySemanticsMask load_order;                                       \
    switch (semantics) {                                                       \
    case SequentiallyConsistent:                                               \
      load_order = SequentiallyConsistent;                                     \
      break;                                                                   \
    case Acquire:                                                              \
    case AcquireRelease:                                                       \
      load_order = Acquire;                                                    \
      break;                                                                   \
    default:                                                                   \
      load_order = None;                                                       \
    }                                                                          \
    volatile ADDR_SPACE TYPE_INT *pointer_int =                                \
        (volatile ADDR_SPACE TYPE_INT *)pointer;                               \
    TYPE_INT val_int = *(TYPE_INT *)&val;                                      \
    TYPE_INT old_int = _Z18__spirv_\
AtomicLoad##POINTER_AND_ADDR_SPACE_MANGLED##K##TYPE_INT_MANGLED##N5__spv5Scope4\
FlagENS1_19MemorySemanticsMask4FlagE(pointer_int, scope, load_order);          \
    TYPE old = *(TYPE *)&old_int;                                              \
    while (val > old) {                                                        \
      TYPE_INT tmp_int = _Z29__spirv_\
AtomicCompareExchange##POINTER_AND_ADDR_SPACE_MANGLED##TYPE_INT_MANGLED##N5__sp\
v5Scope4FlagENS##SUBSTITUTION1##_19MemorySemanticsMask\
4FlagES##SUBSTITUTION2##_##TYPE_INT_MANGLED##TYPE_INT_MANGLED(                 \
          pointer_int, scope, semantics, semantics, val_int, old_int);         \
      if (old_int == tmp_int) {                                                \
        return *(TYPE *)&tmp_int;                                              \
      }                                                                        \
      old_int = tmp_int;                                                       \
      old = *(TYPE *)&old_int;                                                 \
    }                                                                          \
    return old;                                                                \
  }

#define __CLC_NVVM_ATOMIC_MAX(TYPE, TYPE_MANGLED, TYPE_INT, TYPE_INT_MANGLED,  \
                              OP_MANGLED)                                      \
  __CLC_NVVM_ATOMIC_MAX_IMPL(TYPE, TYPE_MANGLED, TYPE_INT, TYPE_INT_MANGLED,   \
                             OP_MANGLED, __global, PU3AS1, 1, 5)               \
  __CLC_NVVM_ATOMIC_MAX_IMPL(TYPE, TYPE_MANGLED, TYPE_INT, TYPE_INT_MANGLED,   \
                             OP_MANGLED, __local, PU3AS3, 1, 5)                \
  __CLC_NVVM_ATOMIC_MAX_IMPL(TYPE, TYPE_MANGLED, TYPE_INT, TYPE_INT_MANGLED,   \
                             OP_MANGLED, , P, 0, 4)

__CLC_NVVM_ATOMIC_MAX(float, f, int, i, FMaxEXT)
__CLC_NVVM_ATOMIC_MAX(double, d, long, l, FMaxEXT)
