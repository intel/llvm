//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <atomic_helpers.h>
#include <spirv/spirv.h>
#include <spirv/spirv_types.h>

__CLC_NVVM_ATOMIC(int, i, int, i, min, _Z18__spirv_AtomicSMin)
__CLC_NVVM_ATOMIC(long, l, long, l, min, _Z18__spirv_AtomicSMin)
__CLC_NVVM_ATOMIC(uint, j, uint, ui, min, _Z18__spirv_AtomicUMin)
__CLC_NVVM_ATOMIC(ulong, m, ulong, ul, min, _Z18__spirv_AtomicUMin)

#undef __CLC_NVVM_ATOMIC_TYPES
#undef __CLC_NVVM_ATOMIC
#undef __CLC_NVVM_ATOMIC_IMPL

#define __CLC_NVVM_ATOMIC_MIN_IMPL(TYPE, TYPE_MANGLED, TYPE_INT,                                                                                                                   \
                                   TYPE_INT_MANGLED, OP_MANGLED, ADDR_SPACE,                                                                                                       \
                                   ADDR_SPACE_MANGLED)                                                                                                                             \
  TYPE_INT                                                                                                                                                                         \
      _Z18__spirv_AtomicLoadPU3##ADDR_SPACE_MANGLED##K##TYPE_INT_MANGLED##N5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE(                                                      \
          volatile ADDR_SPACE const TYPE_INT *, enum Scope,                                                                                                                        \
          enum MemorySemanticsMask);                                                                                                                                               \
  TYPE_INT                                                                                                                                                                         \
  _Z29__spirv_AtomicCompareExchange##PU3##ADDR_SPACE_MANGLED##TYPE_INT_MANGLED##N5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagES5_##TYPE_INT_MANGLED##TYPE_INT_MANGLED(         \
      volatile ADDR_SPACE TYPE_INT *, enum Scope, enum MemorySemanticsMask,                                                                                                        \
      enum MemorySemanticsMask, TYPE_INT, TYPE_INT);                                                                                                                               \
  _CLC_DECL TYPE                                                                                                                                                                   \
      _Z21__spirv_Atomic##OP_MANGLED##PU3##ADDR_SPACE_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE##TYPE_MANGLED(                                     \
          volatile ADDR_SPACE TYPE *pointer, enum Scope scope,                                                                                                                     \
          enum MemorySemanticsMask semantics, TYPE val) {                                                                                                                          \
    enum MemorySemanticsMask load_order;                                                                                                                                           \
    switch (semantics) {                                                                                                                                                           \
    case SequentiallyConsistent:                                                                                                                                                   \
      load_order = SequentiallyConsistent;                                                                                                                                         \
      break;                                                                                                                                                                       \
    case Acquire:                                                                                                                                                                  \
    case AcquireRelease:                                                                                                                                                           \
      load_order = Acquire;                                                                                                                                                        \
      break;                                                                                                                                                                       \
    default:                                                                                                                                                                       \
      load_order = None;                                                                                                                                                           \
    }                                                                                                                                                                              \
    volatile ADDR_SPACE TYPE_INT *pointer_int =                                                                                                                                    \
        (volatile ADDR_SPACE TYPE_INT *)pointer;                                                                                                                                   \
    TYPE_INT val_int = *(TYPE_INT *)&val;                                                                                                                                          \
    TYPE_INT old_int =                                                                                                                                                             \
        _Z18__spirv_AtomicLoadPU3##ADDR_SPACE_MANGLED##K##TYPE_INT_MANGLED##N5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE(                                                    \
            pointer_int, scope, load_order);                                                                                                                                       \
    TYPE old = *(TYPE *)&old_int;                                                                                                                                                  \
    while (val < old) {                                                                                                                                                            \
      TYPE_INT tmp_int =                                                                                                                                                           \
          _Z29__spirv_AtomicCompareExchange##PU3##ADDR_SPACE_MANGLED##TYPE_INT_MANGLED##N5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagES5_##TYPE_INT_MANGLED##TYPE_INT_MANGLED( \
              pointer_int, scope, semantics, semantics, val_int, old_int);                                                                                                         \
      if (old_int == tmp_int) {                                                                                                                                                    \
        return *(TYPE *)&tmp_int;                                                                                                                                                  \
      }                                                                                                                                                                            \
      old_int = tmp_int;                                                                                                                                                           \
      old = *(TYPE *)&old_int;                                                                                                                                                     \
    }                                                                                                                                                                              \
    return old;                                                                                                                                                                    \
  }

#define __CLC_NVVM_ATOMIC_MIN(TYPE, TYPE_MANGLED, TYPE_INT, TYPE_INT_MANGLED,  \
                              OP_MANGLED)                                      \
  __CLC_NVVM_ATOMIC_MIN_IMPL(TYPE, TYPE_MANGLED, TYPE_INT, TYPE_INT_MANGLED,   \
                             OP_MANGLED, __global, AS1)                        \
  __CLC_NVVM_ATOMIC_MIN_IMPL(TYPE, TYPE_MANGLED, TYPE_INT, TYPE_INT_MANGLED,   \
                             OP_MANGLED, __local, AS3)

__CLC_NVVM_ATOMIC_MIN(float, f, int, i, FMinEXT)
__CLC_NVVM_ATOMIC_MIN(double, d, long, l, FMinEXT)