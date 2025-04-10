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

__CLC_NVVM_ATOMIC(int, int, i, min, __spirv_AtomicSMin)
__CLC_NVVM_ATOMIC(long, long, l, min, __spirv_AtomicSMin)
__CLC_NVVM_ATOMIC(uint, uint, ui, min, __spirv_AtomicUMin)
__CLC_NVVM_ATOMIC(ulong, ulong, ul, min, __spirv_AtomicUMin)

#undef __CLC_NVVM_ATOMIC_TYPES
#undef __CLC_NVVM_ATOMIC
#undef __CLC_NVVM_ATOMIC_IMPL

#define __CLC_NVVM_ATOMIC_MIN_IMPL(TYPE, TYPE_INT, OP_MANGLED, ADDR_SPACE)     \
  __attribute__((always_inline)) _CLC_OVERLOAD _CLC_DECL TYPE                  \
      __spirv_Atomic##OP_MANGLED(ADDR_SPACE TYPE *pointer, int scope,          \
                                 int semantics, TYPE val) {                    \
    int load_order;                                                            \
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
    ADDR_SPACE TYPE_INT *pointer_int = (ADDR_SPACE TYPE_INT *)pointer;         \
    TYPE_INT val_int = *(TYPE_INT *)&val;                                      \
    TYPE_INT old_int = __spirv_AtomicLoad(pointer_int, scope, load_order);     \
    TYPE old = *(TYPE *)&old_int;                                              \
    while (val < old) {                                                        \
      TYPE_INT tmp_int = __spirv_AtomicCompareExchange(                        \
          pointer_int, scope, semantics, semantics, val_int, old_int);         \
      if (old_int == tmp_int) {                                                \
        return *(TYPE *)&tmp_int;                                              \
      }                                                                        \
      old_int = tmp_int;                                                       \
      old = *(TYPE *)&old_int;                                                 \
    }                                                                          \
    return old;                                                                \
  }

#define __CLC_NVVM_ATOMIC_MIN(TYPE, TYPE_INT, OP_MANGLED)                      \
  __CLC_NVVM_ATOMIC_MIN_IMPL(TYPE, TYPE_INT, OP_MANGLED, __global)             \
  __CLC_NVVM_ATOMIC_MIN_IMPL(TYPE, TYPE_INT, OP_MANGLED, __local)              \
  __CLC_NVVM_ATOMIC_MIN_IMPL(TYPE, TYPE_INT, OP_MANGLED, )

__CLC_NVVM_ATOMIC_MIN(float, int, FMinEXT)
__CLC_NVVM_ATOMIC_MIN(double, long, FMinEXT)
