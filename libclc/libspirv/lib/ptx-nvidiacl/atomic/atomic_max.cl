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

__CLC_NVVM_ATOMIC(int, int, i, max, __spirv_AtomicSMax)
__CLC_NVVM_ATOMIC(long, long, l, max, __spirv_AtomicSMax)
__CLC_NVVM_ATOMIC(unsigned int, unsigned int, ui, max, __spirv_AtomicUMax)
__CLC_NVVM_ATOMIC(unsigned long, unsigned long, ul, max, __spirv_AtomicUMax)

#undef __CLC_NVVM_ATOMIC_TYPES
#undef __CLC_NVVM_ATOMIC
#undef __CLC_NVVM_ATOMIC_IMPL

#define __CLC_NVVM_ATOMIC_MAX_IMPL(TYPE, TYPE_INT, OP_MANGLED, ADDR_SPACE)     \
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
    while (val > old) {                                                        \
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

#define __CLC_NVVM_ATOMIC_MAX(TYPE, TYPE_INT, OP_MANGLED)                      \
  __CLC_NVVM_ATOMIC_MAX_IMPL(TYPE, TYPE_INT, OP_MANGLED, __global)             \
  __CLC_NVVM_ATOMIC_MAX_IMPL(TYPE, TYPE_INT, OP_MANGLED, __local)              \
  __CLC_NVVM_ATOMIC_MAX_IMPL(TYPE, TYPE_INT, OP_MANGLED, )

__CLC_NVVM_ATOMIC_MAX(float, int, FMaxEXT)
__CLC_NVVM_ATOMIC_MAX(double, long, FMaxEXT)
