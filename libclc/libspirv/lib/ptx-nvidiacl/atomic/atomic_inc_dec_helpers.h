//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_INC_DEC_ATOMIC_HELPERS_H
#define __CLC_INC_DEC_ATOMIC_HELPERS_H

#include <libspirv/spirv.h>
#include <libspirv/spirv_types.h>

#define __CLC_NVVM_ATOMIC_INCDEC_IMPL(                                         \
    TYPE, TYPE_MANGLED, OP_MANGLED, VAL, ADDR_SPACE,                           \
    POINTER_AND_ADDR_SPACE_MANGLED, SUBSTITUTION)                              \
  TYPE _Z21__spirv_\
AtomicIAddEXT##POINTER_AND_ADDR_SPACE_MANGLED##TYPE_MANGLED##N5__spv\
5Scope4FlagENS##SUBSTITUTION##_19MemorySemanticsMask4FlagE##TYPE_MANGLED(      \
      volatile ADDR_SPACE TYPE *, enum Scope, enum MemorySemanticsMask, TYPE); \
  __attribute__((always_inline)) _CLC_DECL TYPE _Z24__spirv_\
Atomic##OP_MANGLED##POINTER_AND_ADDR_SPACE_MANGLED##TYPE_MANGLED##N5__spv\
5Scope4FlagENS##SUBSTITUTION##_19MemorySemanticsMask4FlagE(                    \
      volatile ADDR_SPACE TYPE *pointer, enum Scope scope,                     \
      enum MemorySemanticsMask semantics) {                                    \
    return _Z21__spirv_\
AtomicIAddEXT##POINTER_AND_ADDR_SPACE_MANGLED##TYPE_MANGLED##N5__spv\
5Scope4FlagENS##SUBSTITUTION##_19MemorySemanticsMask4FlagE##TYPE_MANGLED(      \
        pointer, scope, semantics, VAL);                                       \
  }

#define __CLC_NVVM_ATOMIC_INCDEC(TYPE, TYPE_MANGLED, OP_MANGLED, VAL)          \
  __CLC_NVVM_ATOMIC_INCDEC_IMPL(TYPE, TYPE_MANGLED, OP_MANGLED, VAL, __global, \
                                PU3AS1, 1)                                     \
  __CLC_NVVM_ATOMIC_INCDEC_IMPL(TYPE, TYPE_MANGLED, OP_MANGLED, VAL, __local,  \
                                PU3AS3, 1)                                     \
  __CLC_NVVM_ATOMIC_INCDEC_IMPL(TYPE, TYPE_MANGLED, OP_MANGLED, VAL, , P, 0)

#endif
