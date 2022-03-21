//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_INC_DEC_ATOMIC_HELPERS_H
#define __CLC_INC_DEC_ATOMIC_HELPERS_H

#include <spirv/spirv.h>
#include <spirv/spirv_types.h>

#define __CLC_NVVM_ATOMIC_INCDEC_IMPL(TYPE, TYPE_MANGLED, OP_MANGLED, VAL,                                                                                \
                                      ADDR_SPACE, ADDR_SPACE_MANGLED,                                                                                     \
                                      SUBSTITUTION)                                                                                                       \
  TYPE                                                                                                                                                    \
      _Z21__spirv_AtomicIAddEXTP##ADDR_SPACE_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagENS##SUBSTITUTION##_19MemorySemanticsMask4FlagE##TYPE_MANGLED(      \
          volatile ADDR_SPACE TYPE *, enum Scope, enum MemorySemanticsMask,                                                                               \
          TYPE);                                                                                                                                          \
  __attribute__((always_inline)) _CLC_DECL TYPE                                                                                                           \
      _Z24__spirv_Atomic##OP_MANGLED##P##ADDR_SPACE_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagENS##SUBSTITUTION##_19MemorySemanticsMask4FlagE(             \
          volatile ADDR_SPACE TYPE *pointer, enum Scope scope,                                                                                            \
          enum MemorySemanticsMask semantics) {                                                                                                           \
    return _Z21__spirv_AtomicIAddEXTP##ADDR_SPACE_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagENS##SUBSTITUTION##_19MemorySemanticsMask4FlagE##TYPE_MANGLED( \
        pointer, scope, semantics, VAL);                                                                                                                  \
  }

#define __CLC_NVVM_ATOMIC_INCDEC(TYPE, TYPE_MANGLED, OP_MANGLED, VAL)          \
  __CLC_NVVM_ATOMIC_INCDEC_IMPL(TYPE, TYPE_MANGLED, OP_MANGLED, VAL, , , 0)    \
  __CLC_NVVM_ATOMIC_INCDEC_IMPL(TYPE, TYPE_MANGLED, OP_MANGLED, VAL, __global, \
                                U3AS1, 1)                                      \
  __CLC_NVVM_ATOMIC_INCDEC_IMPL(TYPE, TYPE_MANGLED, OP_MANGLED, VAL, __local,  \
                                U3AS3, 1)

#endif
