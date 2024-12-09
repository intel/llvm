//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>
#include <libspirv/spirv_types.h>

#define __CLC_NVVM_ATOMIC_SUB_IMPL(TYPE, TYPE_MANGLED, OP_MANGLED, ADDR_SPACE, \
                                   POINTER_AND_ADDR_SPACE_MANGLED,             \
                                   SUBSTITUTION)                               \
  TYPE _Z18__spirv_\
AtomicIAdd##POINTER_AND_ADDR_SPACE_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagEN\
S##SUBSTITUTION##_19MemorySemanticsMask4FlagE##TYPE_MANGLED(                   \
      volatile ADDR_SPACE TYPE *, enum Scope, enum MemorySemanticsMask, TYPE); \
  __attribute__((always_inline)) _CLC_DECL TYPE _Z18__spirv_\
Atomic##OP_MANGLED##POINTER_AND_ADDR_SPACE_MANGLED##TYPE_MANGLED##N5__spv5Scope\
4FlagENS##SUBSTITUTION##_19MemorySemanticsMask4FlagE##TYPE_MANGLED(            \
      volatile ADDR_SPACE TYPE *pointer, enum Scope scope,                     \
      enum MemorySemanticsMask semantics, TYPE val) {                          \
    return _Z18__spirv_\
AtomicIAdd##POINTER_AND_ADDR_SPACE_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagEN\
S##SUBSTITUTION##_19MemorySemanticsMask4FlagE##TYPE_MANGLED(pointer, scope,    \
                                                            semantics, -val);  \
  }

#define __CLC_NVVM_ATOMIC_SUB(TYPE, TYPE_MANGLED, OP_MANGLED)                  \
  __CLC_NVVM_ATOMIC_SUB_IMPL(TYPE, TYPE_MANGLED, OP_MANGLED, __global, PU3AS1, \
                             1)                                                \
  __CLC_NVVM_ATOMIC_SUB_IMPL(TYPE, TYPE_MANGLED, OP_MANGLED, __local, PU3AS3,  \
                             1)                                                \
  __CLC_NVVM_ATOMIC_SUB_IMPL(TYPE, TYPE_MANGLED, OP_MANGLED, , P, 0)

__CLC_NVVM_ATOMIC_SUB(int, i, ISub)
__CLC_NVVM_ATOMIC_SUB(unsigned int, j, ISub)
__CLC_NVVM_ATOMIC_SUB(long, l, ISub)
__CLC_NVVM_ATOMIC_SUB(unsigned long, m, ISub)

#undef __CLC_NVVM_ATOMIC_SUB_IMPL
#undef __CLC_NVVM_ATOMIC_SUB
