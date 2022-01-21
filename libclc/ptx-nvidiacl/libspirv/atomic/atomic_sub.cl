//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>
#include <spirv/spirv_types.h>

#define __CLC_NVVM_ATOMIC_SUB_IMPL(TYPE, TYPE_MANGLED, OP_MANGLED, ADDR_SPACE,                                                                 \
                                   ADDR_SPACE_MANGLED)                                                                                         \
  TYPE                                                                                                                                         \
      _Z18__spirv_AtomicIAddPU3##ADDR_SPACE_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE##TYPE_MANGLED(           \
          volatile ADDR_SPACE TYPE *, enum Scope, enum MemorySemanticsMask,                                                                    \
          TYPE);                                                                                                                               \
  _CLC_DECL TYPE                                                                                                                               \
      _Z18__spirv_Atomic##OP_MANGLED##PU3##ADDR_SPACE_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE##TYPE_MANGLED( \
          volatile ADDR_SPACE TYPE *pointer, enum Scope scope,                                                                                 \
          enum MemorySemanticsMask semantics, TYPE val) {                                                                                      \
    return _Z18__spirv_AtomicIAddPU3##ADDR_SPACE_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE##TYPE_MANGLED(      \
        pointer, scope, semantics, -val);                                                                                                      \
  }

#define __CLC_NVVM_ATOMIC_SUB(TYPE, TYPE_MANGLED, OP_MANGLED)                  \
  __CLC_NVVM_ATOMIC_SUB_IMPL(TYPE, TYPE_MANGLED, OP_MANGLED, __global, AS1)    \
  __CLC_NVVM_ATOMIC_SUB_IMPL(TYPE, TYPE_MANGLED, OP_MANGLED, __local, AS3)

__CLC_NVVM_ATOMIC_SUB(int, i, ISub)
__CLC_NVVM_ATOMIC_SUB(unsigned int, j, ISub)
__CLC_NVVM_ATOMIC_SUB(long, l, ISub)
__CLC_NVVM_ATOMIC_SUB(unsigned long, m, ISub)

#undef __CLC_NVVM_ATOMIC_SUB_IMPL
#undef __CLC_NVVM_ATOMIC_SUB
