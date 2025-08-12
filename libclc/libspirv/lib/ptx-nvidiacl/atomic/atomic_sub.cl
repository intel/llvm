//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>
#include <libspirv/spirv_types.h>

#define __CLC_NVVM_ATOMIC_SUB_IMPL(TYPE, OP_MANGLED, ADDR_SPACE)               \
  _CLC_OVERLOAD _CLC_DECL TYPE __spirv_AtomicIAdd(ADDR_SPACE TYPE *, int, int, \
                                                  TYPE);                       \
  __attribute__((always_inline)) _CLC_OVERLOAD _CLC_DECL TYPE                  \
      __spirv_Atomic##OP_MANGLED(ADDR_SPACE TYPE *pointer, int scope,          \
                                 int semantics, TYPE val) {                    \
    return __spirv_AtomicIAdd(pointer, scope, semantics, -val);                \
  }

#define __CLC_NVVM_ATOMIC_SUB(TYPE, OP_MANGLED)                                \
  __CLC_NVVM_ATOMIC_SUB_IMPL(TYPE, OP_MANGLED, __global)                       \
  __CLC_NVVM_ATOMIC_SUB_IMPL(TYPE, OP_MANGLED, __local)                        \
  __CLC_NVVM_ATOMIC_SUB_IMPL(TYPE, OP_MANGLED, )

__CLC_NVVM_ATOMIC_SUB(int, ISub)
__CLC_NVVM_ATOMIC_SUB(unsigned int, ISub)
__CLC_NVVM_ATOMIC_SUB(long, ISub)
__CLC_NVVM_ATOMIC_SUB(unsigned long, ISub)

#undef __CLC_NVVM_ATOMIC_SUB_IMPL
#undef __CLC_NVVM_ATOMIC_SUB
