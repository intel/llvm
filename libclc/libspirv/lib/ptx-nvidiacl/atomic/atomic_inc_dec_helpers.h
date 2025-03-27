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

#define __CLC_NVVM_ATOMIC_INCDEC_IMPL(TYPE, OP_MANGLED, VAL, ADDR_SPACE)       \
  __attribute__((always_inline)) _CLC_OVERLOAD _CLC_DECL TYPE                  \
      __spirv_Atomic##OP_MANGLED(ADDR_SPACE TYPE *pointer, int scope,          \
                                 int semantics) {                              \
    return __spirv_AtomicIAdd(pointer, scope, semantics, VAL);                 \
  }

#define __CLC_NVVM_ATOMIC_INCDEC(TYPE, OP_MANGLED, VAL)                        \
  __CLC_NVVM_ATOMIC_INCDEC_IMPL(TYPE, OP_MANGLED, VAL, __global)               \
  __CLC_NVVM_ATOMIC_INCDEC_IMPL(TYPE, OP_MANGLED, VAL, __local)                \
  __CLC_NVVM_ATOMIC_INCDEC_IMPL(TYPE, OP_MANGLED, VAL, )

#endif
