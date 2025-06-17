//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "atomic_helpers.h"
#include <libspirv/spirv.h>
#include <libspirv/spirv_types.h>

#define FUNC_BODY(OP, TYPE, STORAGE_TYPE, AS)                                  \
  {                                                                            \
    int atomic_scope = 0, memory_order = 0;                                    \
    AS STORAGE_TYPE *int_pointer = (AS STORAGE_TYPE *)p;                       \
    STORAGE_TYPE old_int_val = 0, new_int_val = 0;                             \
    TYPE old_val = 0;                                                          \
    do {                                                                       \
      old_int_val = __spirv_AtomicLoad(int_pointer, scope, semantics);         \
      old_val = *(TYPE *)&old_int_val;                                         \
      if (old_val OP val)                                                      \
        return old_val;                                                        \
      new_int_val = *(STORAGE_TYPE *)&val;                                     \
    } while (__spirv_AtomicCompareExchange(int_pointer, scope, semantics,      \
                                           semantics, new_int_val,             \
                                           old_int_val) != old_int_val);       \
                                                                               \
    return old_val;                                                            \
  }

#define AMDGPU_ATOMIC_FP_MINMAX_IMPL(OPNAME, OP, TYPE, STORAGE_TYPE, AS)       \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_AtomicF##OPNAME##EXT(                    \
      AS TYPE *p, int scope, int semantics, TYPE val) {                        \
    FUNC_BODY(OP, TYPE, STORAGE_TYPE, AS)                                      \
  }

#define AMDGPU_ATOMIC_FP_MINMAX_IMPL_CHECK(OPNAME, OP, TYPE, STORAGE_TYPE, AS, \
                                           CHECK, NEW_BUILTIN)                 \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_AtomicF##OPNAME##EXT(                    \
      AS TYPE *p, int scope, int semantics, TYPE val) {                        \
    if (CHECK)                                                                 \
      return NEW_BUILTIN(p, val);                                              \
    FUNC_BODY(OP, TYPE, STORAGE_TYPE, AS)                                      \
  }
