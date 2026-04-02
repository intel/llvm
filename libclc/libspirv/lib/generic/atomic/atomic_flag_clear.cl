//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/atomic/clc_atomic_flag_clear.h>
#include <libspirv/atomic/atomic_helper.h>
#include <libspirv/spirv.h>

#define __CLC_DEFINE_ATOMIC_FLAG_CLEAR(ADDRSPACE)                              \
  _CLC_OVERLOAD _CLC_DEF void __spirv_AtomicFlagClear(                         \
      ADDRSPACE int *Ptr, int Scope, int Semantics) {                          \
    __clc_atomic_flag_clear(Ptr, __spirv_get_clang_memory_order(Semantics),    \
                            __spirv_get_clang_memory_scope(Scope));            \
  }

__CLC_DEFINE_ATOMIC_FLAG_CLEAR(global)
__CLC_DEFINE_ATOMIC_FLAG_CLEAR(local)
#if _CLC_GENERIC_AS_SUPPORTED
__CLC_DEFINE_ATOMIC_FLAG_CLEAR()
#endif
