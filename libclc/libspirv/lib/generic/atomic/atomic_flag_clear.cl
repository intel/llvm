//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#define __CLC_DEFINE_ATOMIC_FLAG_CLEAR(ADDRSPACE)                              \
  _CLC_OVERLOAD _CLC_DEF void __spirv_AtomicFlagClear(                         \
      ADDRSPACE int *Pointer, int Scope, int Semantics) {                      \
    __spirv_AtomicStore(Pointer, Scope, Semantics, 0);                         \
  }

__CLC_DEFINE_ATOMIC_FLAG_CLEAR(global)
__CLC_DEFINE_ATOMIC_FLAG_CLEAR(local)
__CLC_DEFINE_ATOMIC_FLAG_CLEAR(private)
#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
__CLC_DEFINE_ATOMIC_FLAG_CLEAR()
#endif
