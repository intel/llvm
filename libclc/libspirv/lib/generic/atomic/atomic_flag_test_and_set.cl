//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#define __CLC_DEFINE_ATOMIC_FLAG_TEST_AND_SET(ADDRSPACE)                       \
  _CLC_OVERLOAD _CLC_DEF bool __spirv_AtomicFlagTestAndSet(                    \
      ADDRSPACE int *Pointer, int Scope, int Semantics) {                      \
    return (bool)__spirv_AtomicExchange(Pointer, Scope, Semantics, 1);         \
  }

__CLC_DEFINE_ATOMIC_FLAG_TEST_AND_SET(global)
__CLC_DEFINE_ATOMIC_FLAG_TEST_AND_SET(local)
__CLC_DEFINE_ATOMIC_FLAG_TEST_AND_SET(private)
#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
__CLC_DEFINE_ATOMIC_FLAG_TEST_AND_SET()
#endif
