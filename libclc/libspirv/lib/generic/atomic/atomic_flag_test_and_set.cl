//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/atomic/clc_atomic_flag_test_and_set.h>
#include <libspirv/atomic/atomic_helper.h>
#include <libspirv/spirv.h>

#define __CLC_DEFINE_ATOMIC_FLAG_TEST_AND_SET(ADDRSPACE)                       \
  _CLC_OVERLOAD _CLC_DEF bool __spirv_AtomicFlagTestAndSet(                    \
      ADDRSPACE int *Ptr, int Scope, int Semantics) {                          \
    return __clc_atomic_flag_test_and_set(                                     \
        Ptr, __spirv_get_clang_memory_order(Semantics),                        \
        __spirv_get_clang_memory_scope(Scope));                                \
  }

__CLC_DEFINE_ATOMIC_FLAG_TEST_AND_SET(global)
__CLC_DEFINE_ATOMIC_FLAG_TEST_AND_SET(local)
#if _CLC_GENERIC_AS_SUPPORTED
__CLC_DEFINE_ATOMIC_FLAG_TEST_AND_SET()
#endif
