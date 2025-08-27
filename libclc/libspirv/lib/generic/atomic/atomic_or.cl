//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#define IMPL(TYPE, AS, FN_NAME)                                                \
  _CLC_OVERLOAD _CLC_DECL TYPE __spirv_AtomicOr(AS TYPE *p, int scope,         \
                                                int semantics, TYPE val) {     \
    return FN_NAME(p, val);                                                    \
  }

IMPL(int, global, __sync_fetch_and_or)
IMPL(unsigned int, global, __sync_fetch_and_or)
IMPL(int, local, __sync_fetch_and_or)
IMPL(unsigned int, local, __sync_fetch_and_or)

#ifdef cl_khr_int64_extended_atomics
IMPL(long, global, __sync_fetch_and_or_8)
IMPL(unsigned long, global, __sync_fetch_and_or_8)
IMPL(long, local, __sync_fetch_and_or_8)
IMPL(unsigned long, local, __sync_fetch_and_or_8)
#endif

#if _CLC_GENERIC_AS_SUPPORTED

#define IMPL_GENERIC(TYPE, FN_NAME) IMPL(TYPE, , FN_NAME)

IMPL_GENERIC(int, __sync_fetch_and_or)
IMPL_GENERIC(unsigned int, __sync_fetch_and_or)

#ifdef cl_khr_int64_base_atomics
IMPL_GENERIC(long, __sync_fetch_and_or_8)
IMPL_GENERIC(unsigned long, __sync_fetch_and_or_8)
#endif

#endif //_CLC_GENERIC_AS_SUPPORTED

#undef IMPL
