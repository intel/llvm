//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

// TODO: Stop manually mangling this name. Need C++ namespaces to get the exact mangling.

#define IMPL(TYPE, TYPE_MANGLED, AS, AS_MANGLED, SUB, FN_NAME)                                                                   \
  _CLC_DEF TYPE                                                                                                             \
      _Z17__spirv_AtomicXorP##AS_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagENS##SUB##_19MemorySemanticsMask4FlagE##TYPE_MANGLED( \
          volatile AS TYPE *p, enum Scope scope,                                                                            \
          enum MemorySemanticsMask semantics, TYPE val) {                                                                   \
    return FN_NAME(p, val);                                                                                                 \
  }

IMPL(int, i, global, U3AS1, 1, __sync_fetch_and_xor)
IMPL(unsigned int, j, global, U3AS1, 1, __sync_fetch_and_xor)
IMPL(int, i, local, U3AS3, 1, __sync_fetch_and_xor)
IMPL(unsigned int, j, local, U3AS3, 1, __sync_fetch_and_xor)

#ifdef cl_khr_int64_extended_atomics
IMPL(long, l, global, U3AS1, 1, __sync_fetch_and_xor_8)
IMPL(unsigned long, m, global, U3AS1, 1, __sync_fetch_and_xor_8)
IMPL(long, l, local, U3AS3, 1, __sync_fetch_and_xor_8)
IMPL(unsigned long, m, local, U3AS3, 1, __sync_fetch_and_xor_8)
#endif

#if _CLC_GENERIC_AS_SUPPORTED

#define IMPL_GENERIC(TYPE, TYPE_MANGLED, FN_NAME) \
  IMPL(TYPE, TYPE_MANGLED, , , 0, FN_NAME)

IMPL_GENERIC(int, i, __sync_fetch_and_xor)
IMPL_GENERIC(unsigned int, j, __sync_fetch_and_xor)

#ifdef cl_khr_int64_base_atomics
IMPL_GENERIC(long, l, __sync_fetch_and_xor_8)
IMPL_GENERIC(unsigned long, m, __sync_fetch_and_xor_8)
#endif

#endif //_CLC_GENERIC_AS_SUPPORTED
#undef IMPL
