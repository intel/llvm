//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

// TODO: Stop manually mangling this name. Need C++ namespaces to get the exact mangling.

#define IMPL(TYPE, TYPE_MANGLED, AS, AS_MANGLED, NAME, PREFIX, SUFFIX)                                             \
  _CLC_DEF TYPE                                                                                                    \
      _Z18##NAME##PU3##AS_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE##TYPE_MANGLED( \
          volatile AS TYPE *p, enum Scope scope,                                                                   \
          enum MemorySemanticsMask semantics, TYPE val) {                                                          \
    return PREFIX##__sync_fetch_and_##SUFFIX(p, val);                                                              \
  }

IMPL(int, i, global, AS1, __spirv_AtomicSMax, , max)
IMPL(unsigned int, j, global, AS1, __spirv_AtomicUMax, , umax)
IMPL(int, i, local, AS3, __spirv_AtomicSMax, , max)
IMPL(unsigned int, j, local, AS3, __spirv_AtomicUMax, , umax)

#ifdef cl_khr_int64_extended_atomics
unsigned long __clc__sync_fetch_and_max_local_8(volatile local long *, long);
unsigned long __clc__sync_fetch_and_max_global_8(volatile global long *, long);
unsigned long __clc__sync_fetch_and_umax_local_8(volatile local unsigned long *, unsigned long);
unsigned long __clc__sync_fetch_and_umax_global_8(volatile global unsigned long *, unsigned long);

IMPL(long, l, global, AS1, __spirv_AtomicSMax, __clc, max_global_8)
IMPL(unsigned long, m, global, AS1, __spirv_AtomicUMax, __clc, umax_global_8)
IMPL(long, l, local, AS3, __spirv_AtomicSMax, __clc, max_local_8)
IMPL(unsigned long, m, local, AS3, __spirv_AtomicUMax, __clc, umax_local_8)
#endif
#undef IMPL
