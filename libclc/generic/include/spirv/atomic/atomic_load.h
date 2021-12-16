//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO: Stop manually mangling this name. Need C++ namespaces to get the exact mangling.
#define DECL(TYPE, TYPE_MANGLED, AS_PREFIX, AS, AS_MANGLED)                                                                \
  _CLC_DECL TYPE                                                                                                           \
      _Z18__spirv_AtomicLoadP##AS_PREFIX##AS_MANGLED##K##TYPE_MANGLED##N5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE( \
          volatile AS const TYPE *, enum Scope, enum MemorySemanticsMask);

#define DECL_AS(TYPE, TYPE_MANGLED)                                            \
  DECL(TYPE, TYPE_MANGLED, U3, global, AS1)                                    \
  DECL(TYPE, TYPE_MANGLED, U3, local, AS3)                                     \
  DECL(TYPE, TYPE_MANGLED, , , )

DECL_AS(int, i)
DECL_AS(unsigned int, j)

#ifdef cl_khr_int64_base_atomics
DECL_AS(long, l)
DECL_AS(unsigned long, m)
#endif

#undef DECL_AS
#undef DECL
