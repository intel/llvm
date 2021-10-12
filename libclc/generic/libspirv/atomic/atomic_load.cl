//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

// TODO: Stop manually mangling this name. Need C++ namespaces to get the exact mangling.

#define FDECL(TYPE, PREFIX, AS, BYTE_SIZE, MEM_ORDER) \
TYPE __clc__atomic_##PREFIX##load_##AS##_##BYTE_SIZE##_##MEM_ORDER(volatile AS const TYPE *);

#define IMPL(TYPE, TYPE_MANGLED, AS, AS_MANGLED, PREFIX, BYTE_SIZE)                                               \
  FDECL(TYPE, PREFIX, AS, BYTE_SIZE, unordered)                                                                   \
  FDECL(TYPE, PREFIX, AS, BYTE_SIZE, acquire)                                                                     \
  FDECL(TYPE, PREFIX, AS, BYTE_SIZE, seq_cst)                                                                     \
  _CLC_DEF TYPE                                                                                                   \
      _Z18__spirv_AtomicLoadPU3##AS_MANGLED##K##TYPE_MANGLED##N5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE( \
          volatile AS const TYPE *p, enum Scope scope,                                                            \
          enum MemorySemanticsMask semantics) {                                                                   \
    if (semantics & Acquire) {                                                                                    \
      return __clc__atomic_##PREFIX##load_##AS##_##BYTE_SIZE##_acquire(p);                                        \
    }                                                                                                             \
    if (semantics & SequentiallyConsistent) {                                                                     \
      return __clc__atomic_##PREFIX##load_##AS##_##BYTE_SIZE##_seq_cst(p);                                        \
    }                                                                                                             \
    return __clc__atomic_##PREFIX##load_##AS##_##BYTE_SIZE##_unordered(p);                                        \
  }

#define IMPL_AS(TYPE, TYPE_MANGLED, PREFIX, BYTE_SIZE) \
IMPL(TYPE, TYPE_MANGLED, global, AS1, PREFIX, BYTE_SIZE) \
IMPL(TYPE, TYPE_MANGLED, local, AS3, PREFIX, BYTE_SIZE)

IMPL_AS(int, i, , 4)
IMPL_AS(unsigned int, j, u, 4)

#ifdef cl_khr_int64_base_atomics
IMPL_AS(long, l, , 8)
IMPL_AS(unsigned long, m, u, 8)
#endif

#undef FDECL
#undef IMPL_AS
#undef IMPL
