//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

// TODO: Stop manually mangling this name. Need C++ namespaces to get the exact mangling.

_CLC_DEF void
_Z19__spirv_AtomicStorePU3AS1fN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagEf(
    volatile global float *p, enum Scope scope,
    enum MemorySemanticsMask semantics, float val) {
  _Z19__spirv_AtomicStorePU3AS1jN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagEj(
      (volatile global uint *)p, scope, semantics, as_uint(val));
}

_CLC_DEF void
_Z19__spirv_AtomicStorePU3AS3fN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagEf(
    volatile local float *p, enum Scope scope,
    enum MemorySemanticsMask semantics, float val) {
  _Z19__spirv_AtomicStorePU3AS3jN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagEj(
      (volatile local uint *)p, scope, semantics, as_uint(val));
}

#define FDECL(TYPE, PREFIX, AS, BYTE_SIZE, MEM_ORDER) \
TYPE __clc__atomic_##PREFIX##store_##AS##_##BYTE_SIZE##_##MEM_ORDER(volatile AS const TYPE *, TYPE);

#define IMPL(TYPE, TYPE_MANGLED, AS, AS_MANGLED, SUB, PREFIX, BYTE_SIZE)                                                          \
  FDECL(TYPE, PREFIX, AS, BYTE_SIZE, unordered)                                                                                   \
  FDECL(TYPE, PREFIX, AS, BYTE_SIZE, release)                                                                                     \
  FDECL(TYPE, PREFIX, AS, BYTE_SIZE, seq_cst)                                                                                     \
  _CLC_DEF void                                                                                                                   \
      _Z19__spirv_AtomicStoreP##AS_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagENS##SUB##_19MemorySemanticsMask4FlagE##TYPE_MANGLED( \
          volatile AS TYPE *p, enum Scope scope,                                                                                  \
          enum MemorySemanticsMask semantics, TYPE val) {                                                                         \
    if (semantics == Release) {                                                                                                   \
      __clc__atomic_##PREFIX##store_##AS##_##BYTE_SIZE##_release(p, val);                                                         \
    } else if (semantics == SequentiallyConsistent) {                                                                             \
      __clc__atomic_##PREFIX##store_##AS##_##BYTE_SIZE##_seq_cst(p, val);                                                         \
    } else {                                                                                                                      \
      __clc__atomic_##PREFIX##store_##AS##_##BYTE_SIZE##_unordered(p, val);                                                       \
    }                                                                                                                             \
  }

#define IMPL_AS(TYPE, TYPE_MANGLED, PREFIX, BYTE_SIZE)                         \
  IMPL(TYPE, TYPE_MANGLED, global, U3AS1, 1, PREFIX, BYTE_SIZE)                \
  IMPL(TYPE, TYPE_MANGLED, local, U3AS3, 1, PREFIX, BYTE_SIZE)

IMPL_AS(int, i, , 4)
IMPL_AS(unsigned int, j, u, 4)

#ifdef cl_khr_int64_base_atomics
IMPL_AS(long, l, , 8)
IMPL_AS(unsigned long, m, u, 8)
#endif

#if _CLC_GENERIC_AS_SUPPORTED

#define IMPL_GENERIC(TYPE, TYPE_MANGLED, PREFIX, BYTE_SIZE)                    \
  IMPL(TYPE, TYPE_MANGLED, , , 0, PREFIX, BYTE_SIZE)

IMPL_GENERIC(int, i, , 4)
IMPL_GENERIC(unsigned int, j, u, 4)

#ifdef cl_khr_int64_base_atomics
IMPL_GENERIC(long, l, , 8)
IMPL_GENERIC(unsigned long, m, u, 8)
#endif

#endif //_CLC_GENERIC_AS_SUPPORTED

#undef FDECL
#undef IMPL_AS
#undef IMPL
