//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#define FDECL(TYPE, PREFIX, AS, BYTE_SIZE, MEM_ORDER)                          \
  TYPE __clc__atomic_##PREFIX##load_##AS##_##BYTE_SIZE##_##MEM_ORDER(          \
      AS const TYPE *);

#define IMPL(TYPE, AS, PREFIX, BYTE_SIZE)                                      \
  FDECL(TYPE, PREFIX, AS, BYTE_SIZE, unordered)                                \
  FDECL(TYPE, PREFIX, AS, BYTE_SIZE, acquire)                                  \
  FDECL(TYPE, PREFIX, AS, BYTE_SIZE, seq_cst)                                  \
  _CLC_OVERLOAD _CLC_DECL TYPE __spirv_AtomicLoad(AS TYPE *p, int scope,       \
                                                  int semantics) {             \
    if (semantics & Acquire) {                                                 \
      return __clc__atomic_##PREFIX##load_##AS##_##BYTE_SIZE##_acquire(p);     \
    }                                                                          \
    if (semantics & SequentiallyConsistent) {                                  \
      return __clc__atomic_##PREFIX##load_##AS##_##BYTE_SIZE##_seq_cst(p);     \
    }                                                                          \
    return __clc__atomic_##PREFIX##load_##AS##_##BYTE_SIZE##_unordered(p);     \
  }

#define IMPL_AS(TYPE, PREFIX, BYTE_SIZE)                                       \
  IMPL(TYPE, global, PREFIX, BYTE_SIZE)                                        \
  IMPL(TYPE, local, PREFIX, BYTE_SIZE)

IMPL_AS(int, , 4)

#ifdef cl_khr_int64_base_atomics
IMPL_AS(long, , 8)
#endif

#if _CLC_GENERIC_AS_SUPPORTED

#define IMPL_GENERIC(TYPE, PREFIX, BYTE_SIZE) IMPL(TYPE, , PREFIX, BYTE_SIZE)

IMPL_GENERIC(int, , 4)

#ifdef cl_khr_int64_base_atomics
IMPL_GENERIC(long, , 8)
#endif

#endif //_CLC_GENERIC_AS_SUPPORTED

#undef FDECL
#undef IMPL_AS
#undef IMPL
