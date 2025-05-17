//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>
#include <libspirv/spirv.h>

#ifdef cl_khr_int64_base_atomics

#define IMPL(TYPE, AS)                                                         \
  _CLC_OVERLOAD _CLC_DEF TYPE atom_cmpxchg(volatile AS TYPE *p, TYPE cmp,      \
                                           TYPE val) {                         \
    return __spirv_AtomicCompareExchange((AS TYPE *)p, Device,                 \
                                         SequentiallyConsistent,               \
                                         SequentiallyConsistent, cmp, val);    \
  }

IMPL(long, global)
IMPL(unsigned long, global)
IMPL(long, local)
IMPL(unsigned long, local)
#undef IMPL

#endif
