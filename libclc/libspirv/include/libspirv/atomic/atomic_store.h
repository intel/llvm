//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define DECL(TYPE, AS)                                                         \
  _CLC_OVERLOAD _CLC_DECL void __spirv_AtomicStore(                            \
      AS TYPE *, int Scope, int MemorySemanticsMask, TYPE);

#define DECL_AS(TYPE)                                                          \
  DECL(TYPE, global)                                                           \
  DECL(TYPE, local)

DECL_AS(int)
DECL_AS(unsigned int)

#ifdef cl_khr_int64_base_atomics
DECL_AS(long)
DECL_AS(unsigned long)
#endif

#undef DECL_AS
#undef DECL
